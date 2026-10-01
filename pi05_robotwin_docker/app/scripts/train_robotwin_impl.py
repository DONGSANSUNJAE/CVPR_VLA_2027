"""Isolated JAX runner using the paper model and accumulated batch64 optimizer updates.

Integration differences: raw HDF5 factory, matching current data-loader API,
completed-update checkpoint names, local metrics, finite checks and stop saving.
"""
from __future__ import annotations

import argparse
import dataclasses
from datetime import datetime, timezone
import functools
import gc
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'openpi_snapshot/src'))
sys.path.insert(0, str(ROOT / 'openpi_snapshot/packages/openpi-client/src'))

import jax
import jax.numpy as jnp
import numpy as np

from configs.train_config import ASSET_ID, build_config
from robotwin_dataset import RobotwinRawDataset, read_index
from runtime_compat import install_runtime_compat
from resumable_sampler import StepSampler
from distributed_data_loader import DistributedTorchDataLoader
import distributed_runtime
import three_gpu_sharding
from cuda_runtime import preload_cuda_libraries
from accumulating_train_step_uneven import accumulating_train_step
from reproducible_checkpoints import install_checkpoint_provenance
from openpi.training import checkpoints, data_loader, sharding

spec = importlib.util.spec_from_file_location('paper_train', ROOT / 'openpi_snapshot/scripts/train.py')
paper_train = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = paper_train
spec.loader.exec_module(paper_train)
STOP_REQUESTED = False


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)


def status(phase, **values):
    record = dict(phase=phase, time_utc=datetime.now(timezone.utc).isoformat(),
                  slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                  process_rank=int(os.environ.get('SLURM_PROCID', '0')), **values)
    name = 'train_status.json' if distributed_runtime.is_primary() else f'train_rank_{record["process_rank"]}_status.json'
    atomic_json(ROOT / 'logs' / name, record)
    print(json.dumps(record, default=str, allow_nan=False), flush=True)


def validate_inputs():
    frozen = json.loads((ROOT / 'manifests/run_sources.json').read_text())
    for relative, expected in frozen['sha256'].items():
        if sha256(ROOT / relative) != expected:
            raise ValueError(f'Frozen source changed: {relative}')
    index_path = ROOT / 'data/index.json'
    index = read_index(index_path)
    if not index.get('production') or len(index['episodes']) != 27500:
        raise ValueError('Full 27,500-episode production dataset required')
    norm_dir = ROOT / 'assets' / ASSET_ID
    norm_provenance = json.loads((norm_dir / 'norm_stats_provenance.json').read_text())
    if norm_provenance.get('adapt_to_pi') is not False:
        raise ValueError('This portable run requires adapt_to_pi=False normalization')
    checks = [('index_sha256', index_path), ('norm_stats_sha256', norm_dir / 'norm_stats.json'),
              ('numeric_sha256', ROOT / 'data' / index['numeric_path'])]
    for key, path in checks:
        if sha256(path) != norm_provenance[key]:
            raise ValueError(f'Normalization provenance mismatch: {key}')
    if norm_provenance['state_vectors'] != index['total_train_frames'] or norm_provenance['action_horizon'] != 50:
        raise ValueError('Normalization coverage mismatch')
    base = json.loads((ROOT / 'manifests/pi05_base_verified.json').read_text())
    for item in base['verified_files']:
        path = ROOT / 'checkpoints/pi05_base' / item['path']
        if path.stat().st_size != item['size'] or sha256(path) != item['sha256']:
            raise ValueError(f'Pretrained checkpoint integrity mismatch: {path}')
    return index


def create_raw_dataset(data_config, action_horizon, model_config):
    if data_config.repo_id != ASSET_ID:
        raise ValueError('Unexpected dataset ID')
    return RobotwinRawDataset(ROOT / 'data/index.json', action_horizon=action_horizon)


def cpu_worker(worker_id):
    del worker_id
    # Only the parent JAX process owns the allocated GPUs.
    os.environ['JAX_PLATFORMS'] = 'cpu'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    jax.config.update('jax_platforms', 'cpu')
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)


def request_stop(signum, frame):
    del frame
    global STOP_REQUESTED
    STOP_REQUESTED = True
    logging.warning('Signal %s received; will checkpoint after current optimizer update', signum)


def validate_gpu_runtime():
    """Exercise the bf16 GEMM/convolution/backward and device collective paths."""
    @jax.jit
    def compute(x, weight):
        def objective(kernel):
            conv = jax.lax.conv_general_dilated(
                x, kernel, (2, 2), 'VALID', dimension_numbers=('NHWC', 'HWIO', 'NHWC'))
            flat = conv.reshape(-1, conv.shape[-1])
            return jnp.mean((flat.T @ flat).astype(jnp.float32))
        return jax.value_and_grad(objective)(weight)

    for device in jax.local_devices():
        with jax.default_device(device):
            loss, grad = compute(jnp.ones((1, 32, 32, 3), dtype=jnp.bfloat16),
                                 jnp.full((4, 4, 3, 32), 0.01, dtype=jnp.bfloat16))
            if not np.isfinite(np.asarray(loss)) or not np.isfinite(np.asarray(grad)).all():
                raise RuntimeError(f'GPU bf16 forward/backward check failed: {device}')
    mesh = jax.make_mesh((jax.device_count(),), ('health',))
    partitioned = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec('health'))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    # 64 MiB per device exercises the actual cross-host FSDP all-gather path.
    local = np.full((16 * 1024 * 1024 * jax.local_device_count(),), jax.process_index(), dtype=np.float32)
    value = jax.make_array_from_process_local_data(partitioned, local)
    gather = jax.jit(lambda x: x, in_shardings=partitioned, out_shardings=replicated)
    result = gather(value)
    jax.block_until_ready(result)
    times = []
    for _ in range(3):
        begin = time.monotonic()
        result = gather(value)
        jax.block_until_ready(result)
        times.append(time.monotonic() - begin)
    if not np.isclose(np.asarray(jnp.mean(result)), (jax.process_count() - 1) / 2):
        raise RuntimeError('GPU all-gather result check failed')
    if distributed_runtime.is_primary():
        atomic_json(ROOT / 'manifests/gpu_health.json',
                    dict(status='passed', devices=[str(d) for d in jax.devices()],
                         jax_version=jax.__version__, process_count=jax.process_count(),
                         all_gather_input_mib_per_device=64, all_gather_seconds=times,
                         checks=['bf16_gemm', 'convolution', 'gradient', 'jit_sharded_all_gather']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight-only', action='store_true', help='CPU data/transform validation; no weights initialized')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--fsdp-devices', type=int, choices=(2, 3, 4), default=4)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--microbatch-size', type=int, default=1)
    args = parser.parse_args()
    if not args.preflight_only:
        preload_cuda_libraries(ROOT)
    logging.basicConfig(level=logging.INFO)
    paper_train.init_logging()
    if args.distributed:
        distributed_runtime.initialize_distributed(ROOT)
    install_runtime_compat()
    install_checkpoint_provenance(ROOT)
    if not args.preflight_only:
        status('validating_inputs', completed_updates=0)
    index = validate_inputs()
    config = build_config(resume=args.resume, num_workers=args.num_workers, fsdp_devices=args.fsdp_devices)
    data_loader.create_torch_dataset = create_raw_dataset
    data_loader._worker_init_fn = cpu_worker
    if args.preflight_only:
        raw = create_raw_dataset(config.data.create(config.assets_dirs, config.model), 50, config.model)
        transformed = data_loader.transform_dataset(raw, config.data.create(config.assets_dirs, config.model))
        sample_ids = sorted(set([0, len(raw) // 2, len(raw) - 1] +
                                [ep['training_offset'] for ep in index['episodes'][::550]]))
        for sample_id in sample_ids:
            sample = transformed[sample_id]
            if sample['actions'].shape != (50, 32) or not np.isfinite(sample['actions']).all():
                raise ValueError(f'Invalid transformed actions at {sample_id}')
            if sample['state'].shape != (32,) or not np.isfinite(sample['state']).all():
                raise ValueError(f'Invalid transformed state at {sample_id}')
            for image in sample['image'].values():
                if image.shape != (224, 224, 3):
                    raise ValueError(f'Wrong transformed image shape: {image.shape}')
        atomic_json(ROOT / 'manifests/data_preflight.json',
                    dict(status='passed', samples=len(sample_ids), sample_ids=sample_ids,
                         episodes=len(index['episodes']), frames=len(raw),
                         index_sha256=sha256(ROOT / 'data/index.json')))
        print('DATA_PREFLIGHT_PASSED', flush=True)
        return
    expected_processes = int(os.environ.get('PI05_WORLD_SIZE', '1')) if args.distributed else 1
    if jax.process_count() != expected_processes or jax.device_count() != config.fsdp_devices or any(d.platform != 'gpu' for d in jax.devices()):
        raise ValueError(f'Expected {expected_processes} processes with {config.fsdp_devices} total GPUs, got {jax.devices()}')
    validate_gpu_runtime()
    status('initializing', completed_updates=0, devices=[str(d) for d in jax.devices()],
           dataset_episodes=len(index['episodes']), total_train_frames=index['total_train_frames'])
    if distributed_runtime.is_primary():
        atomic_json(ROOT / 'manifests/effective_train_config.json', dataclasses.asdict(config))
    jax.config.update('jax_compilation_cache_dir', str(ROOT / f'output/cache/jax-rank-{jax.process_index()}'))
    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)
    mesh = three_gpu_sharding.make_mesh()
    batch_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_rng = distributed_runtime.global_replicate(train_rng, mesh)
    manager, resuming = distributed_runtime.initialize_checkpoint_manager(config, checkpoints)
    # Never connect to an external tracking service.
    paper_train.init_wandb(config, resuming=resuming, enabled=False)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.transform_dataset(
        create_raw_dataset(data_config, config.model.action_horizon, config.model), data_config)
    sampler = StepSampler(len(dataset), config.batch_size, config.seed,
                          process_count=jax.process_count(), process_index=jax.process_index())
    torch_loader = DistributedTorchDataLoader(
        dataset, config.batch_size // jax.process_count(), sharding=batch_sharding, sampler=sampler,
        num_workers=config.num_workers, seed=config.seed + jax.process_index(), worker_init_fn=cpu_worker)
    loader = data_loader.DataLoaderImpl(data_config, torch_loader)
    status('loading_model', completed_updates=0, resumed=resuming)
    state, state_sharding = distributed_runtime.init_train_state(
        config, init_rng, mesh, resume=resuming, native_train=paper_train,
        sharding_fn=three_gpu_sharding.fsdp_sharding, shard_initial_weights=True)
    jax.block_until_ready(state)
    if resuming:
        prior = config.checkpoint_dir / str(manager.latest_step()) / 'assets/reproduction/effective_train_config.json'
        if not prior.exists() or json.loads(prior.read_text()).get('data', {}).get('adapt_to_pi') is not False:
            raise ValueError('Refusing to resume a checkpoint without False provenance')
        state = checkpoints.restore_state(manager, state, loader)
    step = int(state.step)
    # Initialization and GPU health executables are no longer needed. Release
    # their references before compiling the memory-intensive training update.
    gc.collect()
    jax.clear_caches()
    gc.collect()
    if step >= config.num_train_steps:
        if step != config.num_train_steps:
            raise ValueError('Checkpoint exceeds the frozen target update count')
        manager.wait_until_finished()
        status('complete', completed_updates=step, checkpoint=str(config.checkpoint_dir / str(step)))
        return
    sampler.reset(step)
    status('starting_data_workers', completed_updates=step, workers=config.num_workers)
    iterator = iter(loader)
    batch = next(iterator)
    logging.info('First actual training batch: %s', paper_train.training_utils.array_tree_to_info(batch))
    atomic_json(ROOT / 'logs/memory_before_first_update.json',
                {str(device): device.memory_stats() for device in jax.local_devices()})
    status('compiling_first_update', completed_updates=step, resumed=resuming,
           resume_data_order='epoch permutation seed42+epoch; offset from completed optimizer updates',
           global_batch_size=64, microbatch_size=args.microbatch_size, accumulation_steps=64 // args.microbatch_size)
    update = jax.jit(functools.partial(accumulating_train_step, config, microbatch_size=args.microbatch_size, compute_mesh=mesh),
                     in_shardings=(replicated, state_sharding, batch_sharding),
                     out_shardings=(state_sharding, replicated), donate_argnums=(1,))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    infos = []
    timer = time.monotonic()
    last_saved = None
    while step < config.num_train_steps:
        # Parameter layouts are explicit; replicated microbatches must not use
        # native DATA_AXIS constraints that require batch divisibility by three.
        state, info = update(train_rng, state, batch)
        # Synchronize scalars, so completed_updates never reports queued work.
        host_info = {key: float(value) for key, value in jax.device_get(info).items()}
        if not all(np.isfinite(v) for v in host_info.values()):
            raise FloatingPointError(f'Nonfinite update metrics: {host_info}')
        step += 1
        if step in (1, 2, 3, 10):
            atomic_json(ROOT / f'logs/memory_after_update_{step}.json',
                        {str(device): device.memory_stats() for device in jax.local_devices()})
        from jax.experimental import multihost_utils
        stop_requested = bool(np.any(multihost_utils.process_allgather(np.asarray(STOP_REQUESTED))))
        infos.append(host_info)
        if step in (1, 2, 3) or step % config.log_interval == 0 or stop_requested or step == config.num_train_steps:
            elapsed = time.monotonic() - timer
            record = dict(completed_updates=step, target_updates=config.num_train_steps,
                          learning_rate=float(config.lr_schedule.create()(step - 1)),
                          seconds_per_update=elapsed / len(infos),
                          **{key: float(np.mean([entry[key] for entry in infos])) for key in host_info})
            if distributed_runtime.is_primary():
                with (ROOT / 'logs/metrics.jsonl').open('a') as out:
                    out.write(json.dumps(record, allow_nan=False) + '\n')
            status('training', **record)
            infos = []
            timer = time.monotonic()
        if step == 10 or step % config.save_interval == 0 or step == config.num_train_steps or stop_requested:
            checkpoints.save_state(manager, state, loader, step)
            last_saved = step
        if stop_requested:
            manager.wait_until_finished()
            status('interrupted_checkpoint_saved', completed_updates=step, checkpoint=str(config.checkpoint_dir / str(step)))
            return
        if step < config.num_train_steps:
            batch = next(iterator)
    manager.wait_until_finished()
    if last_saved != config.num_train_steps:
        raise RuntimeError('Final checkpoint was not saved')
    status('complete', completed_updates=step, checkpoint=str(config.checkpoint_dir / str(step)))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Preserve the last known completed step for diagnosis.
        rank = int(os.environ.get('SLURM_PROCID', '0'))
        previous = ROOT / 'logs' / ('train_status.json' if rank == 0 else f'train_rank_{rank}_status.json')
        last = json.loads(previous.read_text()) if previous.exists() else {}
        status('failed', error=repr(exc), last_status=last)
        raise
