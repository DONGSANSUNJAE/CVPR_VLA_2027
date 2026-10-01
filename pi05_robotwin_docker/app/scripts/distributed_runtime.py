"""Explicit multi-host JAX startup and native-equivalent train-state creation.

Importing this module never initializes JAX. DataLoader subprocesses may import
the runner safely. A launcher must supply one shared, unique PI05_RUN_TOKEN and
PI05_WORLD_SIZE; each Slurm task supplies SLURM_PROCID and sees one allocated GPU.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import socket
import time
from typing import Any, Mapping


@dataclass(frozen=True)
class DistributedContext:
    world_size: int
    rank: int
    run_token: str | None
    coordinator_address: str | None


_CONTEXT: DistributedContext | None = None
_PROXY_VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy",
)


def _request(environ: Mapping[str, str]) -> tuple[int, int, str | None]:
    world = int(environ.get("PI05_WORLD_SIZE", "1"))
    rank = int(environ.get("SLURM_PROCID", "0"))
    if world < 1 or not 0 <= rank < world:
        raise ValueError(f"Invalid distributed world_size={world}, rank={rank}")
    token = environ.get("PI05_RUN_TOKEN")
    if world > 1 and (not token or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", token)
                      or token in (".", "..")):
        raise ValueError("Multi-host launch requires a unique, path-safe PI05_RUN_TOKEN")
    return world, rank, token


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def is_primary() -> bool:
    """Rank check that does not query (and therefore initialize) a JAX backend."""
    return (_CONTEXT.rank if _CONTEXT else int(os.environ.get("SLURM_PROCID", "0"))) == 0


def initialize_distributed(root: Path | str, *, timeout: int = 600) -> DistributedContext:
    """Initialize once, before any device discovery or JAX array construction.

    Rank 0 publishes its reachable IPv4 coordinator address on the shared file
    system. A unique run token prevents another launch's rendezvous from being
    reused. There is deliberately no automatic initialization on module import.
    """
    global _CONTEXT
    world, rank, token = _request(os.environ)
    if _CONTEXT is not None:
        if (_CONTEXT.world_size, _CONTEXT.rank, _CONTEXT.run_token) != (world, rank, token):
            raise RuntimeError("Distributed environment changed after initialization")
        return _CONTEXT
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    if world == 1:
        _CONTEXT = DistributedContext(world, rank, token, None)
        return _CONTEXT
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    cpu_only = os.environ.get("JAX_PLATFORMS") == "cpu"
    if not cpu_only and visible is not None and len([part for part in visible.split(",") if part.strip()]) != 1:
        raise ValueError(f"Each distributed rank must see exactly one allocated GPU; CUDA_VISIBLE_DEVICES={visible!r}")

    directory = Path(root) / "logs" / "distributed" / str(token)
    directory.mkdir(parents=True, exist_ok=True)
    coordinator_file = directory / "coordinator.json"
    if rank == 0:
        if coordinator_file.exists():
            raise FileExistsError(f"Run token has already been used: {coordinator_file}")
        host = os.environ.get("PI05_COORDINATOR_HOST", socket.gethostname())
        address = socket.gethostbyname(host)
        if address.startswith("127.") and os.environ.get("PI05_ALLOW_LOOPBACK") != "1":
            raise ValueError(f"Coordinator {host!r} resolves to loopback; remote ranks cannot reach it")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.bind((address, 0))
            port = reservation.getsockname()[1]
        _atomic_json(coordinator_file, {
            "run_token": token, "world_size": world, "hostname": socket.gethostname(),
            "ipv4": address, "port": port, "coordinator_address": f"{address}:{port}",
        })
    deadline = time.monotonic() + timeout
    while not coordinator_file.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Rank {rank} timed out waiting for {coordinator_file}")
        time.sleep(0.2)
    coordinator = json.loads(coordinator_file.read_text())
    if coordinator["run_token"] != token or coordinator["world_size"] != world:
        raise ValueError("Rendezvous record does not match this distributed launch")

    # Internal gRPC coordination must not be routed through outbound web proxies.
    # Never print their values; those URLs can contain credentials.
    removed_proxies = [name for name in _PROXY_VARIABLES if os.environ.pop(name, None) is not None]
    import jax
    try:
        jax.distributed.initialize(
            coordinator_address=coordinator["coordinator_address"],
            num_processes=world,
            process_id=rank,
            local_device_ids=[0],
            initialization_timeout=timeout,
            coordinator_bind_address=f"0.0.0.0:{coordinator['port']}",
        )
        if jax.process_count() != world or jax.process_index() != rank or jax.local_device_count() != 1:
            raise RuntimeError("Distributed JAX topology differs from one allocated device per rank")
    except BaseException as error:
        _atomic_json(directory / f"rank_{rank}_initialization.json", {
            "status": "failed", "rank": rank, "exception": type(error).__name__, "message": str(error),
        })
        raise
    _CONTEXT = DistributedContext(world, rank, token, coordinator["coordinator_address"])
    _atomic_json(directory / f"rank_{rank}_initialization.json", {
        "status": "initialized", "rank": rank, "world_size": world,
        "hostname": socket.gethostname(), "jax_version": jax.__version__,
        "local_devices": [str(device) for device in jax.local_devices()],
        "global_device_count": jax.device_count(), "removed_proxy_variable_names": removed_proxies,
    })
    return _CONTEXT


def global_replicate(tree: Any, mesh: Any) -> Any:
    """Create global replicated arrays from identical process-local values.

    Plain device_put cannot target remote, non-addressable devices. This API
    supplies each process's local replica to a global array, including typed
    PRNG keys used by the native model initialization and training update.
    """
    import jax
    import numpy as np
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    def convert(value):
        if isinstance(value, jax.Array) and value.sharding.is_equivalent_to(replicated, value.ndim):
            return value
        if hasattr(value, "dtype") and jax.dtypes.issubdtype(value.dtype, jax.dtypes.prng_key):
            key_impl = jax.random.key_impl(value)
            data = convert(jax.random.key_data(value))
            return jax.random.wrap_key_data(data, impl=key_impl)
        if isinstance(value, jax.Array) and not value.is_fully_addressable:
            if not value.is_fully_replicated:
                raise ValueError("Cannot replicate a partial process-local slice of a sharded global array")
            value = value.addressable_data(0)
        local = np.asarray(value)
        return jax.make_array_from_process_local_data(replicated, local, global_shape=local.shape)

    return jax.tree.map(convert, tree)


def initialize_checkpoint_manager(config: Any, native_checkpoints: Any) -> tuple[Any, bool]:
    """Avoid the native exists/mkdir race while creating an Orbax manager on all ranks."""
    import jax
    from jax.experimental import multihost_utils

    directory = Path(config.checkpoint_dir)
    error = None
    if is_primary():
        try:
            if directory.exists() and not config.resume:
                raise FileExistsError(f"Checkpoint directory already exists; explicit resume required: {directory}")
            if config.overwrite:
                raise ValueError("Distributed training refuses checkpoint overwrite")
            directory.mkdir(parents=True, exist_ok=True)
        except Exception as exception:
            error = f"{type(exception).__name__}: {exception}"
    # Propagate the policy decision before entering Orbax collectives. Every rank
    # must either fail together or construct its own manager.
    import numpy as np
    failed = multihost_utils.broadcast_one_to_all(np.asarray(error is not None, dtype=np.int32))
    if bool(failed):
        raise RuntimeError(error or "Rank 0 rejected checkpoint directory initialization")
    multihost_utils.sync_global_devices("pi05_checkpoint_directory_ready")
    manager, resuming = native_checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=False, resume=True,
    )
    return manager, resuming


def init_train_state(config: Any, init_rng: Any, mesh: Any, *, resume: bool, native_train: Any, sharding_fn=None, shard_initial_weights=False) -> tuple[Any, Any]:
    """Native initialization math with explicitly global replicated input buffers.

    The optimizer, model creation, freeze filter, EMA state, validation and FSDP
    partitioning match openpi_snapshot/scripts/train.py. Only the placement of
    input parameters and RNG differs to support non-addressable remote devices.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp
    from openpi.shared import nnx_utils
    from openpi.training import optimizer, sharding, utils

    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng, partial_params=None):
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(
            params, config.freeze_filter, lambda parameter: parameter.replace(parameter.value.astype(jnp.bfloat16)),
        )
        return utils.TrainState(
            step=0, params=params, model_def=nnx.graphdef(model), tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay, ema_params=None if config.ema_decay is None else params,
        )

    init_rng = global_replicate(init_rng, mesh)
    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = (sharding_fn or sharding.fsdp_sharding)(train_state_shape, mesh, log=is_primary())
    if resume:
        # Give Orbax the requested layout explicitly. Otherwise its native
        # fallback reads device topology from the saved checkpoint, which need
        # not match the hosts assigned by a later Slurm allocation.
        train_state_shape = jax.tree.map(
            lambda shape, layout: jax.ShapeDtypeStruct(
                shape.shape, shape.dtype, sharding=layout, weak_type=shape.weak_type,
            ),
            train_state_shape, state_sharding,
        )
        return train_state_shape, state_sharding
    partial_params = native_train._load_weights_and_validate(
        config.weight_loader, train_state_shape.params.to_pure_dict(),
    )
    if shard_initial_weights:
        parameter_layout = state_sharding.params.to_pure_dict()
        partial_params = jax.device_put(partial_params, parameter_layout)
    else:
        partial_params = global_replicate(partial_params, mesh)
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    state = jax.jit(
        init, donate_argnums=(1,), in_shardings=(replicated, parameter_layout) if shard_initial_weights else replicated, out_shardings=state_sharding,
    )(init_rng, partial_params)
    return state, state_sharding
