"""One-container training, validation, graceful stop and same-recipe resume.

`idle` is a separate, payload-free mode for interactive remote-workspace
platforms (e.g. a VESSL Cloud custom workspace image): it only starts sshd
and otherwise does nothing, so the container stays up with no data, base
weights or GPUs mounted and a user can SSH in and drive training by hand.
It never touches /opt/pi05/data, /opt/pi05/manifests or nvidia-smi.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path('/opt/pi05')


def run_idle():
    """Start sshd in the foreground; used only for interactive workspace images."""
    Path('/run/sshd').mkdir(parents=True, exist_ok=True)
    ssh_dir = Path('/root/.ssh')
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(ssh_dir, 0o700)
    subprocess.run(['ssh-keygen', '-A'], check=True)
    print('PI05_IDLE: sshd starting on :22; no payload/GPU checks run in this mode', flush=True)
    os.execv('/usr/sbin/sshd', ['/usr/sbin/sshd', '-D', '-e'])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    if mode not in {'train', 'preflight', 'gpu-check', 'idle'}:
        raise SystemExit('Usage: train | preflight | gpu-check | idle')
    if mode == 'idle':
        run_idle()
        return
    for name in ('logs', 'manifests', 'cache'):
        (ROOT / 'output' / name).mkdir(parents=True, exist_ok=True)
    for source in (ROOT / 'frozen_manifests').iterdir():
        if source.is_file():
            target = ROOT / 'manifests' / source.name
            if target.exists() and target.read_bytes() != source.read_bytes():
                raise ValueError(f'Output belongs to a different frozen recipe: {target}')
            if not target.exists():
                shutil.copy2(source, target)
    if not (ROOT / 'data/index.json').exists():
        raise FileNotFoundError('Restore payload using tools/restore_payload.py first')
    if mode == 'preflight':
        os.environ['JAX_PLATFORMS'] = 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.execv(sys.executable, [sys.executable, 'scripts/train_robotwin_impl.py', '--preflight-only'])
    os.environ['JAX_PLATFORMS'] = 'cuda'
    os.environ['PI05_WORLD_SIZE'] = '1'
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,uuid,driver_version',
        '--format=csv,noheader,nounits'], text=True)
    rows = [r.split(',') for r in inventory.strip().splitlines()]
    expected = int(os.environ.get('PI05_EXPECTED_GPUS', '4'))
    if expected != 4 or len(rows) != 4:
        raise RuntimeError('This recipe requires exactly four visible GPUs on one host')
    minimum = int(os.environ.get('PI05_MIN_GPU_MIB', '75000'))
    if minimum < 75000 or any(int(r[1]) < minimum or 'A100' not in r[0] for r in rows):
        raise RuntimeError('Expected four A100 80GB GPUs; received:\n' + inventory)
    (ROOT / 'logs/allocated_gpu.txt').write_text(inventory)
    topology = subprocess.run(['nvidia-smi', 'topo', '-m'], capture_output=True, text=True)
    (ROOT / 'logs/gpu_topology.txt').write_text(topology.stdout + topology.stderr)
    if mode == 'gpu-check':
        from train_robotwin_impl import validate_gpu_runtime
        from cuda_runtime import preload_cuda_libraries
        preload_cuda_libraries(ROOT)
        validate_gpu_runtime()
        print('GPU_CHECK_PASSED', flush=True)
        return
    plan = dict(global_batch_size=64, microbatch_size=4, accumulation_steps=16,
                fsdp_devices=4, adapt_to_pi=False, target_updates=60000)
    path = ROOT / 'manifests/runtime_plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Refusing to resume with a changed batch/sharding/adapter recipe')
    path.write_text(json.dumps(plan, indent=2) + '\n')
    os.execv(sys.executable, [sys.executable, '-u', 'scripts/train_robotwin_impl.py',
        '--resume', '--fsdp-devices', '4', '--microbatch-size', '4',
        '--num-workers', os.environ.get('PI05_NUM_WORKERS', '8')])


if __name__ == '__main__':
    main()
