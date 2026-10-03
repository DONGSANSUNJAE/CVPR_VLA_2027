#!/usr/bin/env bash
# Run inside the prepared VESSL A100 x4 image; no Docker daemon required.
set -euo pipefail
test -x /opt/venv/bin/python
test -f /opt/pi05/entrypoint.py
command -v flock >/dev/null
mkdir -p /HW/pi05_robotwin_false_60k/outputs/logs

nohup bash >> /HW/pi05_robotwin_false_60k/outputs/logs/launch.log 2>&1 <<'RUN' &
set -euo pipefail
exec 9>/tmp/pi05_robotwin_false_60k.lock
flock -n 9 || { echo 'ALREADY_RUNNING: duplicate launch stopped'; exit 1; }
date -Is
cd /opt/pi05
export PATH="/opt/venv/bin:/opt/venv/lib/python3.11/site-packages/nvidia/cuda_nvcc/bin:$PATH"
export PYTHONPATH=/opt/pi05:/opt/pi05/scripts:/opt/pi05/openpi_snapshot/src:/opt/pi05/openpi_snapshot/packages/openpi-client/src
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false JAX_PLATFORMS=cuda
export PI05_TOKENIZER_PATH=/opt/pi05/tokenizer/paligemma_tokenizer.model
export OPENPI_DATA_HOME=/opt/pi05/output/cache/openpi
export XLA_PYTHON_CLIENT_MEM_FRACTION=.90 XLA_PYTHON_CLIENT_PREALLOCATE=true
export PI05_EXPECTED_GPUS=4 PI05_MIN_GPU_MIB=75000 PI05_NUM_WORKERS=8

python - <<'PY'
import json
from pathlib import Path

root = Path('/opt/pi05')
payload = Path('/HW/pi05_robotwin_false_60k/payload')
marker = json.loads((payload / 'PAYLOAD_VERIFIED.json').read_text())
if marker.get('episodes') != 27500 or marker.get('manifest_sha256') != '6bcffd2568eeda728fb491b6750acde6dee2180138043bbc2aacccd577cdc5d6':
    raise SystemExit('Payload completion marker does not match this dataset')
links = {
    root / 'data': payload / 'data',
    root / 'checkpoints/pi05_base': payload / 'base',
    root / 'tokenizer': payload / 'tokenizer',
    root / 'output': Path('/HW/pi05_robotwin_false_60k/outputs'),
}
for link, target in links.items():
    if not target.is_dir():
        raise SystemExit(f'Missing directory: {target}')
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise SystemExit(f'Existing link points elsewhere: {link}')
    elif link.exists() and (not link.is_dir() or any(link.iterdir())):
        raise SystemExit(f'Refusing to replace existing files: {link}')
for link, target in links.items():
    if link.is_symlink():
        continue
    if link.exists():
        link.rmdir()  # Only the empty image directory passed above.
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
print('PAYLOAD_PATHS_READY', flush=True)
PY

echo DATA_PREFLIGHT_START
python -u entrypoint.py preflight
echo GPU_CHECK_START
python -u entrypoint.py gpu-check
echo "TRAIN_START pid=$$"
exec python -u entrypoint.py train
RUN

echo "Launch requested (process $!). Follow the log; this is not proof that training started."
echo 'tail -n 60 -f /HW/pi05_robotwin_false_60k/outputs/logs/launch.log'
