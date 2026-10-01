"""CPU-only package/configuration smoke check; no data, weights or GPU required."""
import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path
import sys

os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import cv2
import flax.nnx as nnx
import h5py
import jax
import torch

from configs.train_config import build_config

root = Path(os.environ.get('PI05_APP_ROOT', '/opt/pi05')).resolve()
expected = {
    'jax': '0.6.0', 'jaxlib': '0.6.0', 'jax-cuda12-plugin': '0.6.0',
    'jax-cuda12-pjrt': '0.6.0', 'flax': '0.10.2', 'numpy': '1.26.4',
    'torch': '2.6.0+cpu', 'nvidia-cuda-runtime-cu12': '12.8.90',
}
actual = {name: version(name) for name in expected}
if actual != expected or sys.version_info[:2] != (3, 11):
    raise RuntimeError(f'Unexpected Python/package versions: {sys.version}; {actual}')
cfg = build_config(run_root=root)
checks = {
    'pi05': cfg.model.pi05 is True,
    'full_ft': cfg.freeze_filter is nnx.Nothing,
    'batch64_60k': cfg.batch_size == 64 and cfg.num_train_steps == 60000,
    'four_gpu_config': cfg.fsdp_devices == 4,
    'false_delta': cfg.data.adapt_to_pi is False and cfg.data.use_delta_joint_actions,
    'horizon50': cfg.model.action_horizon == 50,
    'adamw': (cfg.optimizer.b1, cfg.optimizer.b2, cfg.optimizer.clip_gradient_norm) == (0.9, 0.95, 1.0),
    'learning_rate': (cfg.lr_schedule.warmup_steps, cfg.lr_schedule.peak_lr,
                      cfg.lr_schedule.decay_steps, cfg.lr_schedule.decay_lr) == (1000, 2.5e-5, 60000, 2.5e-6),
    'cpu_only': all(d.platform == 'cpu' for d in jax.devices()),
}
asset = root / 'assets/robotwin_clean_randomized_27500'
norm = json.loads((asset / 'norm_stats_provenance.json').read_text())
checks['false_norm'] = norm['adapt_to_pi'] is False and norm['episodes'] == 27500
checks['norm_hash'] = hashlib.sha256((asset / 'norm_stats.json').read_bytes()).hexdigest() == norm['norm_stats_sha256']
if not all(checks.values()):
    raise RuntimeError(f'CPU smoke check failed: {checks}')
print(json.dumps({
    'status': 'passed', 'scope': 'CPU imports, pinned package versions, configuration and normalization',
    'python': sys.version.split()[0], 'packages': actual, 'checks': checks,
    'gpu_training_tested': False, 'dataset_tested': False, 'model_weights_loaded': False,
}, indent=2))
