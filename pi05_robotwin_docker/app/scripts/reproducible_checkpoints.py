"""Add the frozen replay bundle and run provenance to native checkpoint assets."""
import hashlib
import json
from pathlib import Path
import shutil

import jax
from openpi.training import checkpoints


def install_checkpoint_provenance(root):
    root = Path(root)
    original = checkpoints.CallbackHandler.save
    if getattr(original, '_pi05_reproduction', False):
        return

    def save(self, directory, args):
        original(self, directory, args)
        if jax.process_index() != 0:
            return
        target = Path(directory) / 'reproduction'
        target.mkdir(parents=True, exist_ok=True)
        bundle = root / 'manifests/reproduction_bundle.tar.gz'
        manifest = root / 'manifests/reproduction_bundle.json'
        if not bundle.exists() or not manifest.exists():
            raise FileNotFoundError('Frozen reproduction bundle is required before checkpointing')
        description = json.loads(manifest.read_text())
        if hashlib.sha256(bundle.read_bytes()).hexdigest() != description['sha256']:
            raise ValueError('Reproduction bundle changed after freezing')
        shutil.copy2(bundle, target / bundle.name)
        shutil.copy2(manifest, target / manifest.name)
        for relative in ['manifests/effective_train_config.json', 'manifests/run_sources.json',
                         'manifests/submission.json', 'manifests/gpu_health.json',
                         'manifests/runtime_plan.json', 'manifests/compute_layout_validation.json',
                         'runtime_pro/inventory.json', 'manifests/runtime_recovery.json',
                         'manifests/experiment_origin.json', 'manifests/a6000_memory_audit.json',
                         'manifests/data_recovery_verified.json',
                         'manifests/reproduction_recovery_verified.json',
                         'assets/robotwin_clean_randomized_27500/norm_stats_provenance.json']:
            source = root / relative
            if source.exists():
                shutil.copy2(source, target / source.name)
        (target / 'REPLAY.txt').write_text(
            'Extract reproduction_bundle.tar.gz in a new directory.\n'
            'Use its pinned runtime and scripts/reproduce_checkpoint.py with this committed checkpoint.\n'
            'Fixed inputs are training examples; offline action replay is not held-out task success.\n'
            'To resume training, restore train_state/params and reconstruct the epoch permutation\n'
            'from saved optimizer step, global batch 64, and seed 42 (StepSampler).\n')

    save._pi05_reproduction = True
    checkpoints.CallbackHandler.save = save
