"""Portable RoboTwin full FT with the user-required adapt_to_pi=False.

The caller must put this experiment's openpi_snapshot/src first on PYTHONPATH.
Raw dataset samples already contain CHW ``images``, 14D ``state``, a 50x14
absolute ``actions`` chunk and ``prompt``; they are not LeRobot Hub records.
"""

from __future__ import annotations

from pathlib import Path

import flax.nnx as nnx

from openpi import transforms
from openpi.models import pi0_config
from openpi.training import config, optimizer, weight_loaders


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAME = "pi05_robotwin_clean_randomized_false_60k"
ASSET_ID = "robotwin_clean_randomized_27500"
ACTION_HORIZON = 50
GLOBAL_BATCH_SIZE = 64
NUM_TRAIN_STEPS = 60_000
DELTA_MASK = (True,) * 6 + (False,) + (True,) * 6 + (False,)


def build_config(
    *,
    run_root: Path | str = EXPERIMENT_ROOT,
    fsdp_devices: int = 4,
    resume: bool = False,
    exp_name: str = "run_v1",
    num_workers: int = 8,
) -> config.TrainConfig:
    """Build the full JAX fine-tuning config without modifying shared registries."""
    root = Path(run_root).resolve()
    if fsdp_devices not in (2, 3, 4):
        raise ValueError("This experiment uses two, three or four local 48GB GPUs.")
    return config.TrainConfig(
        name=CONFIG_NAME,
        project_name="pi05_robotwin_docker_false_60k",
        exp_name=exp_name,
        model=pi0_config.Pi0Config(
            pi05=True,
            dtype="bfloat16",
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            action_dim=32,
            action_horizon=ACTION_HORIZON,
            max_token_len=200,
            discrete_state_input=True,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(root / "checkpoints" / "pi05_base" / "params")
        ),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=2.5e-5,
            decay_steps=NUM_TRAIN_STEPS,
            decay_lr=2.5e-6,
        ),
        optimizer=optimizer.AdamW(
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=1e-10,
            clip_gradient_norm=1.0,
        ),
        ema_decay=0.99,
        freeze_filter=nnx.Nothing,
        data=config.LeRobotAlohaDataConfig(
            repo_id=ASSET_ID,
            assets=config.AssetsConfig(assets_dir=str(root / "assets"), asset_id=ASSET_ID),
            use_delta_joint_actions=True,
            adapt_to_pi=False,
            repack_transforms=transforms.Group(),
            action_sequence_keys=("actions",),
            base_config=config.DataConfig(prompt_from_task=False),
        ),
        assets_base_dir=str(root / "assets"),
        checkpoint_base_dir=str(root / "output" / "checkpoints"),
        seed=42,
        batch_size=GLOBAL_BATCH_SIZE,
        num_workers=num_workers,
        num_train_steps=NUM_TRAIN_STEPS,
        log_interval=10,
        save_interval=1_000,
        keep_period=5_000,
        overwrite=False,
        resume=resume,
        wandb_enabled=False,
        fsdp_devices=fsdp_devices,
        policy_metadata={
            "dataset": ASSET_ID,
            "framework": "jax",
            "training_source": "raw RoboTwin clean + randomized HDF5",
            "adapt_to_pi": False,
            "delta_joint_mask": list(DELTA_MASK),
            "paper": "https://arxiv.org/abs/2603.22078",
        },
    )
