#!/usr/bin/env python3
"""Compute public-openpi RunningStats on all numeric training samples.

No images are opened. The raw-data converter aligns state q[t] with action
q[t+1]; future chunks repeat the final action at episode boundaries, exactly
like the dataset. All T-1 starts per episode and all 50 chunk offsets count.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

# Statistics are a CPU task, including when invoked on a GPU-capable host.
os.environ["JAX_PLATFORMS"] = "cpu"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "openpi_snapshot" / "src"))
sys.path.insert(0, str(ROOT / "openpi_snapshot" / "packages" / "openpi-client" / "src"))

import numpy as np

from openpi.policies import aloha_policy
from openpi.shared import normalize

from configs.train_config import ACTION_HORIZON, ASSET_ID, DELTA_MASK


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def transform_numeric(states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized exact AlohaInputs(adapt_to_pi=False) -> DeltaActions.

    The native state transform indexes a single vector; the formulas below call
    the same native gripper and joint helpers over a batch. Inputs are copied so
    the memory-mapped raw data remains immutable.
    """
    if states.ndim != 2 or states.shape[-1] != 14:
        raise ValueError(f"Expected [B,14] states, received {states.shape}")
    if actions.ndim != 3 or actions.shape[0] != states.shape[0] or actions.shape[-1] != 14:
        raise ValueError(f"Expected [B,H,14] actions, received {actions.shape}")
    transformed_states = np.array(states, copy=True)
    transformed_actions = np.array(actions, copy=True)
    transformed_actions -= np.where(np.asarray(DELTA_MASK), transformed_states, 0)[:, None, :]
    if not np.all(np.isfinite(transformed_states)) or not np.all(np.isfinite(transformed_actions)):
        raise ValueError("Nonfinite state/action after Aloha and delta transforms")
    return transformed_states, transformed_actions


def episode_batches(qpos: np.ndarray, episode: dict, batch_size: int):
    length = int(episode["raw_num_frames"])
    training_length = int(episode["training_length"])
    if length < 2 or training_length != length - 1:
        raise ValueError("Episode must contain T raw frames and T-1 aligned training starts")
    offset = int(episode["state_offset"])
    positions = qpos[offset: offset + length]
    if positions.shape != (length, 14):
        raise ValueError("Numeric episode slice has the wrong shape")
    for first in range(0, training_length, batch_size):
        starts = np.arange(first, min(first + batch_size, training_length))
        future = np.minimum(starts[:, None] + 1 + np.arange(ACTION_HORIZON)[None, :], length - 1)
        yield transform_numeric(positions[starts], positions[future])


def compute(index_path: Path, output_dir: Path, *, batch_size: int = 256) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    output_path = output_dir / "norm_stats.json"
    if output_path.exists():
        raise FileExistsError(f"Refusing to replace existing normalization statistics: {output_path}")
    index_sha = sha256(index_path)
    index = json.loads(index_path.read_text())
    if index.get("schema") != "robotwin_raw_hdf5_v1":
        raise ValueError("Unknown raw dataset index schema")
    if index.get("action_horizon") != ACTION_HORIZON:
        raise ValueError("Index action horizon differs from frozen model horizon")
    numeric_path = index_path.parent / index["numeric_path"]
    qpos = np.load(numeric_path, mmap_mode="r", allow_pickle=False)
    if qpos.ndim != 2 or qpos.shape[1] != 14 or qpos.dtype != np.float32:
        raise ValueError(f"Expected float32 [N,14] qpos, got {qpos.shape} / {qpos.dtype}")
    episodes = index["episodes"]
    expected = sum(int(episode["training_length"]) for episode in episodes)
    if expected != int(index["total_train_frames"]):
        raise ValueError("Episode lengths do not sum to total_train_frames")
    if sum(int(episode["raw_num_frames"]) for episode in episodes) != len(qpos):
        raise ValueError("Raw episode lengths do not sum to numeric array length")
    for previous, current in zip(episodes, episodes[1:]):
        if previous["state_offset"] + previous["raw_num_frames"] != current["state_offset"]:
            raise ValueError("Noncontiguous numeric episode offsets")
    if not episodes or episodes[0]["state_offset"] != 0:
        raise ValueError("Expected nonempty contiguous episode index beginning at zero")

    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    started = time.monotonic()
    start_utc = datetime.now(timezone.utc).isoformat()
    status_path = output_dir / "norm_stats_status.json"
    frames_seen = 0
    for episode_index, episode in enumerate(episodes):
        for states, actions in episode_batches(qpos, episode, batch_size):
            stats["state"].update(states)
            stats["actions"].update(actions)
            frames_seen += len(states)
        if (episode_index + 1) % 100 == 0 or episode_index + 1 == len(episodes):
            status = {
                "status": "running", "started_utc": start_utc,
                "completed_episodes": episode_index + 1, "total_episodes": len(episodes),
                "state_vectors": frames_seen, "action_vectors": frames_seen * ACTION_HORIZON,
                "expected_state_vectors": expected, "elapsed_seconds": time.monotonic() - started,
            }
            atomic_json(status_path, status)
            print(json.dumps(status), flush=True)
    if frames_seen != expected:
        raise AssertionError("Normalization did not cover every training frame")
    if sha256(index_path) != index_sha:
        raise RuntimeError("Index changed while computing normalization")
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = normalize.serialize_json({key: value.get_statistics() for key, value in stats.items()})
    # Exclusive creation makes concurrent accidental invocations fail, not overwrite.
    with output_path.open("x") as handle:
        handle.write(serialized)
        handle.write("\n")
    metadata = {
        "status": "complete", "started_utc": start_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "index_path": str(index_path.resolve()), "index_sha256": index_sha,
        "numeric_path": str(numeric_path.resolve()), "numeric_sha256": sha256(numeric_path),
        "norm_stats_sha256": sha256(output_path), "script_sha256": sha256(Path(__file__)),
        "running_stats_source_sha256": sha256(Path(normalize.__file__)),
        "episodes": len(episodes), "state_vectors": frames_seen,
        "action_vectors": frames_seen * ACTION_HORIZON, "action_horizon": ACTION_HORIZON,
        "normalization": "openpi RunningStats, 5000-bin q01/q99",
        "adapt_to_pi": False, "delta_mask": list(DELTA_MASK),
        "alignment": "state=q[t], actions=q[min(t+1+k,T-1)], k=0..49, t=0..T-2",
        "batch_size": batch_size, "dropped_starts": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    atomic_json(output_dir / "norm_stats_provenance.json", metadata)
    atomic_json(status_path, metadata)
    print(json.dumps(metadata, indent=2), flush=True)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=ROOT / "data" / "index.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets" / ASSET_ID)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    compute(args.index.resolve(), args.output_dir.resolve(), batch_size=args.batch_size)


if __name__ == "__main__":
    main()
