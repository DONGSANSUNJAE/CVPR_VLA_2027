"""Replay fixed training observations through a finalized native JAX checkpoint.

This is an offline action reproducibility check, NOT held-out task evaluation.
Preparing the lossless 50-observation input bundle requires only CPU and NumPy;
inference explicitly requires an allocated/local GPU before loading model weights.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT_SCHEMA = "robotwin_fixed_training_observations_v1"
REPLAY_SCHEMA = "robotwin_checkpoint_action_replay_v1"
ASSET_ID = "robotwin_clean_randomized_27500"
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ACTION_SHAPE = (50, 14)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def select_samples(index: dict, *, expected_count: int = 50) -> list[dict]:
    """Choose the first frame of each task's lowest numbered clean episode."""
    selected = {}
    for episode in index["episodes"]:
        if episode["setting"] != "clean":
            continue
        task = episode["task"]
        if task not in selected or episode["episode_index"] < selected[task]["episode_index"]:
            selected[task] = episode
    all_tasks = {episode["task"] for episode in index["episodes"]}
    if len(selected) != expected_count or set(selected) != all_tasks:
        raise ValueError(f"Expected one clean observation for each of {expected_count} tasks")
    result = []
    for task, ep in sorted(selected.items()):
        result.append({
            "sample_id": f"{task}/clean/episode{ep['episode_index']}/frame0",
            "task": task, "setting": "clean", "episode_index": ep["episode_index"],
            "frame_index": 0, "training_index": ep["training_offset"],
            "raw_path": ep["raw_path"], "instruction_path": ep["instruction_path"],
            "prompt_index": ep["prompt_index"], "prompt": ep["prompt"],
        })
    return result


def validate_observation(observation: dict):
    state = np.asarray(observation["state"])
    if state.shape != (14,) or state.dtype != np.float32 or not np.isfinite(state).all():
        raise ValueError("Expected finite raw float32 state with 14 coordinates")
    if set(observation["images"]) != set(CAMERAS):
        raise ValueError("Expected the three native RoboTwin camera names")
    for camera in CAMERAS:
        image = np.asarray(observation["images"][camera])
        if image.dtype != np.uint8 or image.shape != (3, 480, 640):
            raise ValueError(f"Expected uint8 CHW 640x480 image: {camera}")
    if not isinstance(observation["prompt"], str) or not observation["prompt"].strip():
        raise ValueError("Expected the frozen nonempty training prompt")


def _safe_child(directory: Path, name: str) -> Path:
    path = directory / name
    if Path(name).is_absolute() or ".." in Path(name).parts or path.resolve().parent != directory.resolve():
        raise ValueError(f"Invalid bundle member: {name}")
    return path


def read_inputs(directory: Path, *, expected_count: int = 50) -> dict:
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema") != INPUT_SCHEMA or manifest.get("count") != expected_count:
        raise ValueError("Unexpected fixed input schema/count")
    samples = manifest["samples"]
    ids = [sample["sample_id"] for sample in samples]
    tasks = [sample["task"] for sample in samples]
    if len(ids) != expected_count or len(set(ids)) != len(ids) or tasks != sorted(set(tasks)):
        raise ValueError("Fixed input ordering or uniqueness is invalid")
    for sample in samples:
        path = _safe_child(directory, sample["npz"])
        if sha256(path) != sample["npz_sha256"]:
            raise ValueError(f"Fixed observation changed: {sample['sample_id']}")
    return manifest


def load_observation(directory: Path, sample: dict) -> dict:
    with np.load(_safe_child(Path(directory), sample["npz"]), allow_pickle=False) as arrays:
        if set(arrays.files) != {"state", *CAMERAS}:
            raise ValueError("Unexpected observation archive members")
        observation = {
            "state": arrays["state"].copy(),
            "images": {camera: arrays[camera].copy() for camera in CAMERAS},
            "prompt": sample["prompt"],
        }
    validate_observation(observation)
    return observation


def prepare_inputs(index_path: Path, output: Path, *, expected_count: int = 50,
                   dataset_factory=None, index_reader=None) -> dict:
    """Atomic, idempotent creation; existing snapshots are verified, never replaced."""
    if dataset_factory is None or index_reader is None:
        from robotwin_dataset import RobotwinRawDataset, read_index
        dataset_factory = dataset_factory or RobotwinRawDataset
        index_reader = index_reader or read_index
    index_path, output = Path(index_path).resolve(), Path(output).resolve()
    index = index_reader(index_path)
    digest = sha256(index_path)
    selected = select_samples(index, expected_count=expected_count)
    if output.exists():
        previous = read_inputs(output, expected_count=expected_count)
        if previous["dataset_index_sha256"] != digest:
            raise ValueError("Existing fixed observations belong to a different dataset index")
        if [{key: sample[key] for key in choice} for sample, choice in zip(previous["samples"], selected)] != selected:
            raise ValueError("Existing fixed observation selection changed")
        return previous
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".pending-", dir=output.parent))
    dataset = dataset_factory(index_path)
    try:
        for ordinal, sample in enumerate(selected):
            observation = dataset[sample["training_index"]]
            validate_observation(observation)
            if observation["prompt"] != sample["prompt"]:
                raise ValueError("Dataset prompt differs from the frozen index")
            filename = f"{ordinal:03d}_{sample['task']}.npz"
            np.savez_compressed(staging / filename, state=observation["state"], **observation["images"])
            sample.update(npz=filename, npz_sha256=sha256(staging / filename))
        manifest = {
            "schema": INPUT_SCHEMA, "count": len(selected),
            "purpose": "Offline replay of training observations, not held-out task success evaluation",
            "selection": "Sorted tasks; first frame of lowest-index clean episode; no outcome filtering",
            "dataset_index_sha256": digest, "numeric_sha256": index.get("numeric_sha256"),
            "image_encoding": "Lossless snapshot of dataset loader uint8 CHW arrays; no channel swap",
            "state_encoding": "Raw absolute 14D RoboTwin joints and grippers before native transforms",
            "default_inference_seed": 42, "default_num_steps": 10, "samples": selected,
        }
        (staging / "manifest.json").write_bytes(json_bytes(manifest))
        read_inputs(staging, expected_count=expected_count)
        staging.rename(output)
        return manifest
    finally:
        dataset.close()
        if staging.exists():
            shutil.rmtree(staging)


def validate_checkpoint(path: Path) -> dict:
    """Require native Orbax atomic completion and both finalized PyTree items."""
    path = Path(path).resolve()
    if not path.is_dir() or any(".orbax-checkpoint-tmp" in part or ".pending" in part for part in path.parts):
        raise ValueError(f"Not a finalized checkpoint directory: {path}")
    metadata_path = path / "_CHECKPOINT_METADATA"
    if not metadata_path.is_file():
        raise ValueError("Checkpoint has no Orbax _CHECKPOINT_METADATA commit record")
    metadata = json.loads(metadata_path.read_text())
    timestamp = metadata.get("commit_timestamp_nsecs")
    if not isinstance(timestamp, int) or timestamp <= 0:
        raise ValueError("Orbax checkpoint commit has not completed")
    for item in ("params", "train_state"):
        filename = path / item / "_METADATA"
        if not filename.is_file():
            raise ValueError(f"Checkpoint missing {item}/_METADATA")
        json.loads(filename.read_text())
    norm_path = path / "assets" / ASSET_ID / "norm_stats.json"
    if not norm_path.is_file():
        raise ValueError("Checkpoint does not contain its own normalization statistics")
    json.loads(norm_path.read_text())
    return {"path": str(path), "completed_step": int(path.name) if path.name.isdecimal() else None,
            "commit_timestamp_nsecs": timestamp, "norm_stats_sha256": sha256(norm_path)}


def checkpoint_identity(path: Path) -> dict:
    identity = validate_checkpoint(path)
    path = Path(path).resolve()
    files = [path / "_CHECKPOINT_METADATA", path / "assets" / ASSET_ID / "norm_stats.json"]
    files.extend(p for p in (path / "params").rglob("*") if p.is_file())
    members = []
    for filename in sorted(files):
        before = filename.stat()
        digest = sha256(filename)
        after = filename.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Checkpoint changed while hashing: {filename}")
        members.append({"path": str(filename.relative_to(path)), "size": before.st_size, "sha256": digest})
    identity.update(
        inference_files=members,
        inference_sha256=hashlib.sha256(json_bytes(members)).hexdigest(),
        hash_scope="All params files, checkpoint commit metadata, and checkpoint normalization; not optimizer/train_state",
        inference_weights="Native params item contains EMA parameters (training ema_decay=0.99)",
    )
    return identity


def seed_native_policy(policy, seed: int):
    import jax
    # This frozen upstream Policy uses `rng or jax.random.key(0)` in __init__,
    # which raises TypeError for an explicitly passed typed key. Set its real
    # RNG immediately after native construction, before the first infer call.
    if not hasattr(policy, "_rng"):
        raise TypeError("Expected native JAX Policy with an inference RNG")
    policy._rng = jax.random.key(seed)
    return policy


def load_native_policy(checkpoint: Path, run_root: Path, seed: int, num_steps: int):
    from cuda_runtime import preload_cuda_libraries
    preload_cuda_libraries(run_root)
    for relative in ("", "openpi_snapshot/src", "openpi_snapshot/packages/openpi-client/src"):
        sys.path.insert(0, str(run_root / relative))
    import jax
    from configs.train_config import build_config
    from openpi.policies.policy_config import create_trained_policy

    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("Checkpoint inference needs an allocated GPU; use --prepare-inputs for CPU preparation")
    config = build_config(run_root=run_root)
    # Factory loads the checkpoint's own norms and native inverse delta/Aloha
    # transforms; outputs therefore have 50x14 raw absolute joint coordinates.
    policy = create_trained_policy(config, checkpoint, sample_kwargs={"num_steps": num_steps})
    return seed_native_policy(policy, seed)


def source_provenance(run_root: Path) -> dict:
    candidates = ("scripts/reproduce_checkpoint.py", "scripts/robotwin_dataset.py", "configs/train_config.py",
                  "manifests/run_sources.json", "manifests/runtime_versions.json",
                  "openpi_snapshot/src/openpi/policies/policy.py",
                  "openpi_snapshot/src/openpi/policies/policy_config.py",
                  "openpi_snapshot/src/openpi/policies/aloha_policy.py")
    versions = {}
    for package in ("jax", "jaxlib", "jax-cuda12-plugin", "flax", "orbax-checkpoint", "numpy", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"source_sha256": {name: sha256(run_root / name) for name in candidates if (run_root / name).is_file()},
            "runtime_versions": versions, "python_version": sys.version,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}


def compare_runs(current: dict, actions: np.ndarray, previous_directory: Path) -> dict:
    previous_directory = Path(previous_directory)
    previous = json.loads((previous_directory / "result.json").read_text())
    for key in ("schema", "inputs_manifest_sha256", "sample_ids", "seed", "num_steps", "output_encoding"):
        if current[key] != previous[key]:
            raise ValueError(f"Cannot compare replay runs with different {key}")
    if sha256(previous_directory / "actions.npz") != previous["actions_npz_sha256"]:
        raise ValueError("Previous action archive failed its checksum")
    with np.load(previous_directory / "actions.npz", allow_pickle=False) as saved:
        prior = saved["actions"]
        if saved["sample_ids"].tolist() != current["sample_ids"]:
            raise ValueError("Previous action archive sample order differs from manifest")
    if prior.shape != actions.shape or not np.isfinite(prior).all():
        raise ValueError("Previous replay action shape/values are invalid")
    delta = actions.astype(np.float64) - prior.astype(np.float64)
    same = current["checkpoint"]["inference_sha256"] == previous["checkpoint"]["inference_sha256"]
    return {"previous_directory": str(previous_directory.resolve()),
            "kind": "same_checkpoint_reproducibility" if same else "different_checkpoint_action_change",
            "same_checkpoint": same, "array_equal": bool(np.array_equal(actions, prior)),
            "allclose_atol_1e-5_rtol_1e-5": bool(np.allclose(actions, prior, atol=1e-5, rtol=1e-5)),
            "max_absolute_difference": float(np.abs(delta).max()),
            "mean_absolute_difference": float(np.abs(delta).mean()),
            "root_mean_squared_difference": float(np.sqrt(np.mean(delta ** 2))),
            "note": "Numerical action comparison only; no task-success or held-out generalization claim"}


def replay(policy, inputs_path: Path, output: Path, *, checkpoint: dict, seed: int = 42,
           num_steps: int = 10, provenance: dict | None = None,
           compare_to: Path | None = None, expected_count: int = 50) -> dict:
    manifest = read_inputs(inputs_path, expected_count=expected_count)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    actions, timings = [], []
    for sample in manifest["samples"]:
        observation = load_observation(inputs_path, sample)
        started = time.monotonic()
        prediction = np.asarray(policy.infer(observation)["actions"])
        timings.append(time.monotonic() - started)
        if prediction.shape != ACTION_SHAPE or not np.isfinite(prediction).all():
            raise ValueError(f"Invalid raw 50x14 policy output for {sample['sample_id']}")
        actions.append(prediction.copy())
        print(json.dumps({"sample_id": sample["sample_id"], "seconds": timings[-1]}), flush=True)
    actions = np.stack(actions)
    ids = [sample["sample_id"] for sample in manifest["samples"]]
    np.savez_compressed(output / "actions.npz", actions=actions, sample_ids=np.asarray(ids))
    result = {
        "schema": REPLAY_SCHEMA, "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Offline replay on fixed training examples, not held-out task-success evaluation",
        "checkpoint": checkpoint, "inputs_path": str(Path(inputs_path).resolve()),
        "inputs_manifest_sha256": sha256(Path(inputs_path) / "manifest.json"),
        "dataset_index_sha256": manifest["dataset_index_sha256"], "sample_ids": ids,
        "seed": seed, "rng": "Native JAX Policy key seeded once, then split sequentially in fixed sample order",
        "num_steps": num_steps, "output_encoding": "Raw absolute 14D RoboTwin joints/grippers after native inverse transforms",
        "actions_shape": list(actions.shape), "actions_dtype": str(actions.dtype),
        "actions_npz_sha256": sha256(output / "actions.npz"),
        "actions_array_sha256": hashlib.sha256(np.ascontiguousarray(actions).tobytes()).hexdigest(),
        "sample_inference_seconds": timings, "provenance": provenance or {},
        "reproducibility_limit": "Same inputs, weights, seed, transforms and runtime are recorded; bitwise equality across GPU/runtime versions is not guaranteed",
    }
    if compare_to is not None:
        result["comparison"] = compare_runs(result, actions, compare_to)
    (output / "inputs_manifest.json").write_bytes(json_bytes(manifest))
    (output / "result.json").write_bytes(json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT)
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--inputs-path", type=Path)
    parser.add_argument("--index-path", type=Path)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--step", type=int, default=5000)
    source.add_argument("--checkpoint", type=Path)
    parser.add_argument("--exp-name", default="run_v1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-to", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    inputs = args.inputs_path or root / "reproduction/fixed_inputs"
    if args.prepare_inputs:
        manifest = prepare_inputs(args.index_path or root / "data/index.json", inputs)
        print(json.dumps({"inputs_path": str(inputs.resolve()), "count": manifest["count"],
                          "manifest_sha256": sha256(inputs / "manifest.json")}), flush=True)
        return
    if args.num_steps < 1 or not 0 <= args.seed < 2**32:
        parser.error("--num-steps must be positive and --seed must fit uint32")
    checkpoint_path = args.checkpoint or root / "checkpoints/pi05_robotwin_clean_randomized_60k" / args.exp_name / str(args.step)
    checkpoint = checkpoint_identity(checkpoint_path)
    read_inputs(inputs)
    output = args.output or root / "reproduction" / f"step_{checkpoint['completed_step']}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite replay output: {output}")
    policy = load_native_policy(checkpoint_path.resolve(), root, args.seed, args.num_steps)
    import jax
    provenance = source_provenance(root)
    provenance["jax_devices"] = [{"platform": d.platform, "device_kind": d.device_kind} for d in jax.devices()]
    result = replay(policy, inputs, output, checkpoint=checkpoint, seed=args.seed, num_steps=args.num_steps,
                    provenance=provenance, compare_to=args.compare_to)
    print(json.dumps({"output": str(output.resolve()), "actions_shape": result["actions_shape"],
                      "comparison": result.get("comparison")}), flush=True)


if __name__ == "__main__":
    main()
