"""Two-, three- or four-device FSDP layouts with replicated microbatch inputs.

Only dimensions divisible by the device count are partitioned. All other tensors remain
replicated; there is no padding or change to parameters or the optimizer.
Keep native batch activation constraints inactive for microbatch accumulation.
"""

from __future__ import annotations

import contextlib
import logging
import math
from typing import Any

import jax
import numpy as np


AXIS_THREE = "fsdp3"
MESH_SHAPE = (3,)
MESH_AXES = (AXIS_THREE,)


def make_mesh(devices=None) -> jax.sharding.Mesh:
    """Use exactly two, three or four devices, in their supplied/native JAX order."""
    devices = tuple(jax.devices() if devices is None else devices)
    if len(devices) not in (2, 3, 4) or len(set(devices)) != len(devices):
        raise ValueError("A6000 training requires two, three or four distinct devices")
    return jax.sharding.Mesh(np.asarray(devices, dtype=object).reshape((len(devices),)), MESH_AXES)


def replicated_sharding(mesh: jax.sharding.Mesh) -> jax.sharding.NamedSharding:
    _validate_mesh(mesh)
    return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())


def _validate_mesh(mesh):
    if tuple(mesh.axis_names) != MESH_AXES or tuple(mesh.devices.shape) not in ((2,), (3,), (4,)):
        raise ValueError(f"Expected {MESH_AXES} mesh of shape (2,), (3,) or (4,), got {mesh}")


def partition_spec(shape, dtype, *, min_size_mbytes=4, num_devices=3) -> jax.sharding.PartitionSpec:
    """Shard the largest legal dimension; preserve native small-array policy."""
    if num_devices not in (2, 3, 4):
        raise ValueError("num_devices must be 2, 3 or 4")
    shape = tuple(map(int, shape))
    if min_size_mbytes < 0:
        raise ValueError("min_size_mbytes must be nonnegative")
    nbytes = math.prod(shape) * np.dtype(dtype).itemsize
    if len(shape) < 2 or nbytes < min_size_mbytes * 2**20:
        return jax.sharding.PartitionSpec()
    axes = sorted(range(len(shape)), key=lambda axis: (-shape[axis], axis))
    for axis in axes:
        if shape[axis] % num_devices == 0:
            spec = [None] * len(shape)
            spec[axis] = AXIS_THREE
            return jax.sharding.PartitionSpec(*spec)
    return jax.sharding.PartitionSpec()


def fsdp_sharding(pytree: Any, mesh, *, min_size_mbytes=4, log=False):
    """Return a NamedSharding pytree for params or the complete TrainState."""
    _validate_mesh(mesh)

    def layout(path, array):
        spec = (partition_spec(array.shape, array.dtype, min_size_mbytes=min_size_mbytes, num_devices=mesh.size)
                if hasattr(array, "shape") else jax.sharding.PartitionSpec())
        if log and spec:
            logging.info("A6000 sharding %s shape=%s spec=%s", jax.tree_util.keystr(path), array.shape, spec)
        return jax.sharding.NamedSharding(mesh, spec)

    return jax.tree_util.tree_map_with_path(layout, pytree)


@contextlib.contextmanager
def replicated_activations():
    """Guard against accidentally enabling native batch-sharding constraints.

    Native activation_sharding_constraint is an identity while its context is
    inactive. The compiler may still choose intermediate layouts consistent
    with explicit parameter layouts; no replicated activation buffer is forced.
    """
    from openpi.training import sharding as native_sharding

    if native_sharding._MeshState.active_mesh is not None:
        raise RuntimeError("Do not combine replicated microbatches with native sharding.set_mesh")
    yield


def memory_accounting(pytree, layouts):
    """Exact resident numeric leaf sizes, excluding all execution temporaries."""
    leaves, structure = jax.tree_util.tree_flatten_with_path(pytree)
    layout_leaves, layout_structure = jax.tree_util.tree_flatten(layouts)
    if structure != layout_structure:
        raise ValueError("Array and layout pytrees do not match")
    entries = []
    for (path, array), layout in zip(leaves, layout_leaves, strict=True):
        if not hasattr(array, "shape"):
            raise TypeError(f"Expected shape/dtype leaves, got {type(array)}")
        local_shape = tuple(layout.shard_shape(array.shape))
        global_bytes = math.prod(array.shape) * np.dtype(array.dtype).itemsize
        local_bytes = math.prod(local_shape) * np.dtype(array.dtype).itemsize
        entries.append({"path": jax.tree_util.keystr(path), "shape": list(array.shape),
                        "dtype": str(array.dtype), "spec": repr(layout.spec),
                        "local_shape": list(local_shape), "global_bytes": global_bytes,
                        "per_device_bytes": local_bytes,
                        "partition_factor": global_bytes // local_bytes if local_bytes else 1})
    resident_bytes = sum(entry["per_device_bytes"] for entry in entries)
    return {"leaf_count": len(entries), "global_bytes": sum(x["global_bytes"] for x in entries),
            "per_device_bytes": resident_bytes, "per_device_gib": resident_bytes / 2**30,
            "uniform_per_device_bytes": [resident_bytes] * layout_leaves[0].mesh.size, "leaves": entries}


def audit_model(output):
    """Abstract native Pi0.5/Adam/EMA state; never materialize model weights."""
    from datetime import datetime, timezone
    import hashlib
    import json
    from pathlib import Path

    import flax.nnx as nnx
    import jax.numpy as jnp
    from configs.train_config import build_config
    from openpi.training import optimizer, utils

    config = build_config(fsdp_devices=jax.device_count())
    tx = optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def initialize(rng):
        _, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        params = nnx.state(model)
        return utils.TrainState(step=jnp.asarray(0), params=params, model_def=nnx.graphdef(model),
                               tx=tx, opt_state=tx.init(params.filter(config.trainable_filter)),
                               ema_decay=config.ema_decay, ema_params=params)

    mesh = make_mesh()
    shape = jax.eval_shape(initialize, jax.random.key(config.seed))
    layouts = fsdp_sharding(shape, mesh)
    state = memory_accounting(shape, layouts)
    params = memory_accounting(shape.params, layouts.params)
    trainable_params = shape.params.filter(config.trainable_filter)
    gradients = memory_accounting(trainable_params, layouts.params.filter(config.trainable_filter))
    optimizer_state = memory_accounting(shape.opt_state, layouts.opt_state)
    ema = memory_accounting(shape.ema_params, layouts.ema_params)
    resident = state["per_device_bytes"]
    gradient_bytes = gradients["per_device_bytes"]
    report = {
        "status": "abstract_shape_and_layout_audited_gpu_fit_unverified",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": "jax.eval_shape of configured full Pi0.5, native AdamW and EMA; CPU two/three/four-device NamedSharding",
        "jax_version": jax.__version__, "devices": list(map(str, jax.devices())),
        "mesh_shape": list(mesh.devices.shape), "mesh_axes": list(MESH_AXES),
        "batch_layout": "replicated; microbatch1 or2; total64 per optimizer update",
        "parameter_count": sum(math.prod(x.shape) for x in jax.tree.leaves(shape.params)),
        "trainable_parameter_count": sum(math.prod(x.shape) for x in jax.tree.leaves(trainable_params)),
        "parameter_shapes_and_layouts": params, "optimizer_state": optimizer_state,
        "ema_state": ema, "complete_train_state": state,
        "memory_budget": {
            "resident_state_gib_per_device": resident / 2**30,
            "fp32_gradient_or_accumulator_gib_per_device": gradient_bytes / 2**30,
            "state_plus_one_gradient_buffer_gib_per_device": (resident + gradient_bytes) / 2**30,
            "state_plus_two_gradient_buffers_gib_per_device": (resident + 2 * gradient_bytes) / 2**30,
            "remaining_of_48gib_after_state_and_two_gradient_buffers": 48 - (resident + 2 * gradient_bytes) / 2**30,
            "excluded": ["activations", "temporary parameter gathers", "compiler workspaces",
                         "optimizer temporaries", "input batches", "allocator overhead"],
            "interpretation": "Resident accounting only. XLA may fuse or alias gradient buffers, and may add temporaries; this is not a peak-memory estimate or GPU fit guarantee.",
        },
        "module_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["status", "parameter_count", "trainable_parameter_count", "memory_budget"]}, indent=2))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-output", required=True)
    audit_model(parser.parse_args().audit_output)
