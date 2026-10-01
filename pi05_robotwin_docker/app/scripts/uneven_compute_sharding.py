"""Internal non-layer sharding; stored arrays and checkpoint shapes stay exact.

JAX permits uneven INTERNAL constraints. Public state inputs/outputs continue
using three_gpu_sharding's legal divisible-device-count layouts. XLA alone handles any
padding needed by the internal layout; no model leaf or sample is padded here.
"""
import jax
from three_gpu_sharding import AXIS_THREE, fsdp_sharding


def compute_layouts(params, mesh):
    """Change only large scanned weight arrays currently sharded over layers."""
    storage = fsdp_sharding(params, mesh)
    def choose(value, layout):
        if (len(value.shape) > 2 and value.shape[0] in (18, 27)
                and layout.spec and layout.spec[0] == AXIS_THREE):
            axis = max(range(1, len(value.shape)), key=lambda i: (value.shape[i], -i))
            spec = [None] * len(value.shape)
            spec[axis] = AXIS_THREE
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))
        return layout
    return jax.tree.map(choose, params, storage)


def constrain_compute_params(params, mesh):
    # Apply outside the microbatch scan: repartition constant model weights once
    # per optimizer update rather than gather all scanned layers at each layer.
    return jax.lax.with_sharding_constraint(params, compute_layouts(params, mesh))
