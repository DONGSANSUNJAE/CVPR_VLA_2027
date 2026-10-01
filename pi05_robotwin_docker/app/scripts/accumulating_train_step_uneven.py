"""Exact 64-example optimizer steps using sequential, unpadded microbatches.

The model loss, AdamW transformation (including clipping), EMA and reported
metrics follow the public openpi train_step. Only gradient evaluation is split:
every microbatch sees the same pre-update parameters and FP32 gradients are
averaged before a single optimizer/EMA update. The caller supplies replicated
input arrays and the desired parameter/optimizer shardings to its outer jit.

For more than one microbatch the random streams are deterministically folded by
microbatch index. They are reproducible on resume with the same microbatch size,
but are not bit-identical to one native stochastic forward on all 64 examples.
"""
from __future__ import annotations

import dataclasses
from typing import Any

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import optax

from openpi.shared import nnx_utils
from uneven_compute_sharding import constrain_compute_params


GLOBAL_BATCH_SIZE = 64


def accumulating_train_step(
    config: Any,
    rng: Any,
    state: Any,
    batch: tuple[Any, Any],
    *,
    microbatch_size: int = 1,
    compute_mesh: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Return one complete native-style update over exactly 64 real examples.

    ``microbatch_size`` must be a static positive divisor of 64. Apply an outer
    ``jax.jit(functools.partial(..., config, microbatch_size=...))`` with the
    experiment's state shardings and replicated batch sharding. This function
    deliberately does not activate native openpi's batch-axis mesh constraint.
    """
    if config.batch_size != GLOBAL_BATCH_SIZE:
        raise ValueError("This experiment requires global batch_size=64")
    if (not isinstance(microbatch_size, int) or isinstance(microbatch_size, bool)
            or microbatch_size <= 0 or GLOBAL_BATCH_SIZE % microbatch_size):
        raise ValueError("microbatch_size must be a positive integer divisor of 64")
    for leaf in jax.tree.leaves(batch):
        if not hasattr(leaf, "shape") or not leaf.shape or leaf.shape[0] != GLOBAL_BATCH_SIZE:
            raise ValueError("Every input leaf must contain exactly 64 real examples; padding is not supported")
    if compute_mesh is None:
        raise ValueError("compute_mesh is required for internal uneven constraints")
    compute_params = constrain_compute_params(state.params, compute_mesh)
    num_microbatches = GLOBAL_BATCH_SIZE // microbatch_size
    micro_observations, micro_actions = jax.tree.map(
        lambda value: value.reshape((num_microbatches, microbatch_size, *value.shape[1:])), batch,
    )
    train_rng = jax.random.fold_in(rng, state.step)
    params = state.params.filter(config.trainable_filter)
    initial_gradients = jax.tree.map(lambda value: jnp.zeros_like(value, dtype=jnp.float32), params)
    weight = jnp.asarray(1.0 / num_microbatches, dtype=jnp.float32)
    differentiate = nnx.DiffState(0, config.trainable_filter)

    def loss_fn(model, micro_rng, observation, actions):
        return jnp.mean(model.compute_loss(micro_rng, observation, actions, train=True))

    def accumulate(carry, inputs):
        mean_loss, mean_gradients = carry
        index, observation, actions = inputs
        # A fresh module per traced scan body keeps NNX's transient bookkeeping
        # inside that body. No parameter or optimizer update occurs here.
        model = nnx.merge(state.model_def, compute_params)
        model.train()
        micro_rng = train_rng if num_microbatches == 1 else jax.random.fold_in(train_rng, index)
        loss, gradients = nnx.value_and_grad(loss_fn, argnums=differentiate)(
            model, micro_rng, observation, actions,
        )
        mean_gradients = jax.tree.map(
            lambda accumulated, gradient: accumulated + gradient.astype(jnp.float32) * weight,
            mean_gradients, gradients,
        )
        return (mean_loss + loss.astype(jnp.float32) * weight, mean_gradients), None

    (loss, grads), _ = jax.lax.scan(
        accumulate,
        (jnp.zeros((), dtype=jnp.float32), initial_gradients),
        (jnp.arange(num_microbatches, dtype=jnp.int32), micro_observations, micro_actions),
        unroll=1,
    )

    # The configured transformation clips the complete mean gradient once and
    # advances Adam and its learning-rate schedule once per 64 real examples.
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    updated_trainable_params = optax.apply_updates(params, updates)
    model = nnx.merge(state.model_def, state.params)
    model.train()
    nnx.update(model, updated_trainable_params)
    new_params = nnx.state(model)
    new_state = dataclasses.replace(
        state, step=state.step + 1, params=new_params, opt_state=new_opt_state,
    )
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params, new_params,
            ),
        )
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, value: value.value.ndim > 1,
        ),
    )
    return new_state, {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
    }
