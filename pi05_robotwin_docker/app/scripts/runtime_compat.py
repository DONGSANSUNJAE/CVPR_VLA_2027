"""Narrow runtime adapters for the paper source and its pinned dependencies.

The paper checkout's callback handler uses a future class introduced after its
declared orbax-checkpoint==0.11.1 dependency. This adapter keeps asset writes
inside Orbax's async-save phase and completes them before checkpoint commit.
It does not modify model computation, weights, optimizer, or checkpoint format.
"""

from __future__ import annotations

import asyncio
import logging


def install_runtime_compat() -> dict[str, bool]:
    import jax
    # augmax 0.3.4 still calls the removed top-level alias; keep its exact
    # transformation while using the supported JAX tree implementation.
    tree_map_needed = not hasattr(jax, "tree_map")
    if tree_map_needed:
        jax.tree_map = jax.tree.map
    import orbax.checkpoint.future as future
    from openpi.training import checkpoints

    needed = not hasattr(future, "CommitFutureAwaitingContractedSignals")
    if needed:
        async def save_callback_before_commit(self, directory, args):
            await asyncio.to_thread(self.save, directory, args)
            return []

        checkpoints.CallbackHandler.async_save = save_callback_before_commit
        logging.info("Enabled Orbax 0.11.1 callback compatibility; assets finish before commit")
    return {"orbax_callback_compat": needed, "augmax_jax_tree_map_alias": tree_map_needed}
