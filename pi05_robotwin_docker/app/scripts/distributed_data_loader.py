"""Native openpi batch conversion with explicit per-process frame sampling.

The public TorchDataLoader rejects process_count>1. This constructor supplies
the missing local batch and distributed sampler; its iteration/collation/model
transforms remain native openpi implementations.
"""
import multiprocessing

import torch
from openpi.training import data_loader


class DistributedTorchDataLoader(data_loader.TorchDataLoader):
    def __init__(self, dataset, local_batch_size, *, sharding, sampler, num_workers, seed, worker_init_fn):
        self._sharding = sharding
        self._num_batches = None
        generator = torch.Generator().manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=local_batch_size, sampler=sampler, shuffle=False,
            num_workers=num_workers,
            multiprocessing_context=multiprocessing.get_context('spawn') if num_workers else None,
            persistent_workers=num_workers > 0,
            collate_fn=data_loader._collate_fn, worker_init_fn=worker_init_fn,
            drop_last=True, generator=generator)
