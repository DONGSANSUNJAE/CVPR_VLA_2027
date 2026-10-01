"""Frame-uniform shuffle reproducible from the completed optimizer step.

Each epoch uses torch.randperm(seed + epoch). Incomplete final batches are
dropped, matching the public trainer's batch policy. Resume skips indices, never
decodes discarded images, and is independent of worker prefetch positions.
"""
import torch


class StepSampler(torch.utils.data.Sampler):
    def __init__(self, size, batch_size, seed, completed_updates=0, *, process_count=1, process_index=0):
        if size < batch_size or batch_size < 1:
            raise ValueError('Dataset must contain at least one full batch')
        self.size = int(size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        if batch_size % process_count or not 0 <= process_index < process_count:
            raise ValueError('Global batch must divide over valid process ranks')
        self.process_count = int(process_count)
        self.process_index = int(process_index)
        self.local_batch_size = self.batch_size // self.process_count
        self.steps_per_epoch = self.size // self.batch_size
        self.reset(completed_updates)

    def reset(self, completed_updates):
        if completed_updates < 0:
            raise ValueError('Completed updates cannot be negative')
        self.epoch, offset = divmod(int(completed_updates), self.steps_per_epoch)
        self.first_index = offset * self.batch_size

    def __iter__(self):
        epoch, start = self.epoch, self.first_index
        self.epoch += 1
        self.first_index = 0
        generator = torch.Generator().manual_seed(self.seed + epoch)
        permutation = torch.randperm(self.size, generator=generator)
        end = self.steps_per_epoch * self.batch_size
        for batch_start in range(start, end, self.batch_size):
            local_start = batch_start + self.process_index * self.local_batch_size
            yield from permutation[local_start:local_start + self.local_batch_size].tolist()

    def __len__(self):
        return (self.steps_per_epoch * self.batch_size - self.first_index) // self.process_count
