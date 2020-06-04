import numpy as np

from sequence_models.data import SortishSampler, ApproxBatchSampler

n_samples = np.random.randint(100, 500)
lengths = np.random.randint(100, 200, n_samples)
bucket_size = np.random.randint(10, 20, dtype=int)
sampler = SortishSampler(lengths, bucket_size)
assert len(sampler) == n_samples
assert len(sampler.data) == np.ceil(n_samples / bucket_size)

max_tokens = 1000
max_batch = 12
batch_sampler = ApproxBatchSampler(sampler, max_tokens, max_batch, lengths)
for batch in batch_sampler:
    assert len(batch) <= max_batch
    assert len(batch) * max(lengths[batch]) <= max_tokens