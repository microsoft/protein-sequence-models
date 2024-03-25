from typing import Iterable
import math

import numpy as np
from torch.utils.data import Sampler, BatchSampler


class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close together."""

    def __init__(self, sequence_lengths: Iterable, bucket_size: int, num_replicas: int = 1, rank: int = 0):
        self.data = np.argsort(sequence_lengths)
        self.num_replicas = num_replicas
        self.num_samples = int(math.ceil(len(self.data) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [self.data[i * bucket_size: i * bucket_size + bucket_size] for i in range(n_buckets)]
        self.rank = rank
        self.epoch = 0
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        indices = [item for sublist in self.data for item in sublist]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size
        # subsample
        start = self.rank * self.num_samples
        end = start + self.num_samples
        indices = indices[start:end]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
        np.random.seed(self.epoch)


class ClusteredSortishSampler(SortishSampler):
    """Samples from clusters, then yields indices such that inputs with similar lengths are close together."""

    def __init__(self, sequence_lengths: Iterable, clusters: Iterable,
                 bucket_size: int, num_replicas: int = 1, rank: int = 0):
        self.num_replicas = num_replicas
        self.clusters = clusters
        self.cluster_sizes = np.array([len(c) for c in self.clusters])
        self.num_samples = int(math.ceil(len(self.clusters) * 1.0 / self.num_replicas))
        self.bucket_size = bucket_size
        self.n_buckets = int(np.ceil(len(self.clusters) / self.bucket_size))
        self.lengths = sequence_lengths
        self.rank = rank
        self.total_size = self.num_samples * self.num_replicas
        self.all_data = np.argsort(sequence_lengths)

    def set_epoch(self, epoch):
        self.epoch = epoch
        np.random.seed(self.epoch)
        selected = np.random.randint(self.cluster_sizes)
        selected_indices = [c[s] for c, s in zip(self.clusters, selected)]
        self.data = self.all_data[np.isin(self.all_data, selected_indices, assume_unique=True)]
        self.data = [self.data[i * self.bucket_size: i * self.bucket_size + self.bucket_size] for i in
                     range(self.n_buckets)]



class ApproxBatchSampler(BatchSampler):
    """
	Parameters:
	-----------
	sampler : Pytorch Sampler
		Choose base sampler class to use for bucketing

	max_tokens : int
		Maximum number of tokens per batch

	max_batch: int
		Maximum batch size

	sample_lengths : array-like
		List of lengths of sequences in the order of the dataset
	"""

    def __init__(self, sampler, max_tokens, max_batch, sample_lengths, max_square_tokens=np.inf,
                 msa_depth=None, batch_mult=1):
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths
        self.max_square_tokens = max_square_tokens
        self.msa_depth = msa_depth
        self.batch_mult = batch_mult

    def __iter__(self):
        batch = []
        length = 0
        ell_sq = 0
        for idx in self.sampler:
            this_length = self.sample_lengths[idx]
            if self.msa_depth is None:
                linear = (len(batch) + 1) * max(length, this_length)
            else:
                max_len = max(length, this_length)
                linear = (len(batch) + 1) * (max_len * self.msa_depth ** 2 + max_len ** 2 * self.msa_depth)
            quadratic = (len(batch) + 1) * max(ell_sq, this_length ** 2)
            if linear <= self.max_tokens and quadratic < self.max_square_tokens:
                batch.append(idx)
                length = max(length, this_length)
                ell_sq = max(ell_sq, this_length ** 2)
                if len(batch) == self.max_batch:
                    yield batch
                    batch = []
                    length = 0
            else:
                rounded_n = (len(batch) // self.batch_mult) * self.batch_mult
                rounded_n = max(1, rounded_n)
                yield batch[:rounded_n]
                batch = batch[rounded_n:] + [idx]
                length = max([self.sample_lengths[i] for i in batch])
                ell_sq = length ** 2
        if len(batch) > 0:
            yield batch
