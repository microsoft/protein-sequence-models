from typing import List, Any, Iterable
import random
from copy import copy

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
import pandas as pd

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD, START, STOP
from sequence_models.constants import PROTEIN_ALPHABET


class CSVDataset(Dataset):

    def __init__(self, fpath: str, split=None, outputs=[]):
        self.data = pd.read_csv(fpath)
        if split is not None:
            self.data = self.data[self.data['split'] == split]
        self.outputs = outputs
        self.data = self.data[['sequence'] + self.outputs]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        return [row['sequence'], *row[self.outputs]]


class SimpleCollater(object):

    def __init__(self, alphabet: str, pad=False):
        self.pad = pad
        self.tokenizer = Tokenizer(alphabet)

    def __call__(self, batch: List[Any],) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        if self.pad:
            sequences = _pad(sequences, PAD)
        sequences = torch.LongTensor([self.tokenizer.tokenize(s) for s in sequences])
        data = (torch.tensor(d) for d in data[1:])
        return [sequences, *data]


class LMCollater(SimpleCollater):

    def __call__(self, batch: List[Any],) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        src = [START + s for s in sequences]
        tgt = [s + STOP for s in sequences]
        if self.pad:
            src = _pad(src, PAD)
            tgt = _pad(tgt, PAD)
        src = torch.LongTensor([self.tokenizer.tokenize(s) for s in src])
        tgt = torch.LongTensor([self.tokenizer.tokenize(s) for s in tgt])
        data = (torch.tensor(d) for d in data[1:])
        return [src, tgt, *data]


def _pad(tokenized: List[torch.Tensor], value: int) -> torch.Tensor:
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    output = torch.zeros((batch_size, max_len), dtype=tokenized[0].dtype) + value
    for row, t in enumerate(tokenized):
        output[row, :len(t)] = t
    return output


class SortishSampler(Sampler):
    """Returns indices such that inputs with similar lengths are close together."""

    def __init__(self, sequence_lengths: Iterable, bucket_size: int):
        self.data = np.argsort(sequence_lengths)
        self.bucket_size = bucket_size
        n_buckets = int(np.ceil(len(self.data) / self.bucket_size))
        self.data = [self.data[i * bucket_size: i * bucket_size + bucket_size] for i in range(n_buckets)]

    def __iter__(self):
        for bucket in self.data:
            np.random.shuffle(bucket)
        np.random.shuffle(self.data)
        for bucket in self.data:
            for idx in bucket:
                yield idx

    def __len__(self):
        return sum([len(data) for data in self.data])


class ApproxBatchSampler(BatchSampler):
    '''
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
	'''

    def __init__(self, sampler, max_tokens, max_batch, sample_lengths):
        self.longest_token = 0
        self.max_tokens = max_tokens
        self.max_batch = max_batch
        self.sampler = sampler
        self.sample_lengths = sample_lengths

    def __iter__(self):
        batch = []
        length = 0
        for idx in self.sampler:
            this_length = self.sample_lengths[idx]
            if (len(batch) + 1) * max(length, this_length) <= self.max_tokens:
                batch.append(idx)
                length = max(length, this_length)
                if len(batch) == self.max_batch:
                    yield batch
                    batch = []
                    length = 0
            else:
                yield batch
                batch = [idx]
                length = this_length


def mlm_collate_fn(batch):
    batch_mod = []
    batch_mod_bool = []
    for seq in batch:
        mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
        if len(mod_idx) == 0:
            mod_idx = np.random.choice(list(range(len(seq))))  # make sure at least one aa is chosen
        seq_mod = copy(seq)
        seq_mod_bool = np.array([False] * len(seq))

        for idx in mod_idx:
            p = np.random.uniform()

            if p <= 0.10:  # do nothing
                mod = seq[idx]
            if 0.10 < p <= 0.20:  # replace with random amino acid
                mod = np.random.choice([i for i in range(26) if i != seq[idx]])
                seq_mod_bool[idx] = True
            if 0.20 < p <= 1.00:  # mask
                mod = PROTEIN_ALPHABET.index('#')
                seq_mod_bool[idx] = True

            seq_mod[idx] = mod

        batch_mod.append(seq_mod)
        batch_mod_bool.append(seq_mod_bool)

    # padding
    batch_mod = _pad(batch_mod, PROTEIN_ALPHABET.index('-'))
    batch = _pad(batch, PROTEIN_ALPHABET.index('-'))
    batch_mod_bool = _pad(batch_mod_bool, False)

    # rename and convert to tensor
    src = torch.from_numpy(np.array(batch_mod)).long()
    tgt = torch.from_numpy(np.array(batch)).long()
    loss_mask = torch.from_numpy(np.array(batch_mod_bool)).bool()

    return src, tgt, loss_mask
