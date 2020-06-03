from typing import Iterable

import pandas as pd
import numpy as np
import random
from copy import copy

from torch.utils.data import Dataset
import torch
from torch.utils.data.sampler import Sampler, BatchSampler

from sequence_models.utils import Tokenizer
from sequence_models.constants import PROTEIN_ALPHABET


def pad(batch, pad_idx):
	max_len = max([len(i) for i in batch])
	padded = [list(i) + [pad_idx]*(max_len - len(i)) for i in batch]
	return padded


class CSVDataset(Dataset):

	def __init__(self, fpath: str, alphabet,):
		self.data = pd.read_csv(fpath)
		self.tokenizer = Tokenizer(alphabet)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.tokenizer.tokenize(self.data.loc[idx]['sequences'])


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
		mod_idx = random.sample(list(range(len(seq))), int(len(seq)*0.15))
		if len(mod_idx) == 0:
			mod_idx = np.random.choice(list(range(len(seq))))  # make sure at least one aa is chosen
		seq_mod = copy(seq)
		seq_mod_bool = np.array([False]*len(seq))

		for idx in mod_idx:
			p = np.random.uniform()

			if p <= 0.10: # do nothing
				mod = seq[idx]
			if 0.10 < p <= 0.20: # replace with random amino acid
				mod = np.random.choice([i for i in range(26) if i != seq[idx] ])
				seq_mod_bool[idx] = True
			if 0.20 < p <= 1.00: # mask
				mod = PROTEIN_ALPHABET.index('#')
				seq_mod_bool[idx] = True

			seq_mod[idx] = mod

		batch_mod.append(seq_mod)
		batch_mod_bool.append(seq_mod_bool)

	# padding
	batch_mod = pad(batch_mod, PROTEIN_ALPHABET.index('-'))
	batch = pad(batch, PROTEIN_ALPHABET.index('-'))
	batch_mod_bool = pad(batch_mod_bool, False)

	# rename and convert to tensor
	src = torch.from_numpy(np.array(batch_mod)).long()
	tgt = torch.from_numpy(np.array(batch)).long()
	loss_mask = torch.from_numpy(np.array(batch_mod_bool)).bool()

	return src, tgt, loss_mask