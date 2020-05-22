import pandas as pd
import numpy as np
import math

from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, Sampler, \
	BatchSampler, SubsetRandomSampler

from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.utils import identity

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD


def pad(seq, pad_token, max_length):
	padded = seq + pad_token * (max_length-len(seq))
	return padded

class CSVDataset(Dataset):

	def __init__(self, fpath: str, alphabet, pad_token=PAD):
		self.data = pd.read_csv(fpath)
		self.tokenizer = Tokenizer(alphabet)
		self.pad_token = pad_token
	
	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx_padlen):
		idx = idx_padlen[0]
		padlen = idx_padlen[1]
		
		row = self.data.loc[idx]
		padded = pad(row['sequences'], self.pad_token, padlen)
		return self.tokenizer.tokenize(padded)


class BucketBatchSampler(BatchSampler):
	'''
	Parameters:
	-----------
	sampler : Pytorch Sampler
		Choose base sampler class to use for bucketing
	
	approx_token : int
		Approximately number of tokens per batch

	sample_lengths : array-like
		List of lengths of sequences in the order of the dataset

	drop_last : bool
		If `True` the sampler will drop the last batch if its 
		size would be less than `batch_size`.

	sort_key : callable
		Callable to specify a comparison key for sorting

	bucket_size : int 
		Size of buckets to divide data into
	'''

	def __init__(self, sampler, approx_token, sample_lengths,
				 drop_last, sort_key=identity, bucket_size=100):
		super().__init__(sampler, 1, drop_last)
		self.batch_size = 1
		self.longest_token = 0
		self.approx_token = approx_token
		self.sample_lengths = sample_lengths
		self.sort_key = sort_key
		self.bucket_sampler = BatchSampler(sampler,
										   min(bucket_size, len(sampler)),
										   False)
		

	def __iter__(self):
		for bucket in self.bucket_sampler:
			sorted_sampler = SortedSampler(bucket, self.sort_key)
			for mini_batch in SubsetRandomSampler(
					list(BatchSampler(sorted_sampler, 1,self.drop_last))):
				if self.longest_token == 0:
					batch = []
				batch.append(bucket[mini_batch[0]])
				self.longest_token = max(self.longest_token, self.sample_lengths[bucket[mini_batch[0]]])
				if self.longest_token * len(batch) >= self.approx_token:
					yield_batch = [(i, self.longest_token) for i in batch]
					self.longest_token = 0
					yield yield_batch
	
	def __len__(self):
		if self.drop_last:
			return len(self.sampler) // self.batch_size
		else:
			return math.ceil(len(self.sampler) / self.batch_size)



'''
Example:
--------

# load dataset into CSVDataset
dataset = CSVDataset('UniLanguage/data/arc_frag_exp/train.csv', PROTEIN_ALPHABET, '-')

# extract sequence lengths
data_df = pd.read_csv('UniLanguage/data/arc_frag_exp/train.csv')
sequence_lengths = [len(i) for i in data_df.sequences]

# set up batch sampler
base_sampler = SequentialSampler(dataset)
batch_sampler = BucketBatchSampler(base_sampler, approx_token=1000, sample_lengths=sequence_lengths, 
                             bucket_size=100, drop_last=False)

# build dataloader
dataloader = DataLoader(dataset=dataset, shuffle=False, sampler=None,
               batch_sampler=batch_sampler, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)

'''