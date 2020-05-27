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
    

class BucketSampler(Sampler):

    def __init__(self, sequence_lengths, bucket_size, sort_key=identity):
        self.data = sequence_lengths
        self.bucket_size = bucket_size
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]
        
    def __iter__(self):
        bucket = []
        for idx in self.sorted_indexes:
            bucket.append(idx)
            if len(bucket) == self.bucket_size:
                yield bucket
                bucket = []
        if len(bucket) > 0:
            yield bucket    
    def __len__(self):
        return len(self.data)

class ApproxBatchSampler(BatchSampler):
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
	'''

	def __init__(self, sampler, approx_token, sample_lengths,):
		self.longest_token = 0
		self.approx_token = approx_token
		self.sample_lengths = sample_lengths
		self.sampler = sampler
		self.batch = []

	def __iter__(self):
		for bucket_idx in RandomSampler(list(self.sampler)): # get random bucket
			bucket = list(self.sampler)[bucket_idx]
			self.batch = []
			for single_sample in SubsetRandomSampler(bucket): # get random sample in bucket
				if self.longest_token == 0:
					self.batch = []
				self.batch.append(single_sample) # fill batch until approx_token is met
				self.longest_token = max(self.longest_token, self.sample_lengths[single_sample])
				if self.longest_token * len(self.batch) >= self.approx_token:
					yield_batch = [(i, self.longest_token) for i in self.batch]
					self.longest_token = 0
					yield yield_batch
	
	def __len__(self):
		return len(self.sampler)
"""
# Example

# load Dataset 
dataset = CSVDataset('UniLanguage/data/arc_frag_exp/train.csv', PROTEIN_ALPHABET, '-')

# extract sequence lengths
data_df = pd.read_csv('UniLanguage/data/arc_frag_exp/train.csv')
sequence_lengths = [len(i) for i in data_df.sequences]

# build bucket_sampler
bucket_sampler = BucketSampler(sequence_lengths, 100)

# build batch_sampler
batch_sampler = ApproxBatchSampler(bucket_sampler, approx_token=1000, sample_lengths= sequence_lengths)

# build dataloader
dataloader = DataLoader(dataset=dataset, shuffle=False, sampler=None,
               batch_sampler=batch_sampler, num_workers=0, collate_fn=None,
               pin_memory=False, drop_last=False, timeout=0,
               worker_init_fn=None)

"""