import pandas as pd
import numpy as np
import math
import random
from copy import copy

from torch.utils.data import Dataset, DataLoader
import torch
from torch.utils.data.sampler import SequentialSampler, RandomSampler, Sampler, \
    BatchSampler, SubsetRandomSampler

from torchnlp.samplers.sorted_sampler import SortedSampler
from torchnlp.utils import identity

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD, PROTEIN_ALPHABET


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
                    self.longest_token = 0
                    yield self.batch
    
    def __len__(self):
        return len(self.sampler)
    
def mask_collate_fn(batch):
    batch_mod = []
    batch_mod_bool = []
    for seq in batch:
        mod_idx = random.sample(list(range(len(seq))), int(len(seq)*0.15))
        if len(mod_idx) == 0:
            mod_idx = np.random.choice(list(range(len(seq))))# make sure at least one aa is chosen
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
        
    #padding
    batch_mod = pad(batch_mod, PROTEIN_ALPHABET.index('-'))
    batch = pad(batch, PROTEIN_ALPHABET.index('-'))
    batch_mod_bool = pad(batch_mod_bool, False)
    
    # rename and convert to tensor
    src = torch.from_numpy(np.array(batch_mod)).long()
    tgt = torch.from_numpy(np.array(batch)).long()
    loss_mask = torch.from_numpy(np.array(batch_mod_bool)).bool()
    
    return src, tgt, loss_mask 