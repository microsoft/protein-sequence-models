import re

import numpy as np
import pandas as pd

from torch.utils.data import Dataset


def load_flip_data(data_fpath, dataset, split, max_len=2048):
    """returns dataframe of train, (val), test sets"""
    datadir = data_fpath + dataset + '/splits/'
    path = datadir + split + '.csv'
    df = pd.read_csv(path)
    df['sequence'] = df.sequence.apply(lambda s: re.sub(r'[^A-Z]', '', s.upper()))  # remove special characters
    test = df[df.set == 'test']
    train = df[(df.set == 'train') & (df.validation.isnull())]
    valid = df[~df.validation.isnull()]
    return FlipDataset(train, max_len=max_len), FlipDataset(valid, max_len=max_len), FlipDataset(test, max_len=max_len)


class FlipDataset(Dataset):

    def __init__(self, data, max_len=2048):
        self.sequences = data['sequence'].values
        self.targets = data['target'].values
        self.max_len = max_len

    def __getitem__(self, idx):
        s = self.sequences[idx]
        if len(s) > self.max_len:
            start = np.random.choice(len(s) - self.max_len)
            s = s[start: start + self.max_len]
        return s, self.targets[idx]

    def __len__(self):
        return len(self.sequences)