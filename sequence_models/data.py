from typing import List, Any

import torch
from torch.utils.data import Dataset
import pandas as pd

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD


class CSVDataset(Dataset):

    def __init__(self, fpath: str, alphabet: str, split=None, outputs=[], pad=False):
        self.data = pd.read_csv(fpath)
        if split is not None:
            self.data = self.data[self.data['split'] == split]
        self.tokenizer = Tokenizer(alphabet)
        self.outputs = outputs
        self.data = self.data[['sequence'] + self.outputs]
        self.pad = pad
        if pad:
            self.pad_idx = alphabet.index(PAD)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        tokenized = self.tokenizer.tokenize(row['sequence'])
        return [tokenized, *row[self.outputs]]

    def collate_fn(self, batch: List[Any]) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        if self.pad:
            sequences = _pad(sequences, self.pad_idx)
        else:
            sequences = torch.LongTensor(sequences)
        data = (torch.tensor(d) for d in data[1:])
        return [sequences, *data]


def _pad(tokenized: List[torch.Tensor], value: int) -> torch.Tensor:
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    output = torch.zeros((batch_size, max_len), dtype=tokenized[0].dtype) + value
    for row, t in enumerate(tokenized):
        output[row, :len(t)] = t
    return output


