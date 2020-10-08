from typing import List

import torch.nn as nn


class PositionFeedForward(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionFeedForward2d(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = nn.Conv2d(d_in, d_out, 1)

    def forward(self, x):
        return self.conv(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


class FCStack(nn.Sequential):
    """A stack of fully-connected layers.

     Every nn.Linear is optionally followed by  a normalization layer,
     a dropout layer, and then a ReLU.

     Args:
         sizes (List of ints): the all layer dimensions from input to output
         norm (str): type of norm. 'bn' for batchnorm, 'ln' for layer norm. Default 'bn'
         p (float): dropout probability

     Input (N, sizes[0])
     Output (N, sizes[-1])
     """

    def __init__(self, sizes: List[int], norm='bn', p=0.0):
        layers = []
        for d0, d1 in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(d0, d1))
            if norm == 'ln':
                layers.append(nn.LayerNorm(d1))
            elif norm == 'bn':
                layers.append(nn.BatchNorm1d(d1))
            if p != 0:
                layers.append(nn.Dropout(p))
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)
