from typing import List

import torch.nn as nn

from sequence_models.convolutional import MaskedConv1d, MaskedCausalConv1d


class PositionFeedForward(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.conv = nn.Conv1d(d_in, d_out, 1)

    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099)."""

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, causal=False):
        super().__init__()
        if causal:
            self.conv = MaskedCausalConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        else:
            self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        layers1 = [
            nn.LayerNorm(d_in),
            nn.ReLU(),
            PositionFeedForward(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.ReLU()
            ]
        layers2 = [
            nn.LayerNorm(d_h),
            nn.ReLU(),
            PositionFeedForward(d_h, d_out),
            ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        return x + self.sequence2(
            self.conv(self.sequence1(x), input_mask=input_mask)
        )


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
