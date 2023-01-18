from typing import List

import math
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class DoubleEmbedding(nn.Module):

    """Embedding layer that allows some frozen and some trainable embeddings.

    An embedding layer where the first n_trainable embeddings are trainable and the
    remaining n_frozen embeddings are frozen.
    """

    def __init__(self, n_trainable, n_frozen, embedding_dim, padding_idx=None):
        super().__init__()
        if padding_idx is None:
            train_padding_idx = None
            freeze_padding_idx = None
        elif padding_idx < n_trainable:
            train_padding_idx = padding_idx
            freeze_padding_idx = None
        else:
            train_padding_idx = None
            freeze_padding_idx = padding_idx - n_trainable
        self.n_trainable = n_trainable
        self.embedding_dim = embedding_dim
        self.trainable = nn.Embedding(n_trainable, embedding_dim, padding_idx=train_padding_idx)
        self.frozen = nn.Embedding(n_frozen, embedding_dim, padding_idx=freeze_padding_idx)
        self.frozen.weight.requires_grad = False

    def forward(self, idx):
        i = torch.where(idx < self.n_trainable)
        j = torch.where(idx >= self.n_trainable)
        b, ell = idx.shape
        e = torch.empty(b, ell, self.embedding_dim, device=idx.device, dtype=self.trainable.weight.dtype)
        e[i] = self.trainable(idx[i])
        e[j] = self.frozen(idx[j] - self.n_trainable)
        return e


class FactorizedLinear(nn.Module):

    def __init__(self, d_in, d_out, rank):
        super().__init__()
        layer = nn.Linear(d_in, d_out)
        w = layer.weight.data
        self.bias = layer.bias
        u, s, v = torch.svd(w)
        s = torch.diag(s[:rank].sqrt())
        u = u[:, :rank]
        v = v.t()[:rank]
        self.u = nn.Parameter((u @ s).t())
        self.v = nn.Parameter((s @ v).t())

    def forward(self, x):
        return x @ self.v @ self.u + self.bias


class PositionFeedForward(nn.Module):

    def __init__(self, d_in, d_out, rank=None):
        super().__init__()
        if rank is None:
            self.conv = nn.Conv1d(d_in, d_out, 1)
            self.factorized = False
        else:
            layer = nn.Linear(d_in, d_out)
            w = layer.weight.data
            self.bias = layer.bias
            u, s, v = torch.svd(w)
            s = torch.diag(s[:rank].sqrt())
            u = u[:, :rank]
            v = v.t()[:rank]
            self.u = nn.Parameter(u @ s)
            self.v = nn.Parameter(s @ v)
            self.factorized = True

    def forward(self, x):
        if self.factorized:
            w = self.u @ self.v
            return x @ w.t() + self.bias
        else:
            return self.conv(x.transpose(1, 2)).transpose(1, 2)


class PositionFeedForward2d(nn.Module):

    def __init__(self, d_in, d_out):
        super().__init__()
        self.dense = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.dense(x)


class MaskedInstanceNorm2d(nn.InstanceNorm2d):
    ### Expects square inputs before and after masking!!

    def __init__(self, n_dims, affine=True, eps=1e-6):
        super().__init__(n_dims, affine=affine, eps=eps)


    def forward(self, x, input_mask=None):
        if input_mask is None:
            return super().forward(x)
        input_mask = input_mask.bool()
        normed = []
        _, _, max_len, _ = x.shape
        for input, mask in zip(x, input_mask):
            input = torch.masked_select(input, mask)
            el = int(np.sqrt(input.shape[0] // self.num_features))
            input = input.reshape(1, self.num_features, el, el)
            n = max_len - el
            normed.append(F.pad(super().forward(input), (0, n, 0, n), value=0))
        return torch.cat(normed, dim=0)


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


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, x.size(0), :]
        return self.dropout(x)
