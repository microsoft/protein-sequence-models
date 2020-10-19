from typing import List, Any, Iterable
import random
import math
import subprocess
import string
import json
from os import path
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
import pandas as pd

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD, START, STOP, MASK
from sequence_models.constants import ALL_AAS, trR_ALPHABET
from sequence_models.gnn import get_node_features, get_edge_features, get_mask, get_k_neighbors, replace_nan, bins_to_vals

class SimpleCollater(object):

    def __init__(self, alphabet: str, pad=False):
        self.pad = pad
        self.tokenizer = Tokenizer(alphabet)

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        prepped = self._prep(sequences)
        return prepped

    def _prep(self, sequences):
        sequences = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]
        if self.pad:
            pad_idx = self.tokenizer.alphabet.index(PAD)
            sequences = _pad(sequences, pad_idx)
        else:
            sequences = torch.stack(sequences)
        return (sequences, )


class LMCollater(SimpleCollater):

    def __init__(self, alphabet: str, pad=False, backwards=False):
        super().__init__(alphabet, pad=pad)
        self.backwards = backwards

    def _prep(self, sequences):
        return self._tokenize_and_mask(*self._split(sequences))

    def _split(self, sequences):
        if not self.backwards:
            src = [START + s for s in sequences]
            tgt = [s + STOP for s in sequences]
        else:
            src = [STOP + s[::-1] for s in sequences]
            tgt = [s[::-1] + START for s in sequences]
        return src, tgt

    def _tokenize_and_mask(self, src, tgt):
        src = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in src]
        tgt = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt]
        mask = [torch.ones_like(t) for t in tgt]
        pad_idx = self.tokenizer.alphabet.index(PAD)
        src = _pad(src, pad_idx)
        tgt = _pad(tgt, pad_idx)
        mask = _pad(mask, 0)
        return src, tgt, mask


def _pad(tokenized: List[torch.Tensor], value: int) -> torch.Tensor:
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    output = torch.zeros((batch_size, max_len), dtype=tokenized[0].dtype) + value
    for row, t in enumerate(tokenized):
        output[row, :len(t)] = t
    return output


class AncestorCollater(LMCollater):

    def __call__(self, batch):
        data = tuple(zip(*batch))
        sequences, ancestors = data[:2]
        prepped = self._prep(sequences, ancestors)
        return prepped

    def _prep(self, sequences, ancestors):
        if self.backwards:
            sequences = [s[::-1] for s in sequences]
            ancestors = [a[::-1] for a in ancestors]
        src = [START + s + STOP + a for s, a in zip(sequences, ancestors)]
        tgt = [s + STOP + a + STOP for s, a in zip(sequences, ancestors)]
        return self._tokenize_and_mask(src, tgt)


class MLMCollater(SimpleCollater):

    def _prep(self, sequences):
        tgt = list(sequences[:])
        src = []
        mask = []
        for seq in sequences:
            if len(seq) == 0:
                tgt.remove(seq)
                continue
            mod_idx = random.sample(list(range(len(seq))), int(len(seq) * 0.15))
            if len(mod_idx) == 0:
                mod_idx = [np.random.choice(len(seq))]  # make sure at least one aa is chosen
            seq_mod = list(seq)
            for idx in mod_idx:
                p = np.random.uniform()
                if p <= 0.10:  # do nothing
                    mod = seq[idx]
                elif 0.10 < p <= 0.20:  # replace with random amino acid
                    mod = np.random.choice([i for i in ALL_AAS if i != seq[idx]])
                else:  # mask
                    mod = MASK
                seq_mod[idx] = mod
            src.append(''.join(seq_mod))
            m = torch.zeros(len(seq_mod))
            m[mod_idx] = 1
            mask.append(m)
        src = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in src]
        tgt = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt]
        pad_idx = self.tokenizer.alphabet.index(PAD)
        src = _pad(src, pad_idx)
        tgt = _pad(tgt, pad_idx)
        mask = _pad(mask, 0)
        return src, tgt, mask


class StructureImageCollater(object):

    def __init__(self, sequence_collater: SimpleCollater, backwards=False):
        self.sequence_collater = sequence_collater
        self.backwards = backwards

    def __call__(self, batch: List[Any]) -> Iterable[torch.Tensor]:
        sequences, dists, omegas, thetas, phis = tuple(zip(*batch))
        collated_seqs = self.sequence_collater._prep(sequences)
        ells = [len(s) for s in sequences]
        max_ell = max(ells)
        n = len(sequences)
        structure = torch.zeros(n, max_ell, max_ell, 4)
        structure_mask = torch.zeros(n, max_ell, max_ell)
        for i, (dist, omega, theta, phi, ell) in enumerate(zip(dists, omegas, thetas, phis, ells)):
            st = torch.stack([dist, omega, theta, phi], dim=-1)  # ell, ell, 4
            if self.backwards:
                st = torch.flip(st, [0, 1])
            structure[i, :ell, :ell, :] = st
            structure_mask[i, :ell, :ell] = 1.0
        structure[torch.isnan(structure)] = 0.0
        return (*collated_seqs, structure, structure_mask)


class StructureCollater(object):

    def __init__(self, sequence_collater: SimpleCollater, n_connections=20, backwards=False):
        self.sequence_collater = sequence_collater
        self.n_connections = n_connections
        self.backwards = backwards

    def __call__(self, batch: List[Any], ) -> Iterable[torch.Tensor]:
        sequences, dists, omegas, thetas, phis = tuple(zip(*batch))
        collated_seqs = self.sequence_collater._prep(sequences)
        ells = [len(s) for s in sequences]
        max_ell = max(ells) + 1
        n = len(sequences)
        nodes = torch.zeros(n, max_ell, 10)
        edges = torch.zeros(n, max_ell, self.n_connections, 6)
        connections = torch.zeros(n, max_ell, self.n_connections, dtype=torch.long)
        edge_mask = torch.zeros(n, max_ell, self.n_connections, 1)
        for i, (ell, dist, omega, theta, phi) in enumerate(zip(ells, dists, omegas, thetas, phis)):
            if self.backwards:
                dist = torch.flip(dist, [0, 1])
                omega = torch.flip(omega, [0, 1])
                theta = torch.flip(theta, [0, 1])
                phi = torch.flip(phi, [0, 1])
            # process features
            V = get_node_features(omega, theta, phi)
            E_idx = get_k_neighbors(dist, self.n_connections)
            E = get_edge_features(dist, omega, theta, phi, E_idx)
            str_mask = get_mask(E)
            E = replace_nan(E)
            V = replace_nan(V)
            # reshape
            nc = min(ell - 1, self.n_connections)
            nodes[i, 1: ell + 1] = V
            edges[i, 1: ell + 1, :nc] = E
            connections[i, 1: ell + 1, :nc] = E_idx
            str_mask = str_mask.view(1, ell, -1)
            edge_mask[i, 1: ell + 1, :nc, 0] = str_mask
        return (*collated_seqs, nodes, edges, connections, edge_mask)


class TAPECollater(SimpleCollater):

    def __init__(self, alphabet: str, pad=True,):
        super().__init__(alphabet, pad=pad)

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        prepped = self._prep(sequences)
        y = data[1]
        return prepped, y
