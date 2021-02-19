from typing import List, Any, Iterable
import random

import numpy as np
import torch
import torch.nn.functional as F

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD, START, STOP, MASK, 
from sequence_models.constants import ALL_AAS
from sequence_models.gnn import get_node_features, get_edge_features, get_mask, get_k_neighbors, replace_nan


class SimpleCollater(object):

    def __init__(self, alphabet: str, pad=False, backwards=False):
        self.pad = pad
        self.tokenizer = Tokenizer(alphabet)
        self.backwards = backwards

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        if self.backwards:
            sequences = [s[::-1] for s in sequences]
        prepped = self._prep(sequences)
        return prepped

    def _prep(self, sequences):
        sequences = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]
        if self.pad:
            pad_idx = self.tokenizer.alphabet.index(PAD)
            sequences = _pad(sequences, pad_idx)
        else:
            sequences = torch.stack(sequences)
        return (sequences,)


class TAPECollater(SimpleCollater):

    def __init__(self, alphabet: str, pad=True):
        super().__init__(alphabet, pad=pad)

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        prepped = self._prep(sequences)
        y = data[1]
        mask = data[2]
        if isinstance(y[0], float):
            y = torch.tensor(y).unsqueeze(-1)

        elif isinstance(y[0], int):
            y = torch.tensor(y)

        elif len(y[0].shape) == 1:  # secondary structure
            # mask = [torch.ones_like(yi) for yi in y]
            # mask = _pad(mask, -100)
            y = _pad(y, -100).long()

        elif len(y[0].shape) == 2:  # contact
            max_len = max(len(i) for i in y)
            mask = [F.pad(mask_i,
                      (0, max_len - len(mask_i), 0, max_len - len(mask_i)), value=False) for mask_i in mask]
            mask = torch.stack(mask, dim=0)
            y = [F.pad(yi, (0, max_len - len(yi), 0, max_len - len(yi))) for yi in y]
            y = torch.stack(y, dim=0).float()

        return prepped[0], y, mask


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
        max_ell = max(ells)
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
            dist.fill_diagonal_(np.nan)
            E_idx = get_k_neighbors(dist, self.n_connections)
            E = get_edge_features(dist, omega, theta, phi, E_idx)
            str_mask = get_mask(E)
            E = replace_nan(E)
            V = replace_nan(V)
            # reshape
            nc = min(ell - 1, self.n_connections)
            nodes[i, :ell] = V
            edges[i, :ell, :nc] = E
            connections[i, :ell, :nc] = E_idx
            str_mask = str_mask.view(1, ell, -1)
            edge_mask[i, :ell, :nc, 0] = str_mask
        return (*collated_seqs, nodes, edges, connections, edge_mask)


class StructureOutputCollater(object):

    def __init__(self, sequence_collater: SimpleCollater, exp=True):
        self.exp = exp
        self.sequence_collater = sequence_collater

    def _pad(self, squares, ells, value=0.0):
        max_len = max(ells)
        squares = [F.pad(d, [0, max_len - ell, 0, max_len - ell], value=value)
                   for d, ell in zip(squares, ells)]
        squares = torch.stack(squares, dim=0)
        return squares

    def __call__(self, batch: List[Any], ) -> Iterable[torch.Tensor]:
        sequences, dists, omegas, thetas, phis = tuple(zip(*batch))
        ells = [len(s) for s in sequences]
        seqs = self.sequence_collater._prep(sequences)[0]
        if self.exp:
            dists = [torch.exp(-d ** 2 / 64) for d in dists]
            masks = [~torch.isnan(dist) for dist in dists]
        else:
            masks = [torch.ones_like(dist).bool() for dist in dists]
        masks = self._pad(masks, ells, value=False)
        dists = self._pad(dists, ells)
        omegas = self._pad(omegas, ells)
        thetas = self._pad(thetas, ells)
        phis = self._pad(phis, ells)
        return seqs, dists, omegas, thetas, phis, masks

    
class TAPE2trRosettaCollater(SimpleCollater):

    def __init__(self, alphabet: str, pad=True):
        super().__init__(alphabet, pad=pad)
        self.featurization = trRosettaPreprocessing(alphabet)

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        if len(data) == 0:
            return data
        sequences = data[0]
        sequences = [i.replace('X', '-') for i in sequences] # get rid of X found in secondary_stucture data
        lens = [len(i) for i in sequences]
        max_len = max(lens)
        prepped = self._prep(sequences)[0]
        prepped = torch.stack([self.featurization.process(i.view(1,-1)).squeeze(0) for i in prepped])
        y = data[1]
        tgt_mask = data[2]
        src_mask = [torch.ones(i, i).bool() for i in lens]
        src_mask = [F.pad(mask_i,
                          (0, max_len - len(mask_i), 0, max_len - len(mask_i)), value=False) for mask_i in src_mask]
        src_mask = torch.stack(src_mask, dim=0).unsqueeze(1)
        
        if isinstance(y[0], float): # stability or fluorescence
            y = torch.tensor(y).unsqueeze(-1)
            tgt_mask = torch.ones_like(y)

        elif isinstance(y[0], int): # remote homology
            y = torch.tensor(y).long()
            tgt_mask = torch.ones_like(y)

        elif len(y[0].shape) == 1:  # secondary structure
            tgt_mask = [torch.ones(i) for i in lens]
            y = _pad(y, 0).long()
            tgt_mask = _pad(tgt_mask, 0).long()
            
        elif len(y[0].shape) == 2:  # contact
            max_len = max(len(i) for i in y)
            tgt_mask = [F.pad(mask_i,
                      (0, max_len - len(mask_i), 0, max_len - len(mask_i)), value=False) for mask_i in tgt_mask]
            tgt_mask = torch.stack(tgt_mask, dim=0)
            y = [F.pad(yi, (0, max_len - len(yi), 0, max_len - len(yi)), value=-1) for yi in y]
            y = torch.stack(y, dim=0).long()
        return prepped.float(), y, tgt_mask, src_mask
    

class MSAGapCollater(object):

    def __init__(self, alphabet, input_type, output_type, n_connections=20):
        """
        Args:
            alphabet: str,
                protein alphabet
            input_type: str
                type of input data, 'structure' or 'sequence'
            output_type:
                type of output data, 'gap_prob' or 'lm'
            n_connections: int
                if structure is input data type, number of connections
        """
        # collaters
        self.simple_collater = SimpleCollater(alphabet=alphabet, pad=True)
        self.structure_collater = StructureCollater(self.simple_collater, n_connections=n_connections)

        # data type
        self.input_type = input_type
        self.output_type = output_type

        # grab tokenizer just in case
        self.tokenizer = Tokenizer(alphabet)

    def __call__(self, batch: List[Any], ) -> Iterable[torch.Tensor]:
        if self.input_type == 'structure':
            seq, dist, omega, theta, phi, y, y_mask = tuple(zip(*batch))
            seq = [self.tokenizer.untokenize(i.numpy()) for i in seq]
            rebatch = [(seq[i], dist[i], omega[i], theta[i], phi[i]) for i in range(len(seq))]
            seqs, nodes, edges, connections, edge_mask = self.structure_collater.__call__(rebatch)
            mask_x = None
            X = (seqs, nodes, edges, connections, edge_mask)

        elif self.input_type == 'sequence':  # by design, y is gap prob
            seq, y = tuple(zip(*batch))
            mask_x = [torch.ones_like(i).bool() for i in seq]
            seq = _pad(seq, 20)
            mask_x = _pad(mask_x, False)
            X = seq

        if (self.output_type == 'gab-prob'):
            mask_y = [torch.ones_like(i).bool() for i in y]
            y = _pad(y, 0)
            mask_y = _pad(mask_y, False)
        return X + (mask_x, y, mask_y)