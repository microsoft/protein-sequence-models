from typing import List, Any, Iterable
import random

import numpy as np
import torch
import torch.nn.functional as F

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD, GAP, START, STOP, MASK, MSA_PAD, PROTEIN_ALPHABET
from sequence_models.constants import ALL_AAS
from sequence_models.gnn import get_node_features, get_edge_features, get_mask, get_k_neighbors, replace_nan
from sequence_models.trRosetta_utils import trRosettaPreprocessing


def _pad(tokenized: List[torch.Tensor], value: int) -> torch.Tensor:
    """Utility function that pads batches to the same length."""
    batch_size = len(tokenized)
    max_len = max(len(t) for t in tokenized)
    output = torch.zeros((batch_size, max_len), dtype=tokenized[0].dtype) + value
    for row, t in enumerate(tokenized):
        output[row, :len(t)] = t
    return output


class BGCCollater(object):
    """A collater for BiGCARP models."""

    def __init__(self, tokens, pfam_to_domain):
        self.tokens = tokens
        self.pfam_to_domain = pfam_to_domain

    def __call__(self, domains):
        data = tuple(zip(*domains))
        sequences = data[0]
        t = []
        for sequence in sequences:
            tok = []
            for pfam in sequence.split(';'):
                if pfam in self.tokens['specials']:
                    tok.append(self.tokens['specials'][pfam])
                    continue
                if pfam in self.pfam_to_domain:
                    domain = self.pfam_to_domain[pfam]
                else:
                    domain = 'UNK'
                tok.append(self.tokens['domains'][domain])
            t.append(torch.tensor(tok))
        t = _pad(t, self.tokens['specials'][PAD])
        return (t, )


class TokenCollater(object):
    """A collater that pads batches of tokens."""

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[torch.Tensor]) -> List[torch.tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        sequences = _pad(sequences, self.pad_idx)
        return [sequences]


class SimpleCollater(object):
    """A collater that pads and possibly reverses batches of sequences.

    Parameters:
        alphabet (str)
        pad (Boolean)
        backwards (Boolean)

    If sequences are reversed, the padding is still on the right!

    Input (list): a batch of sequences as strings
    Output (torch.LongTensor): tokenized batch of sequences
    """

    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD, start=False, stop=False):
        self.pad = pad
        self.tokenizer = Tokenizer(alphabet)
        self.backwards = backwards
        self.pad_idx = self.tokenizer.alphabet.index(pad_token)
        self.start = start
        self.stop = stop

    def __call__(self, batch: List[Any], ) -> List[torch.Tensor]:
        data = tuple(zip(*batch))
        sequences = data[0]
        prepped = self._prep(sequences)
        return prepped

    def _prep(self, sequences):
        if self.start:
            sequences = [START + s for s in sequences]
        if self.stop:
            sequences = [s + STOP for s in sequences]
        if self.backwards:
            sequences = [s[::-1] for s in sequences]
        sequences = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in sequences]
        if self.pad:
            sequences = _pad(sequences, self.pad_idx)
        else:
            sequences = torch.stack(sequences)
        return (sequences,)


class TAPECollater(SimpleCollater):
    """Collater for TAPE datasets.

    For single-dimensional outputs, this pads the sequences and masks and batches everything.

    For ss, this also pads the output with -100.

    For contacts, this pads the contacts on the bottom and right.
    """

    def __init__(self, alphabet: str, pad=True, start=False, stop=False):
        super().__init__(alphabet, pad=pad, start=start, stop=stop)

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
    """Collater for autoregressive sequence models.

    Parameters:
        alphabet (str)
        pad (Boolean)
        backwards (Boolean)

    If sequences are reversed, the padding is still on the right!

    Input (list): a batch of sequences as strings
    Output:
        src (torch.LongTensor): START + input + padding
        tgt (torch.LongTensor): input + STOP + padding
        mask (torch.LongTensor): 1 where tgt is not padding
    """

    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD):
        super().__init__(alphabet, pad=pad)
        self.backwards = backwards
        self.pad_idx = self.tokenizer.alphabet.index(pad_token)


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
        src = _pad(src, self.pad_idx)
        tgt = _pad(tgt, self.pad_idx)
        mask = _pad(mask, 0)
        return src, tgt, mask


class AncestorCollater(LMCollater):
    """Collater for autoregressive sequence models with ancestors.

    Parameters:
        alphabet (str)
        pad (Boolean)
        backwards (Boolean)

    If sequences are reversed, the padding is still on the right!

    Input (list): a batch of sequences as strings
    Output:
        src (torch.LongTensor): START + input + STOP + ancestor  + padding
        tgt (torch.LongTensor): input + STOP + ancestor + STOP + padding
        mask (torch.LongTensor): 1 where tgt is not padding
    """

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
    """Collater for masked language sequence models.

    Parameters:
        alphabet (str)
        pad (Boolean)

    Input (list): a batch of sequences as strings
    Output:
        src (torch.LongTensor): corrupted input + padding
        tgt (torch.LongTensor): input + padding
        mask (torch.LongTensor): 1 where loss should be calculated for tgt
    """

    def __init__(self, alphabet: str, pad=False, backwards=False, pad_token=PAD, mut_alphabet=ALL_AAS, startstop=False):
        super().__init__(alphabet, pad=pad, backwards=backwards, pad_token=pad_token)
        self.mut_alphabet=mut_alphabet
        self.startstop = startstop

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
                    mod = np.random.choice([i for i in self.mut_alphabet if i != seq[idx]])
                else:  # mask
                    mod = MASK
                seq_mod[idx] = mod
            src.append(''.join(seq_mod))
            m = torch.zeros(len(seq_mod))
            m[mod_idx] = 1
            mask.append(m)
        if self.startstop:
            src = [START + s + STOP for s in src]
            tgt = [START + s + STOP for s in tgt]
            mask = [torch.cat([torch.zeros(1), m, torch.zeros(1)]) for m in mask]
        src = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in src]
        tgt = [torch.LongTensor(self.tokenizer.tokenize(s)) for s in tgt]
        pad_idx = self.tokenizer.alphabet.index(PAD)
        src = _pad(src, pad_idx)
        tgt = _pad(tgt, pad_idx)
        mask = _pad(mask, 0)
        return src, tgt, mask


class StructureCollater(object):
    """Collater for combined seq/str GNNs.

    Parameters:
        sequence collater (SimpleCollater)
        n_connections (int)
        n_node_features (int)
        n_edge_features (int)
        startstop (boolean): if true, expect the sequence collater to add starts/stops, and adds an
            extra zeroed node at the left of the graph.

    Input (list): a batch of sequences as strings
    Output:
        sequences from sequence_collater
        nodes, edges, connections, edge_mask for GNN
    """

    def __init__(self, sequence_collater: SimpleCollater, n_connections=20,
                 n_node_features=10, n_edge_features=11):
        self.sequence_collater = sequence_collater
        self.n_connections = n_connections
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

    def __call__(self, batch: List[Any], ) -> Iterable[torch.Tensor]:
        sequences, dists, omegas, thetas, phis = tuple(zip(*batch))
        collated_seqs = self.sequence_collater._prep(sequences)
        ells = [len(s) for s in sequences]
        max_ell = max(ells)
        n = len(sequences)
        nodes = torch.zeros(n, max_ell, self.n_node_features)
        edges = torch.zeros(n, max_ell, self.n_connections, self.n_edge_features)
        connections = torch.zeros(n, max_ell, self.n_connections, dtype=torch.long)
        edge_mask = torch.zeros(n, max_ell, self.n_connections, 1)
        for i, (ell, dist, omega, theta, phi) in enumerate(zip(ells, dists, omegas, thetas, phis)):
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
    """Collater that batches sequences and ell x ell structure targets.

    Currently cannot deal with starts/stops!
    """

    def __init__(self, sequence_collater: SimpleCollater, exp=True, dist_only=False):
        self.exp = exp
        self.sequence_collater = sequence_collater
        self.dist_only = dist_only

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
        masks = [~torch.isnan(omega) & m for omega, m in zip(omegas, masks)]
        masks = [~torch.isnan(theta) & m for theta, m in zip(thetas, masks)]
        masks = [~torch.isnan(phi) & m for phi, m in zip(phis, masks)]
        masks = self._pad(masks, ells, value=False)
        dists = self._pad(dists, ells)
        if self.dist_only:
            return seqs, dists, masks
        omegas = self._pad(omegas, ells)
        thetas = self._pad(thetas, ells)
        phis = self._pad(phis, ells)
        return seqs, dists, omegas, thetas, phis, masks

    
class TAPE2trRosettaCollater(SimpleCollater):
    """Does trRosetta preprocessing for TAPE datasets. """

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
    

class MSAStructureCollater(StructureOutputCollater):
    """Collater that batches msas and ell x ell structure targets.

    Currently cannot deal with starts/stops!

    MSAs should be pre-tokenized.
    """

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch: List[Any], ) -> Iterable[torch.Tensor]:
        msas, dists, omegas, thetas, phis = tuple(zip(*batch))
        ells = [s.shape[1] for s in msas]
        max_ell = max(ells)
        msas = [F.pad(msa, [0, max_ell - ell], value=self.pad_idx).long() for msa, ell in zip(msas, ells)]
        masks = [torch.ones_like(dist).bool() for dist in dists]
        masks = self._pad(masks, ells, value=False)
        dists = self._pad(dists, ells)
        omegas = self._pad(omegas, ells)
        thetas = self._pad(thetas, ells)
        phis = self._pad(phis, ells)
        return msas, dists, omegas, thetas, phis, masks


class MSAGapCollater(object):

    def __init__(self, sequence_collater, n_connections=30, direction='bidirectional', task='gap-prob'):
        """Collater for gap probability prediction with a GNN. y is (p_gap, 1 - p_gap).

        Uses MASK to pad to distinguish between GAP and padding. 
        For bidirectional:
            src: <seq><MASKs>
            tgt: <pre><mask this out>
        
        For forward: 
            src: <START><seq><MASKs>
            tgt:        <seq><MASKs>
        
        for backward:
            src: <MASKS><seq><START>
            tgt: <MASKS><seq>

            
        Args:
            sequence_collater: should only return src
            direction (str)
            n_connections (int)
            task (str): gap-prob or ar

        Returns:
            seqs, anchor_seq, nodes, edges, connections, edge_mask, y, mask_y
        """
        # collaters
        self.sequence_collater = sequence_collater
        self.structure_collater = StructureCollater(self.sequence_collater,
                                                    n_connections=n_connections)
        self.direction = direction
        self.pad_idx = sequence_collater.tokenizer.alphabet.index(MASK)
        if direction != 'bidirectional':
            self.start_idx = sequence_collater.tokenizer.alphabet.index(START)
        self.task = task
        self.gap_idx = sequence_collater.tokenizer.alphabet.index(GAP)

    def __call__(self, batch: List[Any], ) -> Iterable[torch.Tensor]:
        seq, anchor_seq, dist, omega, theta, phi, y, y_mask = tuple(zip(*batch))
        anchor_seq = _pad(anchor_seq, self.pad_idx)
        seq = [self.sequence_collater.tokenizer.untokenize(i.numpy()) for i in seq]
        rebatch = [(seq[i], dist[i], omega[i], theta[i], phi[i]) for i in range(len(seq))]
        seqs, nodes, edges, connections, edge_mask = self.structure_collater.__call__(rebatch)

        # If backward, reverse everything
        if self.direction != 'bidirectional':
            if self.direction == 'backward':
                d1_pad = [0, 1]
                node_pad = [0, 0, 0, 1]
                edge_pad = [0, 0] + node_pad
            if self.direction == 'forward':
                d1_pad = [1, 0]
                node_pad = [0, 0, 1, 0]
                edge_pad = [0, 0] + node_pad
                connections = connections + 1
            seqs = F.pad(seqs, d1_pad, value=self.start_idx)
            anchor_seq = F.pad(anchor_seq, d1_pad, value=self.start_idx)
            nodes = F.pad(nodes, node_pad, value=0.0)
            edges = F.pad(edges, edge_pad, value=0.0)
            connections = F.pad(connections, node_pad, value=0)
            edge_mask = F.pad(edge_mask, edge_pad, value=0.0)

        X = (seqs, anchor_seq, nodes, edges, connections, edge_mask)

        if self.task == 'gap-prob':
            y = _pad(y, 0)
            mask_y = [torch.ones_like(i).bool() for i in y]            
            mask_y = _pad(mask_y, False)    
            if self.direction != 'bidirectional':
                y = F.pad(y, [0, 1, 0, 0], value=0)
                mask_y = F.pad(mask_y, [0, 1, 0, 0], value=False)
            # adjust y format to fit kldivloss
            y = y.unsqueeze(-1)
            y = torch.cat((y, torch.ones_like(y) - y), -1)
        else:
            y = (seqs[:, 1:] == self.gap_idx).long()
            y = F.pad(y, d1_pad)
            mask_y = (seqs[:, 1:] != self.pad_idx).float()
            mask_y = F.pad(mask_y, d1_pad)


        return X + (y, mask_y)


class Seq2PropertyCollater(SimpleCollater):
    """A collater that batches sequences and a 1d target. """

    def __init__(self, alphabet: str, pad=True, scatter=False, return_mask=False, start=False, stop=False):
        super().__init__(alphabet, pad=pad, start=start, stop=stop)
        self.scatter = scatter
        self.mask = return_mask

    def __call__(self, batch):
        data = tuple(zip(*batch))
        sequences = data[0]
        prepped = self._prep(sequences)[0]
        if self.mask:
            mask = prepped != self.tokenizer.alphabet.index(PAD)
        if self.scatter:
            prepped = F.one_hot(prepped, len(self.tokenizer.alphabet))

        y = data[1]
        y = torch.tensor(y).unsqueeze(-1).float()
        if not self.mask:
            return prepped, y
        else:
            return prepped, y, mask


def _pad_msa(tokenized: List, num_seq: int, max_len: int, value: int) -> torch.Tensor:
    """Utility function that pads batches to the same length."""
    batch_size = len(tokenized)
    num_seq = max([len(m) for m in tokenized])
    output = torch.zeros((batch_size, num_seq, max_len), dtype=torch.long) + value
    for i in range(batch_size):
        tokenized[i] = torch.LongTensor(np.array(tokenized[i]))
        output[i, :len(tokenized[i]), :len(tokenized[i][0])] = tokenized[i]
    return output


class MSAAbsorbingCollater(object):
    """Collater for MSA Absorbing Diffusion model.
    Based on implementation described by Hoogeboom et al. in "Autoregressive Diffusion Models"
    https://doi.org/10.48550/arXiv.2110.02037

    Parameters:
        alphabet: str,
            protein alphabet to use
        pad_token: str,
            pad_token to use to pad MSAs, default is PAD token from sequence_models.constants
        num_seqs: int,
            number of sequences to include in each MSA

    Input (list): a batch of Multiple Sequence Alignments (MSAs), each MSA contains 64 sequences
    Output:
        src (torch.LongTensor): corrupted input + padding
        tgt (torch.LongTensor): input + padding
        mask (torch.LongTensor): 1 where tgt is not padding
    """

    def __init__(self, alphabet: str, pad_token=MSA_PAD, num_seqs=64, bert=False):
        self.tokenizer = Tokenizer(alphabet)
        self.pad_idx = self.tokenizer.alphabet.index(pad_token)
        self.num_seqs = num_seqs
        self.bert = bert
        if bert:
            self.choices = [self.tokenizer.alphabet.index(a) for a in PROTEIN_ALPHABET + GAP]

    def __call__(self, batch_msa):
        tgt = list(batch_msa)
        src = tgt.copy()

        longest_msa = 0
        batch_size = len(batch_msa)
        mask_ix = []
        mask_iy = []
        for i in range(batch_size):
            # Tokenize MSA
            tgt[i] = [self.tokenizer.tokenize(s) for s in tgt[i]]
            src[i] = [self.tokenizer.tokenize(s) for s in src[i]]

            curr_msa = src[i]

            curr_msa = np.asarray(curr_msa)
            length, depth = curr_msa.shape  # length = number of seqs in MSA, depth = # AA in MSA

            curr_msa = curr_msa.flatten()  # Flatten MSA to 1D to mask tokens
            d = len(curr_msa)  # number of residues in MSA
            if not self.bert:
                t = np.random.choice(d)  # Pick timestep t
                t += 1  # ensure t cannot be 0
                num_masked_tokens = d - t + 1
                mask_idx = np.random.choice(d, num_masked_tokens, replace=False)
            else:
                num_corr_tokens = int(np.round(0.15 * d))
                corr_idx = np.random.choice(d, num_corr_tokens, replace=False)
                num_masked_tokens = int(np.round(0.8 * num_corr_tokens))
                num_mut_tokens = int(np.round(0.1 * num_corr_tokens))
                mask_idx = corr_idx[:num_masked_tokens]
                muta_idx = corr_idx[-num_mut_tokens:]
                for idx in muta_idx:
                    choices = list(set(self.choices) - set(curr_msa[[idx]]))
                    curr_msa[idx] = np.random.choice(choices)
                mask_ix.append(corr_idx // depth)
                mask_iy.append(corr_idx % depth)

            curr_msa[mask_idx] = self.tokenizer.mask_id
            curr_msa = curr_msa.reshape(length, depth)

            src[i] = list(curr_msa)

            longest_msa = max(depth, longest_msa)  # Keep track of the longest MSA for padding

        # Pad sequences
        src = _pad_msa(src, self.num_seqs, longest_msa, self.pad_idx)
        tgt = _pad_msa(tgt, self.num_seqs, longest_msa, self.pad_idx)
        if self.bert:
            mask = torch.zeros_like(src)
            for i in range(len(mask_ix)):
                mask[i, mask_ix[i], mask_iy[i]] = 1
            mask = mask.bool()
        else:
            mask = (src == self.tokenizer.mask_id)

        return src, tgt, mask