from typing import Iterable

import numpy as np
import pandas as pd
import os
import wget
import json
from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sequence_models.constants import STOP, START, MASK, PAD
from sequence_models.constants import PROTEIN_ALPHABET, AAINDEX_ALPHABET


def warmup(n_warmup_steps):
    def get_lr(step):
        return min((step + 1) / n_warmup_steps, 1.0)
    return get_lr


def get_metrics(fname):
    with open(fname) as f:
        lines = f.readlines()
    valid_lines = []
    train_lines = []
    for i, line in enumerate(lines):
        if 'Training' in line and 'loss' in line:
            last_train = line
        if 'Validation complete' in line:
            valid_lines.append(lines[i - 1])
            train_lines.append(last_train)
    metrics = []
    for t, v in zip(train_lines, valid_lines):
        step = int(t.split()[6])
        t_loss = float(t.split()[13])
        t_accu = float(t.split()[16][:6])
        v_loss = float(v.split()[13])
        v_accu = float(v.split()[16][:6])
        metrics.append((step, t_loss, t_accu, v_loss, v_accu))
    metrics = pd.DataFrame(metrics, columns=['step', 'train_loss', 'train_accu', 'valid_loss', 'valid_accu'])
    return metrics


def get_weights(seqs):
    scale = 1.0
    theta = 0.2
    seqs = np.array([[PROTEIN_ALPHABET.index(a) for a in s] for s in seqs])
    weights = scale / (np.sum(squareform(pdist(seqs, metric="hamming")) < theta, axis=0))
    return weights


def parse_fasta(fasta_fpath, return_names=False):
    """ Read in a fasta file and extract just the sequences."""
    seqs = []
    with open(fasta_fpath) as f_in:
        current = ''
        names = [f_in.readline()[1:-1]]
        for line in f_in:
            if line[0] == '>':
                seqs.append(current)
                current = ''
                names.append(line[1:-1])
            else:
                current += line[:-1]
        seqs.append(current)
    if return_names:
        return seqs, names
    else:
        return seqs


def read_fasta(fasta_fpath, out_fpath, header='sequence'):
    """ Read in a fasta file and extract just the sequences."""
    with open(fasta_fpath) as f_in, open(out_fpath, 'w') as f_out:
        f_out.write(header + '\n')
        current = ''
        _ = f_in.readline()
        for line in f_in:
            if line[0] == '>':
                f_out.write(current + '\n')
                current = ''
            else:
                current += line[:-1]
        f_out.write(current + '\n')


class Tokenizer(object):
    """Convert between strings and their one-hot representations."""
    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a:i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i:a for i, a in enumerate(self.alphabet)}

    @property
    def vocab_size(self) -> int:
        return len(self.alphabet)

    @property
    def start_id(self) -> int:
        return self.alphabet.index(START)

    @property
    def stop_id(self) -> int:
        return self.alphabet.index(STOP)

    @property
    def mask_id(self) -> int:
        return self.alphabet.index(MASK)

    @property
    def pad_id(self) -> int:
        return self.alphabet.index(PAD)

    def tokenize(self, seq: str) -> np.ndarray:
        return np.array([self.a_to_t[a] for a in seq])

    def untokenize(self, x: Iterable) -> str:
        return ''.join([self.t_to_a[t] for t in x])


class AAIndexTokenizer(object):
    """Convert between strings and their AAIndex representations."""
    def __init__(self, dpath: str, n_comp: int = 20):
        """
        Args:
            dpath: directory to save raw and reduced representations
            n_comp: number of components in PCA
        """
        alphabet = AAINDEX_ALPHABET

        if not os.path.exists(dpath):
            os.mkdir(dpath)

        if not os.path.exists(dpath + '/aaindex1'):
            file = wget.download('ftp://ftp.genome.jp/pub/db/community/aaindex/aaindex1',
                                 out=dpath + '/' + 'aaindex1')

        if not os.path.exists(dpath + '/raw_aaindex.json'):
            raw_dict = {i: [] for i in alphabet}
            with open(dpath + '/aaindex1', 'r') as f:
                for line in f:
                    if line[0] == 'I':
                        set1 = next(f).strip().split()
                        set2 = next(f).strip().split()
                        set = set1 + set2
                        for i in range(len(alphabet)):
                            val = set[i]
                            if val == 'NA':
                                val = None
                            else:
                                val = float(val)
                            raw_dict[alphabet[i]].append(val)

            with open(dpath + '/raw_aaindex.json', 'w') as f:
                json.dump(raw_dict, f)

        if not os.path.exists(dpath + '/red_aaindex.json'):
            with open(dpath + '/raw_aaindex.json') as f:
                raw_dict = json.load(f)

            # preprocessing : drop embeddings with missing data (drop 13)
            embed_df = pd.DataFrame(raw_dict).dropna(axis=0)
            embed = embed_df.values.T  # (len(alphabet), 553)

            # scale to 0 mean and unit variance
            scaler = StandardScaler()
            embed = scaler.fit_transform(embed)

            # PCA
            pca = PCA(n_components=n_comp, svd_solver='auto')
            embed_red = pca.fit_transform(embed)
            print('VARIANCE EXPLAINED: ', pca.explained_variance_ratio_.sum())
            red_dict = {alphabet[i]: list(embed_red[i, :]) for i in range(len(alphabet))}
            with open(dpath + '/red_aaindex.json', 'w') as f:
                json.dump(red_dict, f)

        # save reduced representation
        with open(dpath + '/red_aaindex.json') as f:
            self.red_dict = json.load(f)

    def tokenize(self, seq: str) -> np.ndarray:
        """

        Args:
            seq: str
                amino acid sequence

        Returns:
            encoded: np.array
                encoded amino acid sequence based on reduced AAIndex representation, (L,n_comp)
        """
        encoded = np.stack([self.red_dict[a] for a in seq], 0)
        return encoded