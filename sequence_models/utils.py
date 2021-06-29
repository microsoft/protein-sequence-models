from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist

from sequence_models.constants import STOP, START, MASK, PAD
from sequence_models.constants import PROTEIN_ALPHABET


def warmup(n_warmup_steps):
    def get_lr(step):
        return min((step + 1) / n_warmup_steps, 1.0)
    return get_lr


def transformer_lr(n_warmup_steps):
    factor = n_warmup_steps ** 0.5
    def get_lr(step):
        step += 1
        return min(step ** (-0.5), step * n_warmup_steps ** (-1.5)) * factor
    return get_lr


def get_metrics(fname, new=False, tokens=False):
    with open(fname) as f:
        lines = f.readlines()
    valid_lines = []
    train_lines = []
    all_train_lines = []
    for i, line in enumerate(lines):
        if 'Training' in line and 'loss' in line:
            last_train = line
            all_train_lines.append(line)
        if 'Validation complete' in line:
            valid_lines.append(lines[i - 1])
            train_lines.append(last_train)
    metrics = []
    idx_loss = 13
    idx_accu = 16
    idx_step = 6
    if new:
        idx_loss += 2
        idx_accu += 2
        idx_step += 2
    if tokens:
        idx_loss += 2
        idx_accu += 2
        idx_tok = 10
    tok_correction = 0
    last_raw_toks = 0
    for t, v in zip(train_lines, valid_lines):
        step = int(t.split()[idx_step])
        t_loss = float(t.split()[idx_loss])
        t_accu = float(t.split()[idx_accu][:6])
        v_loss = float(v.split()[idx_loss])
        v_accu = float(v.split()[idx_accu][:6])
        if tokens:
            toks = int(t.split()[idx_tok])
            if toks < last_raw_toks:
                tok_correction += last_raw_toks
                doubled = int(all_train_lines[-1].split()[idx_tok]) - int(all_train_lines[-999].split()[idx_tok])
                tok_correction -= doubled
            last_raw_toks = toks
            metrics.append((step, toks + tok_correction, t_loss, t_accu, v_loss, v_accu))

        else:
            metrics.append((step, t_loss, t_accu, v_loss, v_accu))
    if tokens:
        metrics = pd.DataFrame(metrics, columns=['step', 'tokens', 'train_loss',
                                                 'train_accu', 'valid_loss', 'valid_accu'])
    else:
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
        names = [f_in.readline()[1:].replace('\n', '')]
        for line in f_in:
            if line[0] == '>':
                seqs.append(current)
                current = ''
                names.append(line[1:].replace('\n', ''))
            else:
                current += line.replace('\n', '')
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


