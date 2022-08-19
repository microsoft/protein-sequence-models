from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform, pdist
from sklearn.preprocessing import normalize
import torch

from sequence_models.constants import STOP, START, MASK, PAD, ALL_AAS_BLOSUM, PROTEIN_ALPHABET_BLOSUM
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


def matrixMul(a, n):
    if (n <= 1):
        return a
    else:
        return torch.matmul(matrixMul(a, n - 1), a)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def double_stochastic(q):
    q_norm = normalize(q, axis=1, norm='l1')
    while not np.isclose(np.min(np.sum(q_norm, axis=0)),
                         1):  # only checking that one value converges to 1 (prob best to do all 4 min/max)
        q_norm = normalize(q_norm, axis=0, norm='l1')
        q_norm = normalize(q_norm, axis=1, norm='l1')
    return q_norm


def _beta_schedule(num_timesteps, schedule='linear', start=1e-5, end=0.999, max=8):
    """
    Variance schedule for adding noise as introduced by Nichol and Dhariwal and adapted by Hoogeboom et al
    Coined as uniform schedule in Austin et al.
    Start/End will control the magnitude of sigmoidal and cosine schedules..
    #TODO: Check that cosine matches Austin cosine schedule - I think theirs is slightly diff
    #TODO: add mutual information Beta_t introduced by Sohl Dickensen used by Austin
    """
    if schedule == 'linear':
        betas = torch.linspace(start, end, num_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, num_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-10, 10, num_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    elif schedule == "cosine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.cos(betas) * (end - start) + start
    elif schedule == "sine":
        betas = torch.linspace(np.pi / 2, 0, num_timesteps)
        betas = torch.sin(betas) * (end - start) + start
    elif schedule == "exp":
        betas = torch.linspace(0, max, num_timesteps)
        betas = torch.exp(betas) * (end - start) + start
    else:
        print("Must select a valid schedule; ['linear', 'quad', 'sigmoid', 'cosine']")
    return betas


class Tokenizer(object):
    """Convert between strings and their one-hot representations."""

    def __init__(self, alphabet: str):
        self.alphabet = alphabet
        self.a_to_t = {a: i for i, a in enumerate(self.alphabet)}
        self.t_to_a = {i: a for i, a in enumerate(self.alphabet)}

        # self.matrix = loadMatrix(path_to_blosum)
        # self.matrix_dict = dict(self.matrix)
        self.all_aas = list(ALL_AAS_BLOSUM)
        self.protein_alphabet = list(PROTEIN_ALPHABET_BLOSUM)

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

    def one_hot(self, msa, tokenized=False):
        """one hot encode according to indexing"""
        tokens = self.all_aas
        x_onehot = np.zeros((len(msa), len(msa[0]), len(tokens)))
        for i, seq in enumerate(msa):
            for j, a in enumerate(seq):
                if not tokenized:
                    one_index = self.a_to_t[a]
                else:
                    one_index = a
                # print(one_index, len(tokens))
                if one_index < len(tokens):  # everything that isnt an amino acid will be zero
                    # print(i, one_index)
                    x_onehot[i][j][int(one_index)] = 1
        return x_onehot

    # def q_blosum(self):
    #     q = np.array([i for i in self.matrix_dict.values()])
    #     q = q.reshape((len(self.all_aas), len(self.all_aas)))
    #     q = softmax(q)
    #     q = double_stochastic(q)
    #     return q
    #
    # def q_blosum_schedule(self, timesteps=500, end=0.4, max=8):
    #     q = torch.tensor(self.q_blosum())
    #     betas = _beta_schedule(timesteps, 'exp', end=end, max=max)
    #     alphas = betas - end # normalize first value to 0
    #     q_diag = torch.tensor(np.identity(len(self.all_aas))) * q
    #     q_non_diag = torch.tensor((1 - np.identity(len(self.all_aas)))) * q
    #     #print(q_diag, q_non_diag)
    #     q_t = []
    #     for i, a in enumerate(alphas):
    #         R = q_diag + q_non_diag * a
    #         q_temp = double_stochastic(R)
    #         q_t.append(torch.tensor(q_temp))
    #     q_t = torch.stack(q_t)
    #     return q_t

    def q_random_schedule(self, timesteps=500, end=2, max=6):
        betas = _beta_schedule(timesteps, 'exp', end=end, max=max)
        alphas = (betas - betas.min()) / (betas.max() * 0.8)  # normalize first value to 0 and max > 1
        q_diag = torch.tensor(np.identity(len(Tokenizer(PROTEIN_ALPHABET).all_aas)))
        q_non_diag = torch.tensor((1 - np.identity(len(Tokenizer(PROTEIN_ALPHABET).all_aas))))
        q_t = []
        for i, a in enumerate(alphas):
            R = q_diag + q_non_diag * a
            q_temp = double_stochastic(R)
            q_t.append(torch.tensor(q_temp))
        q_t = torch.stack(q_t)
        return q_t
