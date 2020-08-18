from typing import Iterable

import numpy as np

from sequence_models.constants import STOP, START, MASK, PAD


def parse_fasta(fasta_fpath):
    """ Read in a fasta file and extract just the sequences."""
    seqs = []
    with open(fasta_fpath) as f_in:
        current = ''
        _ = f_in.readline()
        for line in f_in:
            if line[0] == '>':
                seqs.append(current)
                current = ''
            else:
                current += line[:-1]
        seqs.append(current)
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