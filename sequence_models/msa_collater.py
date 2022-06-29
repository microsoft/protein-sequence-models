from typing import List

import numpy as np
import torch

from sequence_models.utils import Tokenizer
from sequence_models.constants import PAD


def _pad(tokenized: List, num_seq: int, max_len: int, value: int) -> torch.Tensor:
    """Utility function that pads batches to the same length."""
    batch_size = len(tokenized)
    output = torch.zeros((batch_size, num_seq, max_len), dtype=torch.long) + value
    for i in range(batch_size):
        tokenized[i] = torch.LongTensor(np.array(tokenized[i]))
        output[i, :, :len(tokenized[i][0])] = tokenized[i]
    return output


class MSAAbsorbingCollater():
    """Collater for MSA Absorbing Diffusion model.

    Parameters:
        alphabet (str)

    Input (list): a batch of Multiple Sequence Alignments (MSAs), each MSA contains 64 sequences
    Output:
        src (torch.LongTensor): corrupted input + padding
        tgt (torch.LongTensor): input + padding
    """

    def __init__(self, alphabet: str, pad_token=PAD, num_seq=64):
        self.tokenizer = Tokenizer(alphabet)
        self.pad_idx = self.tokenizer.alphabet.index(pad_token)
        self.num_seq = num_seq

    def __call__(self, batch_msa):
        tgt = list(batch_msa[:])
        src = tgt.copy()

        longest_msa = 0
        batch_size = len(batch_msa)

        for i in range(batch_size):
            # Tokenize MSA
            tgt[i] = [self.tokenizer.tokenize(s) for s in tgt[i]]
            src[i] = [self.tokenizer.tokenize(s) for s in src[i]]

            curr_msa = src[i]

            curr_msa = np.asarray(curr_msa)
            length, depth = curr_msa.shape  # length = number of seqs in MSA, depth = # AA in MSA
            t = np.random.choice(length * depth)  # Pick timestep t
            t += 1  # ensure t cannot be 0

            # Flatten MSA to 1D to mask tokens
            curr_msa = curr_msa.flatten()
            d = len(curr_msa)
            num_masked_tokens = d - t + 1
            mask_idx = np.random.choice(d, num_masked_tokens, replace=False)  # Pick D-t+1 random indices to mask
            curr_msa[mask_idx] = self.tokenizer.mask_id
            curr_msa = curr_msa.reshape(length, depth)

            src[i] = list(curr_msa)

            if depth > longest_msa:  # Keep track of the longest MSA, pad other MSAs in this batch to that length
                longest_msa = depth

        # Pad sequences
        src = _pad(src, self.num_seq, longest_msa, self.pad_idx)
        tgt = _pad(tgt, self.num_seq, longest_msa, self.pad_idx)

        mask = (src == self.tokenizer.mask_id)

        return src, tgt, mask
