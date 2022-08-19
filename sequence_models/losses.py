import torch.nn as nn
import torch
import torch.nn.functional as F
from sequence_models.constants import PROTEIN_ALPHABET
from sequence_models.utils import Tokenizer
from torch.nn import KLDivLoss, CrossEntropyLoss
import numpy as np
from sequence_models.collaters import random_sample, sample_transition_matrix


class MaskedCosineLoss(nn.Module):
    """Masked cosine loss between angles."""

    def __init__(self):
        super().__init__()

    def forward(self, pred, tgt, mask):
        mask = mask.bool()
        p = torch.masked_select(pred, mask)
        t = torch.masked_select(tgt, mask)
        diff = p - t
        return torch.cos(diff).mean()


class MaskedMSELoss(nn.MSELoss):
    """Masked mean square error loss.

    Evaluates the MSE at specified locations.

    Shape:
        Inputs:
            - pred: (N, *)
            - tgt: (N, *)
            - mask: (N, *) boolean
    """

    def __init__(self, reduction='mean'):
        super().__init__(reduction=reduction)

    def forward(self, pred, tgt, mask):
        # Make sure mask is boolean
        mask = mask.bool()
        # Select
        p = torch.masked_select(pred, mask)
        t = torch.masked_select(tgt, mask)
        if len(p) == 0:
            return pred.sum() * 0
        return super().forward(p, t)


class SequenceCrossEntropyLoss(nn.Module):
    """Cross-entropy loss for sequences. """

    def __init__(self, weight=None, ignore_index=-100):
        super(SequenceCrossEntropyLoss, self).__init__()
        self.class_weights = weight  # These are class weights
        self.ignore_index = ignore_index

    def forward(self, prediction, tgt, reduction='mean'):
        # Transpose because pytorch expects (N, C, ...) where C is number of classes
        return F.cross_entropy(prediction.transpose(1, 2), tgt, weight=self.class_weights, reduction=reduction,
                               ignore_index=self.ignore_index)


class MaskedCrossEntropyLoss(nn.CrossEntropyLoss):
    """Masked cross-entropy loss for sequences.

    Evaluates the cross-entropy loss at specified locations in a sequence.

    Shape:
        Inputs:
            - pred: (N, L, n_tokens)
            - tgt: (N, L)
            - mask: (N, L) boolean
            - weight: (C, ): class weights for nn.CrossEntropyLoss
    """

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)

    def forward(self, pred, tgt, mask):
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
        # Make sure mask is boolean
        mask = mask.bool()
        # Number of locations to calculate loss
        n = mask.sum()
        # Select
        p = torch.masked_select(pred, mask).view(n, -1)
        t = torch.masked_select(tgt, mask.squeeze())
        return super().forward(p, t)


class VAELoss(nn.Module):
    """A simple VAE loss.

    This is the sum of a reconstruction loss (calculated on predictions and ground truths)
    and the KL divergence between z (mu, log_var) and a standard normal.

    Args:
        class_weights (): The reconstruction loss

    Inputs:
        pre: The predictions (N, *)
        tgt: The ground truths (N, *)
        mu: Predicted means for the latent space
        log_var: Predicted log variance for the latent space
        beta: Ratio between the reconstruction and KLD losses. Optional: default is 1.0.
        sample_weights: Weight to place on each sample. Default None. Size (N, 1 x *)
        reduction (str): 'mean' or 'none'

    Outputs:
        loss (, ): Tensor containing the VAE loss
    """

    def __init__(self, class_weights=None):
        super(VAELoss, self).__init__()
        self.recon_loss = SequenceCrossEntropyLoss(weight=class_weights)

    def forward(self, pre, tgt, mu, log_var, beta=1.0, sample_weights=None, reduction='mean'):
        kld = -0.5 * (1 + log_var - mu ** 2 - log_var.exp())
        r_loss = self.recon_loss(pre, tgt, reduction='none')
        if sample_weights is None:
            kld = kld.sum(dim=1)
            r_loss = r_loss.sum(dim=1)
        else:
            kld = (kld * sample_weights).mean(dim=1)
            r_loss *= sample_weights
            if self.recon_loss.class_weights is not None:
                r_loss = r_loss.sum(dim=1) / self.recon_loss.class_weights[tgt].sum()
            else:
                r_loss = r_loss.mean(dim=1)
        if reduction == 'mean':
            kld = kld.mean()
            r_loss = r_loss.mean()
        return r_loss + beta * kld, r_loss, kld


class MaskedCrossEntropyLossMSA(nn.CrossEntropyLoss):
    """Masked cross-entropy loss for MSAs.
    Evaluates the cross-entropy loss at specified locations in an MSA.
    Shape:
        Inputs:
            - pred: (BS, N, L, n_tokens)
            - tgt: (BS, N, L): label, with uncorrupted tokens
            - mask: (BS, N, L) boolean
    """

    def __init__(self, ignore_index):
        super().__init__(ignore_index=ignore_index, reduction='none')

    def forward(self, pred, tgt, mask, nonpad_mask):
        # Make sure we have that empty last dimension
        if len(mask.shape) == len(pred.shape) - 1:
            mask = mask.unsqueeze(-1)
            nonpad_mask = nonpad_mask.unsqueeze(-1)

        # Make sure mask is boolean
        mask = mask.bool()
        nonpad_mask = nonpad_mask.bool()

        batch_size = pred.shape[0]

        # Create re-weighting array
        num_masked_tokens = mask.sum(axis=(1, 2))  # D-t+1 masked tokens per MSA in each batch
        num_nonpad_tokens = nonpad_mask.sum(axis=(1, 2))

        n = mask.sum()
        p = torch.masked_select(pred, mask).view(n, -1)
        t = torch.masked_select(tgt, mask.squeeze())

        num_masked_tokens_msa = torch.squeeze(num_masked_tokens)
        val_batch = 1 / num_masked_tokens_msa
        rwt = val_batch.repeat_interleave(num_masked_tokens_msa)

        num_nonpad_tokens_msa = torch.squeeze(num_nonpad_tokens)
        d_term = num_nonpad_tokens_msa.repeat_interleave(num_masked_tokens_msa)

        loss = super().forward(p, t)
        rwt = rwt.type(loss.dtype)

        rwt_loss = (d_term * rwt * loss).sum()
        total_loss = loss.sum()

        return rwt_loss, total_loss


def sample_prior_gaussian(q):
    samples = q.shape[0]
    num_seqs = q.shape[1]
    seq_len = q.shape[2]
    sample_shape = (torch.zeros(1, 1, 26)).shape
    m = torch.distributions.uniform.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
    prior = torch.zeros(q[:, :, :, :len(PROTEIN_ALPHABET)].shape)
    print(q.size)
    for i in range(samples):
        for j in range(num_seqs):
            for k in range(seq_len):
                aa_prob = m.sample(sample_shape=sample_shape).squeeze()
                aa_prob = aa_prob / aa_prob.sum()
                # print(aa_prob)
                # print(aa_prob.shape)
                # print(aa_prob.sum())
                prior[i, j, k] = aa_prob
    return prior


class LVBLoss(KLDivLoss):
    def __init__(self, tmax=500, reduction='batchmean', log_target=False, _lambda=0.01):
        self.tmax = tmax
        self._lambda = _lambda
        self.tokenizer = Tokenizer(PROTEIN_ALPHABET)
        super().__init__(reduction=reduction, log_target=log_target)

    def forward(self, q, pred, tgt, timestep, Q):
        T_inf = self.tmax
        p = torch.nn.functional.softmax(pred[:, :, :, :len(PROTEIN_ALPHABET)], dim=2)  # ignoring mask/pad
        alphabet = self.tokenizer.tokenize([self.tokenizer.alphabet])
        losses = []
        prior = sample_prior_gaussian(q)  # random prior, for absorbing state
        prior = prior.to(tgt.device)
        for i in range(tgt.shape[0]):  # enumerate over batch
            # print(self.tokenizer.untokenize(tgt[i]))
            if timestep[i] == 1:
                # CE (L_t=0)
                # Reconstruction loss
                reconstruction_loss = CrossEntropyLoss()
                r_loss = reconstruction_loss(pred[i], tgt[i])
                losses.append(r_loss)
                print("timestep", timestep[i], "r_loss_i", r_loss)
            elif timestep[i] >= T_inf - 1:
                # D KL (L_T)
                # As T approches infinity, this term goes to zero
                # print(prior[i].shape, q[i, :, :26].shape)
                kl_loss_i = super().forward(prior[i].log(),
                                            q[i, :, :, len(PROTEIN_ALPHABET)])  # KLDivLoss expects input in log-space
                losses.append(kl_loss_i)
                # print("timestep", timestep[i], "seq_len", len(tgt[i]), "kINF_loss_i", kl_loss_i)
            else:
                # D KL (L_t-1) -> (q(x|x_t, x_0), p_theta)
                prob = p[i]
                q_true = q[i, :, :, :26]  # ignoring mask/pad
                x_0_bar = torch.zeros((len(prob), len(prob[0])))
                x_0_bar = random_sample(x_0_bar, prob, alphabet)  # sample x_0_bar from prediction prob
                # print(self.tokenizer.untokenize(x_0_bar))
                x_0_bar = torch.tensor(self.tokenizer.one_hot(x_0_bar, tokenized=True))  # one hot
                x_0_bar = x_0_bar.to(tgt.device)
                # Calculate q given model predictions
                x_t, q_x_t = sample_transition_matrix(x_0_bar, Q[timestep[i]], 1, alphabet)
                x_t = torch.tensor(self.tokenizer.one_hot(x_t, tokenized=True))  # one hot
                x_t = x_t.to(tgt.device)
                p_theta = []  # torch.zeros(q_true.shape)
                for j in range(len(x_0_bar)):  # enumerate over masked tokens in sequence (dim 1xK)
                    for k in range(len(x_0_bar[0])):
                        # A = x_t * torch.transpose(Q_t) (shape - 1 x K)
                        A = torch.matmul(x_t[j, k].unsqueeze(0), torch.t(Q[timestep[i]]))
                        # print("A", A.shape, A)
                        # B = x_0_bar * Q_t-1 (shape - 1 x K)
                        B = torch.matmul(x_0_bar[j, k].unsqueeze(0), Q[timestep[i - 1]])
                        # print("B", B.shape, B)
                        q_t_jk = torch.mul(A, B)  # element wise (shape 1 x K)
                        p_theta_jk = q_t_jk * prob[j,k]
                        p_theta_jk = p_theta_jk / p_theta_jk.sum()  # renormalize; sum prob to 1
                        p_theta.append(p_theta_jk.squeeze())
                p_theta = torch.stack(p_theta)
                p_theta = p_theta.to(tgt.device)
                kl_loss_i = super().forward(p_theta.log(), q_true)  # KLDivLoss expects input in log-space
                print("timestep", timestep[i], "seq_len", len(tgt[i]), "k_loss_i", kl_loss_i)
                losses.append(kl_loss_i)
        # TODO append loss to CSV w/ timestep for plotting #
        losses = torch.stack(losses)
        lvb = ((losses.sum()) / (tgt.shape[0]))  # loss per batch, norm by batchsize
        return losses, lvb
