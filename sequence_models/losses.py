import torch.nn as nn
import torch
import torch.nn.functional as F


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
        mask = mask.view(*mask.shape[:2], 1)
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