import torch
import numpy as np

class SequenceAccuracy(object):
    """Computes accuracy between two sequences.

    Inputs:
        pred (N, L, C)
        tgt (N, L)
        ignore_index (int): index of token to mask out
    """
    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def __call__(self, pred, tgt):
        n = tgt.shape[0]
        pred = pred.argmax(dim=-1).view(n,-1)
        tgt = tgt.view(n, -1)
        mask = tgt != self.ignore_index
        tgt = tgt[mask]
        pred = pred[mask]
        return (pred == tgt).float().mean()


class MaskedAccuracy(object):
    """Masked accuracy.

    Inputs:
        pred (N, L, C)
        tgt (N, L)
        mask (N, L)
    """

    def __call__(self, pred, tgt, mask):
        _, p = torch.max(pred, -1)
        masked_tgt = torch.masked_select(tgt, mask.bool())
        p = torch.masked_select(p, mask.bool())
        return torch.mean((p == masked_tgt).float())


class MaskedTopkAccuracy(object):
    """Masked top k accuracy.

    Inputs:
        pred (N, L, C)
        tgt (N, L)
        mask (N, L)
        k (int)
    """

    def __call__(self, pred, tgt, mask, k):
        _, p = torch.topk(pred, k, -1)
        masked_tgt = torch.masked_select(tgt, mask.bool())
        p = torch.masked_select(p, mask.bool().unsqueeze(-1)).view(-1, k)
        masked_tgt = masked_tgt.repeat(k).view(k, -1).t()
        return (p == masked_tgt).float().sum(dim=1).mean()


class UngappedAccuracy(MaskedAccuracy):

    def __init__(self, gap_index):
        self.gap_index = gap_index

    def __call__(self, pred, tgt):
        mask = tgt != self.gap_index
        return super().__call__(pred, tgt, mask)


class LPrecision(object):
    """
    Calculates top L // k precision where L is length
    * params acquired from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4894841/#FN1

    """
    def __init__(self, k=5, contact_range='medium-long'):
        """
        Args:
            k: L // k number of contacts to check
            contact_range: short, medium or long contacts
        """
        if contact_range == 'short':
            self.res_range = [6, 12]
        elif contact_range == 'medium':
            self.res_range = [12, 24]
        elif contact_range == 'long':
            self.res_range = [24, np.inf]
        elif contact_range == 'medium-long':
            self.res_range = [12, np.inf]
        else:
            raise ValueError("contact_range must be one of 'short', 'medium', 'long', or 'medium-long'.")
        # contact if d < 8 angstroms, or d > exp(-8 ** 2 / 8 ** 2)
        self.contact_threshold = np.exp(-1)
        self.k = k

    def __call__(self, prediction, tgt, mask, ells):
        """
        Args:
            prediction: torch.tensor (N, L, L)
            tgt: torch.tensor (N, L, L)
            mask: torch.tensor (N, L, L)
            ells: torch.tensor (N,)
                lengths of protein sequences
        """

        n, el, _ = tgt.shape

        # update the mask
        # get distance based on primary structure
        pri_dist = torch.abs(torch.arange(el)[None, :].repeat(el, 1) - torch.arange(el).view(-1, 1)).float()
        # repeat for each sample in batch size
        pri_dist = pri_dist.view(1, el, el).repeat(n, 1, 1)
        dist_mask = (pri_dist > self.res_range[0]) & (pri_dist < self.res_range[1])
        mask = dist_mask & mask

        # pull the top_k most likely contacts from each prediction
        prediction = prediction.masked_fill(~mask, -1)
        tgt = tgt.masked_fill(~mask, -1)
        # Get just the upper triangular
        idx = torch.triu_indices(el, el, offset=1)
        prediction = torch.stack([p[idx[0], idx[1]] for p in prediction])  # N x n_triu
        tgt = torch.stack([t[idx[0], idx[1]] for t in tgt])  # N x n_triu
        tgt = tgt > self.contact_threshold
        idx = torch.argsort(prediction, dim=1, descending=True)  # N x tri_u

        # see how many are tp or fp
        # how many contacts to look at
        top_k = ells // self.k
        n_valid = mask.sum(dim=-1).sum(dim=1)
        n_valid = np.minimum(n_valid, top_k).long()  # (N, )
        n_predicted = n_valid.sum().item()
        if n_predicted == 0:
            return 0, 0
        # n_predicted = (prediction > self.contact_threshold).sum(dim=1)
        # n_valid = np.minimum(n_valid, n_predicted).long()
        n_contacts = 0
        for ids, t, n in zip(idx, tgt, n_valid):
            n_contacts += t[ids[:n]].sum().item()
        precision = n_contacts / n_predicted
        return precision, n_predicted

