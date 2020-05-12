import torch

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


class UngappedAccuracy(MaskedAccuracy):

    def __init__(self, gap_index):
        self.gap_index = gap_index

    def __call__(self, pred, tgt):
        mask = tgt != self.gap_index
        return super().__call__(pred, tgt, mask)