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
    def __init__(self, k=5, contact_range='short'):
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
        # contact if d < 8 angstroms, or d > 1/8**2
        self.contact_threshold = 1 / (8.0 ** 2)
        self.k = k

    def __call__(self, prediction, tgt, mask, L):
        """
        Args:
            prediction: torch.tensor (N, L, L)

            tgt: torch.tensor (N, L, L)

            mask: torch.tensor (N, L, L)

            L: torch.tensor (L,)
                lengths of protein sequences
        """

        # reshape if prediction is shape n, el, el, 1 -> n, el, el
        if len(prediction.size()) == 4:
            prediction = prediction.squeeze()
            tgt = tgt.squeeze()

        n, el, _ = tgt.shape

        # find residue pairs that fit res range criteria
        pairs = self._get_contact_pairs(mask, self.res_range)

        # how many contacts to looks at
        top_k = L // self.k

        # 1 is tp, 0 is fp, -1 is everything else
        contacts = torch.ones_like(tgt) * -1.
        contacts[pairs] += (prediction[pairs] >= self.contact_threshold) * 2.
        contacts[pairs] *= (tgt[pairs] >= self.contact_threshold) * 1.

        # get top contacts
        vals, idx = contacts.view(n, -1).sort(descending=True)

        # calculate precision
        precision = [self._precision(vals[i, :top_k[i]]) for i in range(len(top_k))]
        return precision

    def _get_contact_pairs(self, mask, res_range):
        n, el, _ = mask.shape
        # get distance based on primary structure
        pri_dist = torch.abs(torch.arange(el)[None, :].repeat(el, 1) - torch.arange(el).view(-1, 1)).float()
        # repeat for each sample in batch size
        pri_dist = pri_dist.view(1, el, el).repeat(n, 1, 1)
        # apply mask to hide residue pairs to not include
        pri_dist_masked = pri_dist * mask
        # get pairs according to res range interval
        pairs = ((pri_dist_masked >= res_range[0]) & (pri_dist_masked < res_range[1])) * 1.
        # take only upper triangle to prevent duplicates
        pairs = torch.triu(pairs)
        # get idx of pairs
        pairs = torch.nonzero(pairs, as_tuple=True)
        return pairs

    def _precision(self, array):
        try:
            tp = torch.sum(array == 1., dim=0).item()
            fp = torch.sum(array == 0., dim=0).item()
            return tp / (tp + fp)
        except ZeroDivisionError:
            print('NO CONTACTS WERE PREDICTED')
            return None