import torch
import torch.nn as nn

from sequence_models.losses import MaskedCrossEntropyLoss


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def test_masked_cel():
    n = 5
    ell = 7
    t = 11
    scores = torch.randn(n, ell, t).to(device)
    targets = torch.randint(t, (n, ell)).to(device)
    mask = torch.randint(2, (n, ell), device=device).bool()

    weights = None
    mcel = MaskedCrossEntropyLoss(weight=weights, reduction='none')
    loss = mcel(scores, targets, mask)
    assert loss.allclose(mcel(scores, targets, mask.unsqueeze(-1)))
    cel = nn.CrossEntropyLoss(weight=weights, reduction='none')
    full_loss = cel(scores.view(-1, t), targets.view(-1))
    assert loss.allclose(full_loss.masked_select(mask.view(-1)))

    mcel = MaskedCrossEntropyLoss(weight=weights, reduction='mean')
    loss = mcel(scores, targets, mask)
    assert loss.allclose(full_loss.masked_select(mask.view(-1)).mean())

    weights = torch.rand(t, device=device)
    mcel = MaskedCrossEntropyLoss(weight=weights, reduction='none')
    loss = mcel(scores, targets, mask)
    assert loss.allclose(mcel(scores, targets, mask.unsqueeze(-1)))
    cel = nn.CrossEntropyLoss(weight=weights, reduction='none')
    full_loss = cel(scores.view(-1, t), targets.view(-1))
    assert loss.allclose(full_loss.masked_select(mask.view(-1)))

    mcel = MaskedCrossEntropyLoss(weight=weights, reduction='mean')
    loss2 = mcel(scores, targets, mask)
    idx = targets.masked_select(mask).view(-1)
    assert loss2.allclose(full_loss.masked_select(mask.view(-1)).sum() / weights[idx].sum())
