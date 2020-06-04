import numpy as np
import torch
from sequence_models.convolutional import ByteNetBlock, ByteNet, ConditionedByteNetDecoder, \
    HierarchicalCausalConv1d, MaskedCausalConv1d

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
b = 5

d = 8
ell = 7
k = 3
dil = 2
block = ByteNetBlock(2 * d, d, 2 * d, k, dilation=dil, causal=False).to(device)
x = torch.randn(b, ell, 2 * d).to(device)
mask = torch.ones(b, ell, 1, device=device)
mask[:, 4:] = 0.0
out = block(x, input_mask=mask)
assert out.shape == (b, ell, 2 * d)

n_tokens = 9
d_e = 2
n_layers = 4
r = 4
ells = [2, 2, 3]
d_c = 3
x = torch.randint(0, n_tokens, (b, ell), device=device)
c = torch.randn(b, 3, d_c, device=device)
net = ConditionedByteNetDecoder(n_tokens, d_e, d_c, d, n_layers, 3, r, ells).to(device)
out = net((x, c))
assert out.shape == (b, sum(ells), d)

net = ByteNet(n_tokens, d_e, d, n_layers, 3, r).to(device)
out = net(x)
assert out.shape == (b, ell, d)

# Test Causal convolution
din = 8
dout = 9
k = 3
n = 5
ells = [3, 3, 4, 4, 10, 5, 10]
dil = 4
x = torch.randn(n, sum(ells), din).to(device)
layer = MaskedCausalConv1d(din, dout, k, dilation=dil).to(device)
x.requires_grad = True
out = layer(x)
pos = 20
loss = out[0, pos, :].sum()
grad = torch.autograd.grad(loss, x, retain_graph=True)[0][0].sum(dim=1)
start = pos - (k - 1) * dil
for p in range(start, sum(ells)):
    if p > pos:
        assert grad[p].sum() == 0
    elif (p - start) % dil == 0:
        assert grad[p].sum() != 0

# Test gradients in hierarchical causal convolution
dil = 4
layer = HierarchicalCausalConv1d(din, dout, ells, k, dilation=dil).to(device)
x = torch.randn(n, sum(ells), din).to(device)
x.requires_grad = True
out = layer(x)
start = pos - (k - 1) * dil
loss = out[0, pos, :].sum()
grad = torch.autograd.grad(loss, x, retain_graph=True)[0][0].sum(dim=1)
blocks = np.zeros(sum(ells))
for ell in np.cumsum(ells):
    blocks[ell:] += 1
for p in range(start, sum(ells)):
    if p > pos:
        assert grad[p].sum() == 0
    elif (p - start) % dil == 0:
        if blocks[p] == blocks[pos]:
            assert grad[p].sum() != 0