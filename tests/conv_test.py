import torch
from sequence_models.convolutional import ByteNetBlock, ByteNet, ConditionedByteNetDecoder, Conductor


device = torch.device('cuda')
b = 5
dz = 8
d_out = 4
n_f = [1024, 256, 512, 64]
layer = Conductor(dz, n_f, d_out).to(device)
z = torch.randn(b, dz).to(device)
out = layer(z)
assert out.shape == (b, d_out, 2 ** len(n_f))
z = torch.randn(b, dz, 1).to(device)
out = layer(z)
assert out.shape == (b, d_out, 2 ** len(n_f))

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