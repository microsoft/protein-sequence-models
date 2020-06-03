import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch

from sequence_models.losses import VAELoss, SequenceCrossEntropyLoss
from sequence_models.vae import FCDecoder, FCEncoder, VAE, Conductor

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

N = 2
L = 5
D = 3

src = [
    [0, 1, 2, 0, 1],
    [0, 1, 1, 1, 0]
]
src = torch.LongTensor(src)

n_hidden = np.random.choice(np.arange(3, 10))
d_h = list(np.random.choice(np.arange(1, 10), size=n_hidden))
d_z = np.random.choice(np.arange(2, 10))
encoder = FCEncoder(L, D, d_h, d_z)
decoder = FCDecoder(L, D, d_h[::-1], d_z)


def test_encoder():
    assert encoder.embedder.num_embeddings == D
    assert encoder.embedder.embedding_dim == d_h[0]
    mu, logvar = encoder(src)
    nm, dm = mu.size()
    nv, dv = logvar.size()
    assert nm == nv
    assert nm == N
    assert dm == d_z
    assert dv == d_z


def test_decoder():
    z = torch.Tensor(np.random.random((N, d_z)))
    p = decoder(z)
    n, ell, dp = p.size()
    assert n == N
    assert ell == L
    assert dp == D


def test_vae():
    vae = VAE(encoder, decoder)
    p, mu, logvar = vae(src)
    mu2, logvar2 = encoder(src)
    assert torch.allclose(mu2, mu)
    assert torch.allclose(logvar, logvar2)
    # Check shape of p
    n, ell, d = p.size()
    assert n == N
    assert ell == L
    assert d == D
    # Test encoder
    m1, s1 = vae.encode(src)
    m2, s2 = encoder(src)
    assert torch.allclose(m1, m2)
    assert torch.allclose(s1, s2)
    # Test decode
    z = torch.Tensor(np.random.random((N, d_z)))
    p1 = decoder(z)
    p2 = vae.decode(z)
    assert torch.allclose(p1, p2)


def test_loss():
    r_loss_func = SequenceCrossEntropyLoss()
    vae = VAE(encoder, decoder)
    p, mu, logvar = vae(src)
    r_loss = r_loss_func(p, src, reduction='none')
    kl_loss = -0.5 * (1 + logvar - mu ** 2 - logvar.exp())
    beta = torch.rand(1)

    # Without classweights or sample_weights
    vloss = VAELoss(class_weights=None)
    # With reduction
    loss, r, k = vloss(p, src, mu, logvar, beta=beta)
    assert torch.allclose(r_loss.sum(dim=1).mean(dim=0) + beta * kl_loss.sum(dim=1).mean(dim=0), loss)
    assert torch.allclose(r, r_loss.sum(dim=1).mean())
    assert torch.allclose(k, kl_loss.sum(dim=1).mean())
    # Without reduction
    loss, r, k = vloss(p, src, mu, logvar, beta=beta, reduction='none')
    assert torch.allclose(r_loss.sum(dim=1) + beta * kl_loss.sum(dim=1), loss)
    assert torch.allclose(r, r_loss.sum(dim=1))
    assert torch.allclose(k, kl_loss.sum(dim=1))

    # With class_weights and sample_weights
    cw = torch.rand(3)
    sw = torch.rand((N, 1))
    r_loss_func = SequenceCrossEntropyLoss(weight=cw)
    r_loss = r_loss_func(p, src, reduction='none')
    r_loss *= sw
    r_loss = r_loss.sum(dim=1) / r_loss_func.class_weights[src].sum()
    kl_loss *= sw
    vloss = VAELoss(class_weights=cw)
    # With reduction
    loss, r, k = vloss(p, src, mu, logvar, beta=beta, sample_weights=sw)
    assert torch.allclose(r_loss.mean() + beta * kl_loss.sum(dim=1).mean(dim=0), loss)
    assert torch.allclose(r, r_loss.mean())
    assert torch.allclose(k, kl_loss.sum(dim=1).mean())
    # Without reduction
    loss, r, k = vloss(p, src, mu, logvar, beta=beta, sample_weights=sw, reduction='none')
    assert torch.allclose(r_loss + beta * kl_loss.sum(dim=1), loss)
    assert torch.allclose(r, r_loss)
    assert torch.allclose(k, kl_loss.sum(dim=1))


def test_conductor():
    b = 5
    dz = 8
    d_out = 4
    n_f = [np.random.randint(64, 128) for _ in range(np.random.randint(3, 8))]
    layer = Conductor(dz, n_f, d_out).to(device)
    z = torch.randn(b, dz).to(device)
    out = layer(z)
    assert out.shape == (b, 2 ** (len(n_f) + 2), d_out)
    z = torch.randn(b, dz, 1).to(device)
    out = layer(z)
    assert out.shape == (b, 2 ** (len(n_f) + 2), d_out,)