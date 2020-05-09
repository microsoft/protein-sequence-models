from typing import Iterable, List

import torch.nn as nn
import torch


class VAE(nn.Module):
    """A Variational Autoencoder.

    Args:
        encoder (nn.Module): Should produce outputs mu and log_var, both with dimensions (N, d_z)
        decoder (nn.Module): Takes inputs (N, d_z) and attempts to reconstruct the original input

    Inputs:
        x (N, *)

    Ouputs:
        reconstructed (N, *)
        mu (N, d_z)
        log_var (N, d_z)
    """
    def __init__(self, encoder: nn.Module, decoder: nn.Module):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        if self.encoder.d_z != self.decoder.d_z:
            raise ValueError('d_zs do not match!')
        self.d_z = encoder.d_z

    def encode(self, x: torch.tensor):
        return self.encoder(x)

    def decode(self, z: torch.tensor):
        return self.decoder(z)

    def reparameterize(self, mu: torch.tensor, log_var: torch.tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.tensor):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var


class FCStack(nn.Module):
    """A stack of fully-connected layers.

     Every nn.Linear is optionally followed by  a normalization layer,
     a dropout layer, and then a ReLU.

     Args:
         sizes (List of ints): the all layer dimensions from input to output
         norm (str): type of norm. 'bn' for batchnorm, 'ln' for layer norm. Default 'bn'
         p (float): dropout probability

     Input (N, sizes[0])
     Output (N, sizes[-1])
     """

    def __init__(self, sizes: List[int], norm='bn', p=0.0):
        layers = []
        for d0, d1 in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(d0, d1))
            if norm == 'ln':
                layers.append(nn.LayerNorm(d0))
            elif norm == 'bn':
                layers.append(nn.BatchNorm1d(d0))
            if p != 0:
                layers.append(nn.Dropout(p))
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)


class FCEncoder(nn.Module):
    """ A simple fully-connected encoder for sequences.

    Args:
        L (int): Sequence length
        d_in (int): Number of tokens
        d_h (list of ints): the hidden dimensions
        d_z (int): The size of the latent space
        padding_idx (int): Optional: idx for padding to pass to Embedding layer

    Input:
        X (N, L): should be torch.LongTensor

    Outputs:
        mu (N, d_z)
        log_var (N, d_z)
    """

    def __init__(self, L: int, d_in: int, d_h: List[int], d_z: int, padding_idx=None, p=0., norm='bn'):
        super(FCEncoder, self).__init__()
        self.L = L
        self.d_in = d_in
        self.d_z = d_z
        self.embedder = nn.Embedding(d_in, d_h[0], padding_idx=padding_idx)
        sizes = [L * d_h[0]] + d_h[1:]
        self.layers = FCStack(sizes, p=p, norm=norm)
        d1 = sizes[-1]
        self.u_layer = nn.Linear(d1, d_z)  # Calculates the means
        self.s_layer = nn.Linear(d1, d_z)  # Calculates the log sigmas

    def forward(self, X):
        n, _ = X.size()
        e = self.embedder(X).view(n, -1)
        h = self.layers(e)
        return self.u_layer(h), self.s_layer(h)


class FCDecoder(nn.Module):
    """ A simple fully-connected decoder for sequences.

    Args:
        L (int): Sequence length
        d_in (int): Number of tokens
        d_h (list of ints): the hidden dimensions
        d_z (int): The size of the latent space

    Input:
        Z (N, d_z)

    Outputs:
        X (N, L, d_in)
    """

    def __init__(self, L: int, d_in: int, d_h: List[int], d_z: int, p=0., norm='bn'):
        super(FCDecoder, self).__init__()
        self.L = L
        self.d_in = d_in
        self.d_z = d_z
        sizes = [d_z] + d_h + [self.L * self.d_in]
        self.layers = FCStack(sizes, p=p, norm=norm)

    def forward(self, z):
        n, _ = z.shape
        x = self.layers(z)
        return x.view(n, self.L, self.d_in)
