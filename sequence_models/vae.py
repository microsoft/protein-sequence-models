from typing import List

import torch.nn as nn
import torch
import torch.optim as optim
from apex import amp
import mlflow

from sequence_models.losses import VAELoss
from sequence_models.layers import FCStack
from sequence_models.metrics import UngappedAccuracy


class VAETrainer(object):
    """ Trainer for VAEs."""
    def __init__(self, vae, device, pad_idx, class_weights=None, lr=1e-4, beta=1.0, opt_level='O2', optim_kwargs={},
                 early_stopping=True, patience=10, improve_threshold=0.001, save_freq=100, scheduler=None,
                 scheduler_args=[], scheduler_kwargs={}, scheduler_time='epoch', kl_anneal=-1):
        self.vae = vae.to(device)
        self.device = device
        self.beta = beta
        self.anneal_epochs = kl_anneal
        # Store an optimizer
        self.optimizer = optim.Adam(vae.parameters(), lr=lr, **optim_kwargs)
        if opt_level != 'O0':
            self.vae, self.optimizer = amp.initialize(self.vae, self.optimizer, opt_level=opt_level)
        self.opt_level = opt_level
        # Store the loss
        self.loss_func = VAELoss(class_weights=class_weights)
        self.accu_func = UngappedAccuracy(pad_idx)
        if scheduler is None:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 1, gamma=1.0)
            self.scheduler_time = 'epoch'
        else:
            self.scheduler = scheduler(self.optimizer, *scheduler_args, **scheduler_kwargs)
            self.scheduler_time = scheduler_time
        self.early_stopping = early_stopping
        self.patience = patience
        self.improve_threshold = improve_threshold
        self.save_freq = save_freq
        self.current_epoch = 0

    def step(self, src, train=True, weights=None):
        """Do a forward pass. Do a backward pass if train=True. """
        if train:
            self.vae = self.vae.train()
            self.optimizer.zero_grad()
        else:
            self.vae = self.vae.eval()
        loss, r_loss, kl_loss, accu = self._forward(src, weights=weights)
        if train:
            self._backward(loss)
        return loss.item(), r_loss.item(), kl_loss.item(), accu.item()

    def _forward(self, src, weights=None):
        src = src.to(self.device)
        p, z_mu, z_log_var = self.vae(src)
        if self.anneal_epochs == -1:
            beta = self.beta
        else:
            beta = self.beta * min(self.current_epoch / self.anneal_epochs, 1.0)
        loss, r_loss, kl_loss = self.loss_func(p, src, z_mu, z_log_var, beta=beta, sample_weights=weights)
        accu = self.accu_func(p, src)
        return loss, r_loss, kl_loss, accu

    def _backward(self, loss):
        if self.opt_level != 'O0':
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.optimizer.step()
        if self.scheduler_time == 'batch':
            self.scheduler.step()

    def epoch(self, loader, train):
        losses = 0.0
        r_losses = 0.0
        kl_losses = 0.0
        accus = 0.0
        for i, batch in enumerate(loader):
            src = batch[0]
            if len(batch) == 2:
                weights = batch[1]
            else:
                weights = None
            loss, r_loss, kl_loss, accu = self.step(src, train=train, weights=weights)
            losses += loss
            r_losses += r_loss
            kl_losses += kl_loss
            accus += accu
            mean_loss = losses / (i + 1)
            mean_r = r_losses / (i + 1)
            mean_kl = kl_losses / (i + 1)
            mean_accu = accus / (i + 1)
            if train:
                print('\rTraining ', end='')
            else:
                print('\rValidating ', end='')
            print(
                'Epoch %d of %d Batch %d of %d loss = %.4f r = %.4f kld = %.4f accu = %.4f'
                % (
                    self.current_epoch + 1,
                    self.total_epochs,
                    i + 1,
                    len(loader),
                    mean_loss,
                    mean_r,
                    mean_kl,
                    mean_accu
                ),
                  end=''
            )
        print()
        return mean_loss, mean_r, mean_kl, mean_accu

    def train(self, train_loader, epochs, valid_loader=None, save_path=None):
        done = False
        stagnant = 0
        best_loss = 1e8
        self.total_epochs = epochs
        for epoch in range(epochs):
            self.current_epoch = epoch
            if epoch > 0 and (epoch % self.save_freq == 0) and save_path is not None:
                torch.save(self.vae.state_dict(), save_path + 'autosave_epoch_{}.pkl'.format(epoch))
                torch.save(self.optimizer.state_dict(), save_path + 'optim_autosave_epoch_{}.pkl'.format(epoch))
            if not done:
                loss, r_loss, kld, accu = self.epoch(train_loader, True)
                mlflow.log_metrics(
                    {
                        'train_loss': loss,
                        'train_r_loss': r_loss,
                        'train_kld': kld,
                        'train_accu': accu
                    },
                    step=self.current_epoch
                )

                if valid_loader is not None:
                    with torch.no_grad():
                        loss, r_loss, kld, accu = self.epoch(valid_loader, False)
                    if self.scheduler_time == 'epoch':
                        self.scheduler.step(loss)
                    mlflow.log_metrics(
                        {
                            'valid_loss': loss,
                            'valid_r_loss': r_loss,
                            'valid_kld': kld,
                            'valid_accu': accu
                        },
                        step=self.current_epoch
                    )
                    if self.early_stopping and self.current_epoch > self.anneal_epochs:
                        improve = loss <= (1 - self.improve_threshold) * best_loss
                        if not improve:
                            stagnant += 1
                        else:
                            stagnant = 0
                            best_loss = loss
                        done = stagnant >= self.patience
            else:
                print('Stopping early at epoch {}'.format(self.current_epoch))
                break
        return self.vae, self.loss_func, self.optimizer


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


class HierarchicalRecurrentDecoder(nn.Module):
    """ A hierarchical recurrent decoder.

    Args:
        ells (list of ints): subsequence lengths
        d_in (int): Number of tokens
        d_z (int): The size of the latent space
        conductor (nn.Module): outputs conditioning for each subsequence
        decoder (nn.Module): recurrent decoder

    Input:
        z (N, d_z):

    Outputs:
        X (N, L, d_in)
    """
    def __init__(self, ells, d_in, d_z, conductor, decoder):
        self.conductor = conductor
        self.decoder = decoder
        self.ells = ells
        self.d_in = d_in
        self.d_z = d_z

    def forward(self, z, x):
        c = self.conductor(z)
        return self.decoder(x, c, self.ells)
