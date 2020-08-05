import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


from sequence_models.layers import PositionFeedForward


class MaskedConv1d(nn.Conv1d):
    """ A masked 1-dimensional convolution layer.

    Takes the same arguments as torch.nn.Conv1D, except that the padding is set automatically.

         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        """
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param stride: filter shift
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param bias: adds learnable bias to output
        """
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class MaskedCausalConv1d(nn.Module):
    """Masked Causal 1D convolution based on https://github.com/Popgun-Labs/PopGen/. 
         
         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, out_channels)
    """

    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, groups=1, init=None):
        """
        Causal 1d convolutions with caching mechanism for O(L) generation,
        as described in the ByteNet paper (Kalchbrenner et al, 2016) and "Fast Wavenet" (Paine, 2016)
        Usage:
            At train time, API is same as regular convolution. `conv = CausalConv1d(...)`
            At inference time, set `conv.sequential = True` to enable activation caching, and feed
            sequence through step by step. Recurrent state is managed internally.
        References:
            - Neural Machine Translation in Linear Time: https://arxiv.org/abs/1610.10099
            - Fast Wavenet: https://arxiv.org/abs/1611.09482
        :param in_channels: input channels
        :param out_channels: output channels
        :param kernel_size: the kernel width
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param init: optional initialisation function for nn.Conv1d module (e.g xavier)
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.groups = groups

        # if `true` enables fast generation
        self.sequential = False

        # compute required amount of padding to preserve the length
        self.zeros = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)

        # use supplied initialization function
        if init:
            init(self.conv)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        if input_mask is not None:
            x = x * input_mask
        # training mode
        x = torch.transpose(x, 1, 2)
        if not self.sequential:
            # no padding for kw=1
            if self.kernel_size == 1:
                return self.conv(x).transpose(1, 2)

            # left-pad + conv.
            out = self._pad(x)
            return self._unpad(self.conv(out)).transpose(1, 2)

        # sampling mode
        else:
            # note: x refers to a single timestep (batch, features, 1)
            if not hasattr(self, 'recurrent_state'):
                batch_size = x.size(0)
                self._init_recurrent_state(batch_size)

            return self._generate(x).transpose(1, 2)

    def _pad(self, x):
        return F.pad(x, [self.zeros, 0])

    def _unpad(self, x):
        return x

    def clear_cache(self):
        """
        Delete the recurrent state. Note: this should be called between runs, to prevent
        leftover state bleeding into future samples. Note that we delete state (instead of zeroing) to support
        changes in the inference time batch size.
        """
        if hasattr(self, 'recurrent_state'):
            del self.recurrent_state

    def _init_recurrent_state(self, batch):
        """
        Initialize the recurrent state for fast generation.
        :param batch: the batch size to generate
        """

        # extract weights and biases from nn.Conv1d module
        state = self.conv.state_dict()
        self.weight = state['weight']
        self.bias = state['bias']

        # initialize the recurrent states to zeros
        self.recurrent_state = torch.zeros(batch, self.in_channels, self.zeros, device=self.bias.device)

    def _generate(self, x_i):
        """
        Generate a single output activations, from the input activation
        and the cached recurrent state activations from previous steps.
        :param x_i: features of a single timestep (batch, in_channels, 1)
        :return: the next output value in the series (batch, out_channels, 1)
        """

        # if the kernel_size is greater than 1, use recurrent state.
        if self.kernel_size > 1:
            # extract the recurrent state and concat with input column
            recurrent_activations = self.recurrent_state[:, :, :self.zeros]
            f = torch.cat([recurrent_activations, x_i], 2)

            # update the cache for this layer
            self.recurrent_state = torch.cat(
                [self.recurrent_state[:, :, 1:], x_i], 2)
        else:
            f = x_i

        # perform convolution
        activations = F.conv1d(f, self.weight, self.bias,
                               dilation=self.dilation, groups=self.groups)

        return activations


class HierarchicalCausalConv1d(MaskedCausalConv1d):

    """
         Shape:
            Input: (N, L, in_channels)
            input_mask: (N, L, 1), optional
            Output: (N, L, , out_channels)
    """

    def __init__(self, in_channels, out_channels, ells, kernel_size=1, dilation=1, groups=1, init=None):
        """
        :param d_in: input channels dimension
        :param d_h: hidden dimension
        :param out_channels: output channels dimension
        :param kernel_size: the kernel width
        :param dilation: dilation factor
        :param groups: perform depth-wise convolutions
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        :param ells: if not None, use HierarchialCausalConv1d() over MaskedCausalConv1d() or MaskedConv1d()
        """
        super().__init__(in_channels, out_channels, kernel_size=kernel_size,
                         dilation=dilation, groups=groups, init=init)
        self.ells = ells
        self.inds = [0]
        for ell in self.ells:
            self.inds.append(self.inds[-1] + ell)
        self.unpad_idx = torch.arange(sum(ells))
        for ell in np.cumsum(self.ells[:-1]):
            self.unpad_idx[ell:] += self.zeros

    def _pad(self, x):
        return torch.cat([MaskedCausalConv1d._pad(self, x[:, :, i: j])
                          for i, j in zip(self.inds[:-1], self.inds[1:])], dim=2)

    def _unpad(self, x):
        return x[:, :, self.unpad_idx]


class ByteNetBlock(nn.Module):
    """Residual block from ByteNet paper (https://arxiv.org/abs/1610.10099).
         
         Shape:
            Input: (N, L, d_in)
            input_mask: (N, L, 1), optional
            Output: (N, L, d_out)

    """

    def __init__(self, d_in, d_h, d_out, kernel_size, dilation=1, groups=1, causal=False, ells=None):
        super().__init__()
        if ells is not None:
            self.conv = HierarchicalCausalConv1d(d_h, d_h, ells,
                                                 kernel_size=kernel_size, dilation=dilation, groups=groups)
        elif causal:
            self.conv = MaskedCausalConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        else:
            self.conv = MaskedConv1d(d_h, d_h, kernel_size=kernel_size, dilation=dilation, groups=groups)
        layers1 = [
            nn.LayerNorm(d_in),
            nn.ReLU(),
            PositionFeedForward(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.ReLU()
            ]
        layers2 = [
            nn.LayerNorm(d_h),
            nn.ReLU(),
            PositionFeedForward(d_h, d_out),
            ]
        self.sequence1 = nn.Sequential(*layers1)
        self.sequence2 = nn.Sequential(*layers2)

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length, in_channels)
        :param input_mask: (batch, length, 1)
        :return: (batch, length, out_channels)
        """
        return x + self.sequence2(
            self.conv(self.sequence1(x), input_mask=input_mask)
        )


class ByteNet(nn.Module):

    """Stacked residual blocks from ByteNet paper defined by n_layers
         
         Shape:
            Input: (N, L,)
            input_mask: (N, L, 1), optional
            Output: (N, L,)

    """

    def __init__(self, n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                 ells=None, padding_idx=None, causal=False, dropout=0.0):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :param ells: if not None, use HierarchialCausalConv1d() over MaskedCausalConv1d() or MaskedConv1d()
        :padding_idx: location of padding token in ordered alphabet
        :param causal: if True, chooses MaskedCausalConv1d() over MaskedConv1d()
        """
        super().__init__()
        self.embedder = nn.Embedding(n_tokens, d_embedding, padding_idx=padding_idx)
        self.up_embedder = PositionFeedForward(d_embedding, d_model)
        log2 = int(np.log2(r)) + 1
        dilations = [2 ** (n % log2) for n in range(n_layers)]
        layers = [
            ByteNetBlock(d_model, d_model // 2, d_model, kernel_size, dilation=d, causal=causal, ells=ells)
            for d in dilations
        ]
        self.layers = nn.ModuleList(modules=layers)
        self.dropout = dropout

    def forward(self, x, input_mask=None):
        """
        :param x: (batch, length)
        :param input_mask: (batch, length, 1)
        :return: (batch, length,)
        """
        e = self._embed(x)
        return self._convolve(e, input_mask=input_mask)

    def _embed(self, x):
        e = self.embedder(x)
        e = self.up_embedder(e)
        return e

    def _convolve(self, e, input_mask=None):
        for layer in self.layers:
            e = layer(e, input_mask=input_mask)
            if self.dropout > 0.0:
                e = F.dropout(e, self.dropout)
        return e


class ConditionedByteNetDecoder(ByteNet):
    """ A conditioned, (hierarchical) ByteNet decoder.
    Inputs:
        x (n, ell)
        c: (n, n_blocks, d_conditioning)
        ells: lengths of blocks

    """

    def __init__(self, n_tokens, d_embedding, d_conditioning, d_model, n_layers, kernel_size, r, ells):
        """
        :param n_tokens: number of tokens in token dictionary
        :param d_embedding: dimension of embedding
        :param d_conditioning: dimension for conditioning, subtract from d_model
        :param d_model: dimension to use within ByteNet model, //2 every layer
        :param n_layers: number of layers of ByteNet block
        :param kernel_size: the kernel width
        :param r: used to calculate dilation factor
        :param ells: if not None, use HierarchialCausalConv1d() over MaskedCausalConv1d() or MaskedConv1d()
        """
        super().__init__(n_tokens, d_embedding, d_model, n_layers, kernel_size, r,
                         ells=ells, padding_idx=None, causal=True)
        self.up_embedder = PositionFeedForward(d_embedding, d_model - d_conditioning)
        self.ells = nn.Parameter(torch.tensor(ells), requires_grad=False)

    def _embed(self, inputs):
        x, c = inputs
        e = self.embedder(x)
        e = self.up_embedder(e)  # (n, ell, d_model - d_conditioning)
        # Concatenate the conditioning
        c_ = torch.repeat_interleave(c, self.ells, dim=1)  # (n, ell, d_conditioning)
        e = torch.cat([e, c_], dim=2)  # (n, ell, d_model)
        # Mask out the unwanted connections
        return e


