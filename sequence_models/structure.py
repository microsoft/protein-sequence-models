import torch.nn as nn
import torch.nn.functional as F

from sequence_models.convolutional import MaskedConv2d, ByteNet2d, ConditionedByteNetDecoder
from sequence_models.layers import PositionFeedForward


class Attention2d(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.layer = MaskedConv2d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        n, ell, _, _ = x.shape
        attn = self.layer(x)
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)
        out = (attn * x.view(n, ell * ell, -1)).mean(dim=1)
        return out


# class Attention1d(nn.Module):
#
#     def __init__(self, L, h_dim):
#         super().__init__()
#         self.linear = nn.Linear(L*h_dim, L*h_dim)
#
#     def forward(self, x, input_mask=None):
#         """
#         x : torch.Tensor, (N, L, h_dim)
#             input tensor
#
#         input_mask : torch.Tensor, (N, L)
#             to mask specific residues or start/stop token
#
#         """
#         n, el, h_dim = x.shape
#         x = x.view(-1)
#         attn = self.linear(x)
#         if input_mask is not None:
#             attn = attn.masked_fill_(~input_mask.repeat(n,h_dim,1).T.reshape(-1,).bool(),
#                                      float('-inf'))
#         attn = F.softmax(attn, dim=-1)
#         out = (attn * x)
#         return out


class Attention1d(nn.Module):

    def __init__(self, in_dim):
        super().__init__()
        self.layer = MaskedConv2d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        """
        x : torch.Tensor, (N, L, h_dim)
            input tensor

        input_mask : torch.Tensor, (N, L)
            to mask specific residues or start/stop token

        """
        n, ell, _, = x.shape
        x = x.view(n, ell, 1, -1)
        attn = self.layer(x)
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)
        out = (attn * x.view(n, ell, -1)).mean(dim=1)
        return out


class StructureConditioner(nn.Module):

    def __init__(self, d_in, d_model, n_layers, kernel_size, r, dropout=0.0):
        super().__init__()
        self.embedder = ByteNet2d(d_in, d_model, n_layers, kernel_size, r, dropout=dropout)
        self.attention = Attention2d(d_model)

    def forward(self, x, input_mask=None):
        return self.attention(self.embedder(x, input_mask=input_mask), input_mask=input_mask)


class StructureConditionedBytenet(nn.Module):

    def __init__(self, n_tokens, d_embedding, d_conditioning, d_model, n_layers, k_b, r_b,
                 d_structure, n_c_layers, k_c, r_c):
        super().__init__()
        self.conditioner = StructureConditioner(d_structure, d_conditioning, n_c_layers, k_c, r_c)
        self.bytenet = ConditionedByteNetDecoder(n_tokens, d_embedding, d_conditioning, d_model, n_layers, k_b, r_b)
        self.decoder = PositionFeedForward(d_model, n_tokens)

    def forward(self, src, struc, src_mask, str_mask):
        c = self.conditioner(struc, input_mask=str_mask)
        out = self.bytenet((src, c), input_mask=src_mask)
        out = self.decoder(out)
        return out