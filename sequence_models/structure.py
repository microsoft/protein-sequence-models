import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from sequence_models.convolutional import MaskedConv2d, ByteNet2d, ConditionedByteNetDecoder, MaskedConv1d, ByteNet
from sequence_models.layers import PositionFeedForward


class ByteNetStructureModel(nn.Module):
    """Takes a Bytenet embedding and converts it to a 2D structural output.

    Inputs:
        x (n, ell)
        input_mask (n, ell), optional

    Outputs:
        structure (n, ell, ell, d_out)
    """

    def __init__(self, bytenet, d_model, d_out):
        super().__init__()
        self.embedder = bytenet
        self.d_model = d_model
        self.p = MaskedConv1d(d_model, 256, 1)
        self.q = MaskedConv1d(d_model, 256, 1)
        self.relu = nn.ReLU()
        self.linear = MaskedConv2d(16, 1, 1)

    def forward(self, x, input_mask=None):
        e = self.embedder(x, input_mask=input_mask)
        p = checkpoint(self.p, e)
        q = checkpoint(self.q, e)
        return p @ q.transpose(1, 2) / 256


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
        out = (attn * x.view(n, ell * ell, -1)).sum(dim=1)
        return out


class Attention1d(nn.Module):
    
    def __init__(self, in_dim):
        super().__init__()
        self.layer = MaskedConv1d(in_dim, 1, 1)

    def forward(self, x, input_mask=None):
        n, ell, _ = x.shape
        attn = self.layer(x)
        attn = attn.view(n, -1)
        if input_mask is not None:
            attn = attn.masked_fill_(~input_mask.view(n, -1).bool(), float('-inf'))
        attn = F.softmax(attn, dim=-1).view(n, -1, 1)
        out = (attn * x).sum(dim=1)
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