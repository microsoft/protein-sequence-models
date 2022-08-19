import torch.nn as nn
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint

from esm.modules import TransformerLayer, LearnedPositionalEmbedding, RobertaLMHead, ESM1bLayerNorm, \
    AxialTransformerLayer
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK


class ESM1b(nn.Module):
    """
    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, n_tokens=len(PROTEIN_ALPHABET),
                 padding_idx=PROTEIN_ALPHABET.index(PAD), mask_idx=PROTEIN_ALPHABET.index(MASK),
                 max_positions=1024):
        super(ESM1b, self).__init__()
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model, d_hidden, n_heads,
                    add_bias_kv=False,
                    use_esm1b_layer_norm=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_idx = padding_idx

        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = ESM1bLayerNorm(d_model)
        self.emb_layer_norm_after = ESM1bLayerNorm(d_model)
        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=n_tokens,
            weight=self.embed_tokens.weight
        )

    def forward(self, tokens):

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_tokens(tokens)

        # if getattr(self.args, 'token_dropout', False):
        #     x.masked_fill_((tokens == self.mask_idx).unsqueeze(-1), 0.0)
        #     # x: B x T x C
        #     mask_ratio_train = 0.15 * 0.8
        #     src_lengths = (~padding_mask).sum(-1)
        #     mask_ratio_observed = (tokens == self.mask_idx).sum(-1).float() / src_lengths
        #     x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]

        x = x + self.embed_positions(tokens)

        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # (B, T, E) => (T, B, E)
        x = x.transpose(0, 1)

        if not padding_mask.any():
            padding_mask = None

        for layer_idx, layer in enumerate(self.layers):
            x, attn = layer(x, self_attn_padding_mask=padding_mask, need_head_weights=False)

        x = self.emb_layer_norm_after(x)
        x = x.transpose(0, 1)  # (T, B, E) => (B, T, E)
        x = self.lm_head(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=8, length=500):
        super().__init__()
        self.d_model = d_model
        self.length = length

    def forward(self, x):
        """
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if self.d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(self.d_model))
        pe = torch.zeros(self.length, self.d_model)
        position = torch.arange(0, self.length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, self.d_model, 2, dtype=torch.float) *
                              -(torch.log(10000.0) / self.d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe[x].to(x.device)


class MSATransformer(nn.Module):
    """
    Based on implementation described by Rao et al. in "MSA Transformer"
    https://doi.org/10.1101/2021.02.12.430858

    Args:
        d_model: int,
            embedding dimension of model
        d_hidden: int,
            embedding dimension of feed forward network
       n_layers: int,
           number of layers
       n_heads: int,
           number of attention heads
   """

    def __init__(self, d_model, d_hidden, n_layers, n_heads, timesteps=500, use_ckpt=False,
                 n_tokens=len(PROTEIN_ALPHABET),
                 padding_idx=PROTEIN_ALPHABET.index(PAD), mask_idx=PROTEIN_ALPHABET.index(MASK),
                 max_positions=1024):  # TODO: read max timesteps in from config
        super(MSATransformer, self).__init__()
        self.embed_tokens = nn.Embedding(
            n_tokens, d_model, padding_idx=mask_idx
        )
        self.layers = nn.ModuleList(
            [
                AxialTransformerLayer(
                    d_model, d_hidden, n_heads
                )
                for _ in range(n_layers)
            ]
        )
        self.padding_idx = padding_idx

        # self.contact_head = ContactPredictionHead()
        self.embed_positions = LearnedPositionalEmbedding(max_positions, d_model, padding_idx)
        self.emb_layer_norm_before = nn.LayerNorm(d_model)
        self.emb_layer_norm_after = nn.LayerNorm(d_model)
        self.lm_head = RobertaLMHead(
            embed_dim=d_model,
            output_dim=n_tokens,
            weight=self.embed_tokens.weight
        )

        self.use_ckpt = use_ckpt
        self.time_encoding = PositionalEncoding(d_model, timesteps)

    def forward(self, tokens, y=None, use_time=False):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C

        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())
        if use_time:
            val = self.time_encoding(y)
            val = val.expand(x.shape[1], val.shape[0], val.shape[1])
            val = val.reshape(x.shape[0], x.shape[1], x.shape[2])
            x = x + val

        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, None, padding_mask, False)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x
