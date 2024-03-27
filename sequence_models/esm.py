import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from esm.modules import TransformerLayer, LearnedPositionalEmbedding, ESM1bLayerNorm, AxialTransformerLayer
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK


class RobertaLMHead(nn.Module):
    """Head for masked language modeling."""

    def __init__(self, embed_dim, output_dim, weight):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.layer_norm = ESM1bLayerNorm(embed_dim)
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        # project back to size of vocabulary with bias
        x = F.linear(x, self.weight) + self.bias
        return x

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
                 max_positions=1024, tie_weights=True):
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
        if tie_weights:
            self.lm_head = RobertaLMHead(
                embed_dim=d_model,
                output_dim=n_tokens,
                weight=self.embed_tokens.weight
            )
        else:
            self.lm_head = RobertaLMHead(
                embed_dim=d_model,
                output_dim=n_tokens,
                weight=nn.Linear(d_model, n_tokens).weight
            )

    def forward(self, tokens):

        assert tokens.ndim == 2
        padding_mask = tokens.eq(self.padding_idx)  # B, T

        x = self.embed_tokens(tokens)
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

    def __init__(self, d_model, d_hidden, n_layers, n_heads, use_ckpt=False, n_tokens=len(PROTEIN_ALPHABET),
                 padding_idx=PROTEIN_ALPHABET.index(PAD), mask_idx=PROTEIN_ALPHABET.index(MASK),
                 max_positions=1024, tie_weights=True):
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
        if tie_weights:
            self.lm_head = RobertaLMHead(
                embed_dim=d_model,
                output_dim=n_tokens,
                weight=self.embed_tokens.weight
            )
        else:
            self.lm_head = RobertaLMHead(
                embed_dim=d_model,
                output_dim=n_tokens,
                weight=nn.Linear(d_model, n_tokens).weight
            )

        self.use_ckpt = use_ckpt

    def forward(self, tokens):
        assert tokens.ndim == 3
        batch_size, num_alignments, seqlen = tokens.size()
        padding_mask = tokens.eq(self.padding_idx)  # B, R, C

        x = self.embed_tokens(tokens)
        x = x + self.embed_positions(tokens.view(batch_size * num_alignments, seqlen)).view(x.size())

        x = self.emb_layer_norm_before(x)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # B x R x C x D -> R x C x B x D
        x = x.permute(1, 2, 0, 3)

        for layer_idx, layer in enumerate(self.layers):
            x = checkpoint(layer, x, None, padding_mask, False, use_reentrant=True)

        x = self.emb_layer_norm_after(x)
        x = x.permute(2, 0, 1, 3)  # R x C x B x D -> B x R x C x D
        x = self.lm_head(x)
        return x