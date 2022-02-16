import torch.nn as nn


from esm.modules import TransformerLayer, LearnedPositionalEmbedding, RobertaLMHead, ESM1bLayerNorm
from sequence_models.constants import PROTEIN_ALPHABET, PAD, MASK


class ESM1b(nn.Module):

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
