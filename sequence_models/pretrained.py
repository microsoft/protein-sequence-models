import torch
import torch.nn as nn

from sequence_models.constants import PROTEIN_ALPHABET, PAD
from sequence_models.convolutional import ByteNetLM
from sequence_models.collaters import SimpleCollater


def load_model_and_alphabet(model_name):
    if not model_name.endswith(".pt"):  # treat as filepath
        url = 'https://zenodo.org/record/6368484/files/%s.pt?download=1' %model_name
        model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    else:
        model_data = torch.load(model_name, map_location="cpu")
    sd = model_data['model_state_dict']
    n_tokens = len(PROTEIN_ALPHABET)
    if model_data['model'] == 'carp':
        d_embedding = model_data['d_embed']
        d_model = model_data['d_model']
        n_layers = model_data['n_layers']
        kernel_size = model_data['kernel_size']
        activation = model_data['activation']
        slim = model_data['slim']
        r = model_data['r']
        model = ByteNetLM(n_tokens, d_embedding, d_model, n_layers, kernel_size, r, dropout=0.0,
                          activation=activation, causal=False, padding_idx=PROTEIN_ALPHABET.index(PAD),
                          final_ln=True, slim=slim)
        collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    model.load_state_dict(sd)
    return model, collater


class CARP(nn.Module):
    """Wrapper that takes care of input masking."""

    def __init__(self, model: ByteNetLM):
        super().__init__()
        self.model = model

    def forward(self, x, result='repr'):
        padding_mask = x == PROTEIN_ALPHABET.index(PAD)
        if result == 'repr':
            return self.model.embedder(x, input_mask=padding_mask)
        elif result == 'logits':
            return self.model(x, input_mask=padding_mask)
        else:
            raise ValueError("Result must be either 'repr' or 'logits'")

