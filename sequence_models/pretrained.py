import torch
import torch.nn as nn

from sequence_models.constants import PROTEIN_ALPHABET, PAD
from sequence_models.convolutional import ByteNetLM
from sequence_models.gnn import BidirectionalStruct2SeqDecoder
from sequence_models.collaters import SimpleCollater, StructureCollater


CARP_URL = 'https://zenodo.org/record/6564798/files/'
MIF_URL = 'https://zenodo.org/record/6573779/files/'
n_tokens = len(PROTEIN_ALPHABET)


def load_carp(model_data):
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
    sd = model_data['model_state_dict']
    model.load_state_dict(sd)
    model = CARP(model)
    return model

def load_gnn(model_data):
    one_hot_src = model_data['model'] == 'mif'
    gnn = BidirectionalStruct2SeqDecoder(n_tokens, 10, 11,
                                         256, num_decoder_layers=4,
                                         dropout=0.05, use_mpnn=True,
                                         pe=False, one_hot_src=one_hot_src)
    sd = model_data['model_state_dict']
    gnn.load_state_dict(sd)
    return gnn

def load_model_and_alphabet(model_name):
    if not model_name.endswith(".pt"):  # treat as filepath
        if 'carp' in model_name:
            url = CARP_URL + '%s.pt?download=1' %model_name
            model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
        elif 'mif' in model_name:
            url = MIF_URL + '%s.pt?download=1' %model_name
            model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    else:
        model_data = torch.load(model_name, map_location="cpu")
    collater = SimpleCollater(PROTEIN_ALPHABET, pad=True)
    if model_data['model'] == 'carp':
        model = load_carp(model_data)
    elif model_data['model'] in ['mif', 'mif-st']:
        gnn = load_gnn(model_data)
        cnn = None
        if model_data['model'] == 'mif-st':
            url = CARP_URL + '%s.pt?download=1' % 'carp_640M'
            cnn_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
            cnn = load_carp(cnn_data)
        collater = StructureCollater(collater, n_connections=30)
        model = MIF(gnn, cnn=cnn)
    return model, collater


class CARP(nn.Module):
    """Wrapper that takes care of input masking."""

    def __init__(self, model: ByteNetLM):
        super().__init__()
        self.model = model

    def forward(self, x, result='repr'):
        padding_mask = (x == PROTEIN_ALPHABET.index(PAD))
        padding_mask = padding_mask.unsqueeze(-1)
        if result == 'repr':
            return self.model.embedder(x, input_mask=padding_mask)
        elif result == 'logits':
            return self.model(x, input_mask=padding_mask)
        else:
            raise ValueError("Result must be either 'repr' or 'logits'")

class MIF(nn.Module):
    """Wrapper that takes care of input masking."""

    def __init__(self, gnn: BidirectionalStruct2SeqDecoder, cnn=None):
        super().__init__()
        self.gnn = gnn
        self.cnn = cnn

    def forward(self, src, nodes, edges, connections, edge_mask, result='repr'):
        if result == 'logits':
            decoder = True
        elif result == 'repr':
            decoder = False
        else:
            raise ValueError("Result must be either 'repr' or 'logits'")
        if self.cnn is not None:
            src = self.cnn(src, result='logits')
        return self.gnn(nodes, edges, connections, src, edge_mask, decoder=decoder)


