import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_models.trRosetta_utils import *
from sequence_models.constants import WEIGHTS_DIR
from sequence_models.layers import MaskedInstanceNorm2d


def pad_size(d, k, s):
    return int(((139 * s) - 140 + k + ((k - 1) * (d - 1))) / 2)


class trRosettaBlock(nn.Module):
        
    def __init__(self, dilation, p_dropout=0.0):
        
        """Simple convolution block
        
        Parameters:
        -----------
        dilation : int
            dilation for conv
        """

        super(trRosettaBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=dilation, padding=pad_size(dilation, 3, 1))
        self.instnorm1 = MaskedInstanceNorm2d(64, eps=1e-06, affine=True)
        self.instnorm2 = MaskedInstanceNorm2d(64, eps=1e-06, affine=True)
        self.dropout1 = nn.Dropout2d(p_dropout)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=dilation, padding=pad_size(dilation, 3, 1))

    def forward(self, x, input_mask=None, last_elu=True):
        """
        Parameters:
        -----------
        x : torch.Tensor()
            input tensor
            
        old_elu : torch.Tensor()
            copy of x
        
        Returns:
        --------
        x : torch.Tensor
            output of block

        x.clone() : torch.Tensor
            copy of x
        """
        if input_mask is not None:
            x = x * input_mask
        h = F.elu(self.instnorm1(self.conv1(x), input_mask=input_mask))
        h = self.dropout1(h)
        if input_mask is not None:
            h = h * input_mask
        h = self.instnorm2(self.conv2(h), input_mask=input_mask) + x
        if last_elu:
            h = F.elu(h)
        return h


class trRosetta(nn.Module):
    
    """trRosetta for single model"""

    def __init__(self, d_init=526, n2d_layers=61, model_id='a', decoder=True, p_dropout=0.0):
        """
        Parameters:
        -----------
        model_id : str
            pretrained models a, b, c, d and/or e.
    
        decoder : bool
            whether to run the last layers to produce distance 
            and angle outputs

        """
        super(trRosetta, self).__init__()

        self.conv0 = nn.Conv2d(d_init, 64, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        self.instnorm0 = MaskedInstanceNorm2d(64, eps=1e-06, affine=True)

        dilation = 1
        layers = []
        for _ in range(n2d_layers):
            layers.append(trRosettaBlock(dilation, p_dropout=p_dropout))
            dilation *= 2
            if dilation > 16:
                dilation = 1

        self.layers = nn.ModuleList(modules=layers)
        self.decoder = decoder
        if decoder:
            self.softmax = nn.Softmax(dim=1)
            self.conv_theta = nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
            self.conv_phi = nn.Conv2d(64, 13, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
            self.conv_dist = nn.Conv2d(64, 37, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
            self.conv_bb = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
            self.conv_omega = nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        if model_id is not None:
            self.load_weights(model_id)

    def forward(self, x, input_mask=None, softmax=True):
        """
        Parameters:
        -----------
        x : torch.Tensor, (batch, 526, len(sequence), len(sequence))
            inputs after trRosettaPreprocessing
    
        Returns:
        --------
        dist_probs : torch.Tensor
            distance map probabilities
            
        theta_probs : torch.Tensor
            theta angle map probabilities
            
        phi_probs : torch.Tensor
            phi angle map probabilities
        
        omega_probs: torch..Tensor
            omega angle map probabilities
        
        x : torch.Tensor
            outputs before calculating final layers
        """
        if input_mask is not None:
            x = x * input_mask
        h = self.conv0(x)
        h = F.elu(self.instnorm0(h, input_mask=input_mask))
        for i, layer in enumerate(self.layers):
            if not self.decoder:
                last_elu = False
            elif i != len(self.layers) - 1:
                last_elu = True
            else:
                last_elu = False
            h = layer(h, input_mask=input_mask, last_elu=last_elu)
            if input_mask is not None:
                h = h * input_mask
        if self.decoder:
            logits_theta = self.conv_theta(h)

            logits_phi = self.conv_phi(h)

            # symmetrize
            h = 0.5 * (h + torch.transpose(h, 2, 3))

            logits_dist = self.conv_dist(h)

            logits_omega = self.conv_omega(h)
            if not softmax:
                return logits_dist, logits_theta, logits_phi, logits_omega
            else:
                theta_probs = self.softmax(logits_theta)
                phi_probs = self.softmax(logits_phi)
                dist_probs = self.softmax(logits_dist)
                omega_probs = self.softmax(logits_omega)
                return dist_probs, theta_probs, phi_probs, omega_probs
        else:
            return h

    def load_weights(self, model_id):
        
        """
        Parameters:
        -----------
        model_id : str
            pretrained models a, b, c, d and/or e.
        """

        path = WEIGHTS_DIR + 'trrosetta_pytorch_weights/' + model_id + '.pt'

        # check to see if pytorch weights exist, if not -> generate
        if not os.path.exists(path):
            tf_to_pytorch_weights(self.named_parameters(), model_id)
        self.load_state_dict(torch.load(path, ), strict=False)


class trRosettaRegressor(trRosetta):

    def __init__(self, model_id='a', p_dropout=0.0):
        super(trRosettaRegressor, self).__init__(n2d_layers=61, model_id=model_id, decoder=False, p_dropout=p_dropout)
        self.dist_layer = nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        self.theta_layer = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        self.phi_layer =  nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        self.omega_layer = nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        self.relu = nn.ReLU()
        self.tanh = nn.Hardtanh()

    def forward(self, x, input_mask=None, softmax=False):
        h = super(trRosettaRegressor, self).forward(x, input_mask=input_mask)
        h = F.elu(h)
        sc_theta = self.tanh(self.theta_layer(h))
        sc_phi = self.tanh(self.phi_layer(h))

        # symmetrize
        h = 0.5 * (h + torch.transpose(h, 2, 3))
        dist = self.relu(self.dist_layer(h))
        sc_omega = self.tanh(self.omega_layer(h))
        return dist, sc_theta, sc_phi, sc_omega


class trRosettaEnsemble(nn.Module):
    """trRosetta ensemble"""
    def __init__(self, model, n2d_layers=61, model_ids='abcde', decoder=True):
        """
        Parameters:
        -----------
        model : class 
            base model to use in ensemble
        
        n2d_layers : int 
            number of layers of the conv block to use for each base model
        
        model_ids: str
            pretrained models to use in the ensemble a, b, c, d and/or e. 
            
        decoder : bool
            if True, return dist, omega, phi, theta; else return layer prior decoder
        
        """

        super(trRosettaEnsemble, self).__init__()
        self.model_list = nn.ModuleList()
        for i in list(model_ids):
            params = {'model_id': i, 'n2d_layers': n2d_layers, 'decoder': decoder}
            self.model_list.append(model(**params))

    def forward(self, x, input_mask=None, softmax=True):
        """
        Parameters:
        -----------
        x : torch.Tensor, (1, 526, len(sequence), len(sequence))
            inputs after trRosettaPreprocessing
        """
        return [mod(x, input_mask=input_mask, softmax=softmax) for mod in self.model_list]


class trRosettaDist(nn.Module):
    """trRosetta for distance only, does not use pretrained weights"""
    def __init__(self, n2d_layers=61, hdim=128, decoder=True, d_out=1):
        """
        Args:
            n2d_layers: int
                number of layers of the conv block to use for each base model
            hdim: int
                input 1d hidden dimension
            decoder: bool
                 if True, return dist; else return layer prior decoder
        """
        super(trRosettaDist, self).__init__()

        self.conv0 = nn.Conv2d(hdim * 2, 64, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))
        self.instnorm0 = nn.InstanceNorm2d(64, eps=1e-06, affine=True)

        dilation = 1
        layers = []
        for _ in range(n2d_layers):
            layers.append(trRosettaBlock(dilation))
            dilation *= 2
            if dilation > 16:
                dilation = 1

        self.layers = nn.ModuleList(modules=layers)
        self.decoder = decoder

        if decoder:
            self.conv_dist = nn.Conv2d(64, d_out, kernel_size=1, stride=1, padding=pad_size(1, 1, 1))

    def forward(self, x, ):
        """
        Args:
            x: torch.tensor (N, L, hdim)
        Returns:
            dist: torch.tensor(), (N, L, L)
            x: torch.tensor(), (N, 64, L, L)

        """
        n, el, _ = x.shape

        # convert to 2d
        left = x.unsqueeze(2).repeat(1, 1, el, 1)
        right = x.unsqueeze(1).repeat(1, el, 1, 1)
        x = torch.cat((left, right), -1)
        x = x.permute(0, 3, 1, 2)

        x = F.elu(self.instnorm0(self.conv0(x)))
        old_elu = x.clone()
        for layer in self.layers:
            x, old_elu = layer(x, old_elu)

        if self.decoder:
            # symmetrize
            ## TODO: Some things need to be symmetrical and others don't
            x = 0.5 * (x + torch.transpose(x, 2, 3))
            dist = self.conv_dist(x).squeeze(1)
            return dist
        else:
            return x

# EXAMPLE
# filename = 'example/T1001.a3m' 
# seqs = parse_a3m(filename) # grab seqs
# tokenizer = Tokenizer(PROTEIN_ALPHABET) 
# seqs = [tokenizer.tokenize(i) for i in seqs] # ohe into our order

# base_model = trRosetta
# input_token_order = PROTEIN_ALPHABET
# ensemble = trRosettaEnsemble(base_model, n2d_layers=61,model_ids='abcde')
# preprocess = trRosettaPreprocessing(input_token_order=PROTEIN_ALPHABET, wmin=0.8)
# x = preprocess.process(seqs)
# with torch.no_grad():
#     ensemble.eval()
#     outputs = ensemble(x.double())
