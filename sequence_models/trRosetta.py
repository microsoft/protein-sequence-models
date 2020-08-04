import os,sys
import torch
import torch.nn as nn
import torch.nn.functional as F

from sequence_models.trRosetta_utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pad_size(d, k, s):
    return int(((139*s) - 140 + k + ((k-1)*(d-1)))/2)

class trRosettaBlock(nn.Module):
    
    def __init__(self, dilation):
        super(trRosettaBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=dilation, padding=pad_size(dilation,3,1))
        self.instnorm1 = nn.InstanceNorm2d(64, eps=1e-06, affine=True)
#         self.dropout1 = nn.Dropout2d(0.15)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=dilation, padding=pad_size(dilation,3,1) )
        self.instnorm2 = nn.InstanceNorm2d(64, eps=1e-06, affine=True)
        
    def forward(self, x, old_elu,):
        x = F.elu(self.instnorm1(self.conv1(x)))
#         x = self.dropout1(x)
        x = F.elu(self.instnorm2(self.conv2(x)) + old_elu)
        return x, x.clone()
        

class trRosetta(nn.Module):

    def __init__(self, n2d_layers, model_id='a'):
        super(trRosetta, self).__init__()
        
        self.conv0 = nn.Conv2d(526, 64, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.instnorm0 = nn.InstanceNorm2d(64, eps=1e-06, affine=True)
        
        dilation = 1
        layers = []
        for _ in range(n2d_layers): 
            layers.append(trRosettaBlock(dilation))
            dilation *= 2
            if dilation > 16:
                dilation = 1
        
        self.layers = nn.ModuleList(modules=layers)
        
        self.conv_theta = nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_theta = nn.Softmax(dim=1)
        
        self.conv_phi = nn.Conv2d(64, 13, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_phi = nn.Softmax(dim=1)
        
        self.conv_dist = nn.Conv2d(64, 37, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_dist = nn.Softmax(dim=1)

        self.conv_bb = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_bb = nn.Softmax(dim=1)
        
        self.conv_omega = nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_omega = nn.Softmax(dim=1)

        self.load_weights(model_id)
        
    def forward(self, x,):
        x = F.elu(self.instnorm0(self.conv0(x)))
        old_elu = x.clone()
        for layer in self.layers:
            x, old_elu = layer(x, old_elu)
        
        logits_theta = self.conv_theta(x)
        theta_probs = self.softmax_theta(logits_theta)
        
        logits_phi = self.conv_phi(x)
        phi_probs = self.softmax_phi(logits_phi)
        
        # symmetrize
        x = 0.5 * (x + torch.transpose(x,2,3))
    
        logits_dist = self.conv_dist(x)
        dist_probs = self.softmax_dist(logits_dist)
        
        logits_bb = self.conv_bb(x)
        bb_probs = self.softmax_bb(logits_bb)
        
        logits_omega = self.conv_omega(x)
        omega_probs = self.softmax_omega(logits_omega)

        return dist_probs, theta_probs, phi_probs, omega_probs
    
    def load_weights(self, model_id):
        
        path = 'model_weights/pytorch_weights/' + 'pytorch_weights_' + model_id + '.pt'
        
        # check to see if pytorch weights exist, if not -> generate
        if not os.path.exists(path):
            tf_to_pytorch_weights(self.named_parameters(), model_id)
        self.load_state_dict(torch.load(path,))
        
        
class trRosettaEnsemble(nn.Module):
    def __init__(self, model, n2d_layers=61, model_ids='abcde', device=device):
        '''
        Parameters:
        -----------
        model: base model in for ensemble
        
        n2d_layers: number of layers of the conv block to use for each base model
        
        model_ids: pretrained models a, b, c, d and/or e. 
        
        '''
        
        super(trRosettaEnsemble, self).__init__()
        self.model_list = []
        for i in list(model_ids):
            params = {'model_id':i, 'n2d_layers':n2d_layers}
            self.model_list.append(model(**params).to(device=device, dtype=torch.double))
        
    def forward(self,x):
        output = []
        for mod in self.model_list:
            output.append(mod(x))
            
        return output

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