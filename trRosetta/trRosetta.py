import os,sys
import tensorflow as tf
import wget
import tarfile
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from arguments import *

def pad_size(d, k, s):
    return int(((139*s) - 140 + k + ((k-1)*(d-1)))/2)

class trRosettaBlock(nn.Module):
    
    def __init__(self, dilation):
        super(trRosettaBlock, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=dilation, padding=pad_size(dilation,3,1))
        self.instnorm1 = nn.InstanceNorm2d(n2d_filters, eps=1e-06, affine=True)
#         self.dropout1 = nn.Dropout2d(0.15)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=dilation, padding=pad_size(dilation,3,1) )
        self.instnorm2 = nn.InstanceNorm2d(n2d_filters, eps=1e-06, affine=True)
        
    def forward(self, x, old_elu,):
        x = F.elu(self.instnorm1(self.conv1(x)))
#         x = self.dropout1(x)
        x = F.elu(self.instnorm2(self.conv2(x)) + old_elu)
        return x, x.clone()
        

class trRosetta(nn.Module):

    def __init__(self, n2d_layers, model_id='a'):
        super(trRosetta, self).__init__()
        
        self.conv0 = nn.Conv2d(526, n2d_filters, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.instnorm0 = nn.InstanceNorm2d(n2d_filters, eps=1e-06, affine=True)
        
        dilation = 1
        layers = []
        for _ in range(n2d_layers): 
            layers.append(trRosettaBlock(dilation))
            dilation *= 2
            if dilation > 16:
                dilation = 1
        
        self.layers = nn.ModuleList(modules=layers)
        
        self.conv_theta = nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_theta = nn.Softmax()
        
        self.conv_phi = nn.Conv2d(64, 13, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_phi = nn.Softmax()
        
        self.conv_dist = nn.Conv2d(64, 37, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_dist = nn.Softmax()

        self.conv_bb = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_bb = nn.Softmax()
        
        self.conv_omega = nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=pad_size(1,1,1))
        self.softmax_omega = nn.Softmax()

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
        
        # check to see if previously downloaded
        if not os.path.exists('model_weights'):
            os.mkdir('model_weights')
        if len(os.listdir('model_weights')) == 0:
            wget.download('https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2', out = 'model_weights')
            model_file = tarfile.open('model_weights/' + os.listdir('model_weights')[0], mode = 'r:bz2')
            model_file.extractall('model_weights')
            model_file.close()
        
        # extract weights from model_file
        filename = next(os.walk('model_weights'))[1][0]
        all_model = os.listdir('model_weights/' + filename)
        ckpt = 'model_weights/'+ filename + '/' + 'model.xa' + model_id
        print('Loading in model: xa' + model_id)
        w_vars = tf.train.list_variables(ckpt) # get weight names
        
        # filter weights
        w_vars_fil = [i for i in w_vars if 'Adam' not in i[0]]
        instnorm_beta_vars = [i[0] for i in w_vars_fil if 'InstanceNorm' in i[0] and 'beta' in i[0]]
        instnorm_gamma_vars = [i[0] for i in w_vars_fil if 'InstanceNorm' in i[0] and 'gamma' in i[0]]
        conv_kernel_vars = [i[0] for i in w_vars_fil if 'conv2d' in i[0] and 'kernel' in i[0]]
        conv_bias_vars = [i[0] for i in w_vars_fil if 'conv2d' in i[0] and 'bias' in i[0]]

        # order weights
        w_vars_ord = [conv_kernel_vars[0], conv_bias_vars[0], instnorm_gamma_vars[0], instnorm_beta_vars[0]]
        for i in range(len(conv_kernel_vars)):
            if 'conv2d_' + str(i) + '/kernel' in conv_kernel_vars:
                w_vars_ord.append('conv2d_' + str(i) + '/kernel')
            if 'conv2d_' + str(i) + '/bias' in conv_bias_vars:
                w_vars_ord.append('conv2d_' + str(i) + '/bias')
            if 'InstanceNorm_' + str(i) + '/gamma' in instnorm_gamma_vars:
                w_vars_ord.append('InstanceNorm_' + str(i) + '/gamma')
            if 'InstanceNorm_' + str(i) + '/beta' in instnorm_beta_vars:
                w_vars_ord.append('InstanceNorm_' + str(i) + '/beta')
        
#         tf_weight_dict = {name:tf.train.load_variable(ckpt, name) for name in w_vars_ord}
        weights_list = [tf.train.load_variable(ckpt, name) for name in w_vars_ord]
        
        # replace pytorch model weights
        torch_weight_dict = {}
        weights_idx = 0
        for name, param in self.named_parameters():
            if len(weights_list[weights_idx].shape) == 4:
                torch_weight_dict[name] = torch.from_numpy(weights_list[weights_idx]).to(torch.float64).permute(3,2,0,1)
            else:
                torch_weight_dict[name] = torch.from_numpy(weights_list[weights_idx]).to(torch.float64)
            weights_idx += 1
    
        self.load_state_dict(torch_weight_dict)
        
        
        
class trRosettaEnsemble(nn.Module):
    def __init__(self, model, n2d_layers=61, model_ids='abcde'):
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
            self.model_list.append(model(**params).double())
        
    def forward(self,x):
        output = []
        for mod in self.model_list:
            output.append(mod(x))
            
        return output