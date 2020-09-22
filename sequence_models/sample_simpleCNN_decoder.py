from T1001_loader import *

import torch
import torch.nn as nn  
import torch.nn.functional as F 

import numpy as np

# processing steps 
def get_mask(x):
    p_mask = np.max(np.isnan(x)*1, axis=0)
    mask = np.ones(p_mask.shape) - p_mask
    return torch.from_numpy(mask).float()
    
def preprocess_x(x):
    mask = get_mask(x)
    isnan = np.isnan(x)
    x[isnan] = 0
    return torch.from_numpy(x), mask


class SimpleConvDecoder(nn.Module):
    """
    Simple Convolutional Decoder - next residue prediction 
    Example:
        x, mask = preprocess_x(x)
        x = x.view(1, x.shape[0], x.shape[1], x.shape[2]).float()
        model = SimpleConvDecoder(in_dim=4, W_dim=4, h_dim=128, L=140, 
            num_letters=20, n_batch=1)
        # if hidden needs to be init, use initHidden
        # hidden = model.initHidden()
        output, hidden = model(x, hidden, mask)
        
    """
    def __init__(self, in_dim, W_dim, h_dim, L, n_letters, n_batch):
        """
        Parameters:
        -----------
        
        in_dim : int
            in channels
        
        W_dim : int
            conv hidden dim
        
        h_him : int
            hidden layer dim
        
        L : int
            length of seq
        
        num_letters : int
            length of protein vocab
            
        n_batch : int
            batch_size
        
        """
        super(SimpleConvDecoder, self).__init__()
        self.h_dim = h_dim
        self.n_batch = n_batch
        self.conv1 = nn.Conv2d(in_dim, W_dim, 1)
        self.conv2 = nn.Conv2d(W_dim, W_dim, 1)
        
        self.struct_linear = nn.Linear(W_dim * L * L, h_dim)
        
        self.hidden_linear = nn.Linear(h_dim*2, h_dim)
        
        self.out_linear = nn.Linear(h_dim*2, n_letters)
        
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x, hidden, mask):
        """
        Parameters:
        -----------
        x : torch.Tensor, (N_batch, 4, L, L)
            inputs
        
        hidden : torch.Tensor, (N_batch, h_dim)
            hidden state
        
        mask : torch.Tensor, (N_batch, L, L)
            mask nodes with missing values
            
        Returns:
        --------
        output : torch.Tensor, (N_batch, n_letters)
            log prob of residue prediction
        
        hidden : torch.Tensor, (N_batch, h_dim)
            hidden state
            
        
        """
        
        x = F.relu(self.conv1(x))
        x = x * mask
        x = F.relu(self.conv2(x))
        structure_enc = (x * mask).view(self.n_batch,-1)
        structure_enc = self.struct_linear(structure_enc)
        
        combined = torch.cat((structure_enc, hidden), 1)
        
        hidden = self.hidden_linear(combined)
        
        output = self.out_linear(combined)
        
        output = self.softmax(output)
        
        return output, hidden
    
    def initHidden(self):
        return torch.zeros(1, self.h_dim)