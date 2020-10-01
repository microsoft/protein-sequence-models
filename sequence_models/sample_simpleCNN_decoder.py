from T1001_loader import *

import torch
import torch.nn as nn  
import torch.nn.functional as F 

import numpy as np

class SimpleConvDecoder(nn.Module):
    """
    Simple Convolutional Decoder - next residue prediction 
    Example:
    """
    def __init__(self, in_dim, h_dim, k, n_layers):
        """
        Parameters:
        -----------
        
        in_dim : int, 
            in channels
            
        h_dim : int, 
            hidden channels
            
        k : int, 
            kernel size
            
        n_layers : int,
            n layers of convolution
        """
        super(SimpleConvDecoder, self).__init__()
        
        self.h_dim = h_dim
        
        self.pad = self._pad(1, k, 1)
        self.conv0 = nn.Conv2d(in_dim, h_dim, k, padding=self.pad)
        
        self.conv_layer = nn.ModuleList([])
        for _ in range(n_layers):
            self.conv_layer.append(nn.Conv2d(h_dim, h_dim, k, padding=self.pad))
            self.conv_layer.append(nn.ReLU())
    
        self.attn_conv = nn.Conv2d(h_dim, 1, 1)
        
    def forward(self, x):
        """
        Parameters:
        -----------
        x : torch.Tensor, (N, in_dim, L, L)
        
        Returns:
        --------
        out : torch.Tensor, (N, h_dim, 1)
        """
        
        n_batch, _, w_in, h_in = x.shape
        
        # convolution layers
        x = F.relu(self.conv0(x))
        for layer in self.conv_layer:
            x = layer(x)
            
        # generate 2d attn tensor     
        attn = self.attn_conv(x)
    
        # generate mask according to padding 
        mask = self._mask(self.pad, attn)
        
        # flatten 
        x = x.view(n_batch, self.h_dim, -1)
        attn = attn.view(n_batch, 1, -1)
        mask = mask.view(n_batch, 1, -1)
        
        # apply mask onto attention and generate weights
        weights = F.softmax(mask + attn, dim=2)
        
        # get output from x * attention
        out = torch.matmul(x, torch.transpose(weights, -1, 1))
#         weights = weights.view(n_batch, 1, -1)
        
#         weights = F.softmax(mask * attn, dim=2)
#         weights[torch.isnan(weights)] = 0
        

        
#         out = torch.matmul(x, torch.transpose(weights,1,-1))
        
#         return out, weights, x
#         return weights
        return out
        
    def _pad(self, d, k, s):
        return int(((139 * s) - 140 + k + ((k - 1) * (d - 1))) / 2)
    
    def _mask(self, pad, x):
        mask = torch.zeros_like(x)
        el = x.size()[2]
        for i in range(pad):
            mask[:, :, i,:] = -float('inf')
            mask[:, :, el-1-i,:] = -float('inf')
            mask[:, :, :, i] = -float('inf')
            mask[:, :, :, el-1-i] = -float('inf')
        return mask
