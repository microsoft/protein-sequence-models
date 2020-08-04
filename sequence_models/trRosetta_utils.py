import torch
import torch.nn as nn
import numpy as np
import os
import tensorflow as tf
import wget
import tarfile
import string

# probably move this into a collate_fn 
class trRosettaPreprocessing():
    
    def __init__(self, input_token_order, wmin):
        self.ohe_dict = self._build_ohe_dict(input_token_order)
        self.wmin = wmin
        self.seqlen = 0

    def _build_ohe_dict(self, input_order):
        trR_order = "ARNDCQEGHILKMFPSTWYV-"
        ohe_dict = {}
        for i in input_order:
            if i in trR_order:
                ohe_dict[input_order.index(i)] = trR_order.index(i)
            else:
                ohe_dict[input_order.index(i)] = trR_order.index('-')
        return ohe_dict
    
    def _convert_ohe(self, seqs, ohe_dict):
        processed_seqs = []
        for seq in seqs:
            processed_seqs.append([ohe_dict[i] for i in seq])
        return torch.Tensor(np.array(processed_seqs)).to(torch.int8)
    
    def _one_hot_embedding(self, seqs, num_classes):
        one_hot_embedded = []
        for seq in seqs:
            encoded = torch.eye(num_classes) 
            one_hot_embedded.append(encoded[seq.long()]) 
        stacked = torch.stack(one_hot_embedded)
        return stacked.reshape((1,stacked.shape[0], stacked.shape[1], stacked.shape[2]))
    
    def _reweight_py(self, msa1hot, cutoff, eps=1e-9):
        self.seqlen = msa1hot.size(2)
        id_min = self.seqlen * cutoff
        id_mtx = torch.stack([torch.tensordot(el, el, [[1, 2], [1, 2]]) for el in msa1hot], 0)
        id_mask = id_mtx > id_min
        weights = 1.0 / (id_mask.type_as(msa1hot).sum(-1) + eps)
        return weights
    
    def _extract_features_1d(self, msa1hot, weights):
        # 1D Features
        f1d_seq = msa1hot[:, 0, :, :20]
        batch_size = msa1hot.size(0)

        # msa2pssm
        beff = weights.sum()
        f_i = (weights[:, :, None, None] * msa1hot).sum(1) / beff + 1e-9
        h_i = (-f_i * f_i.log()).sum(2, keepdims=True)
        f1d_pssm = torch.cat((f_i, h_i), dim=2)
        f1d = torch.cat((f1d_seq, f1d_pssm), dim=2)
        f1d = f1d.view(batch_size, self.seqlen, 42)
        return f1d
    
    def _extract_features_2d(self, msa1hot, weights, penalty=4.5):
        # 2D Features
        batch_size = msa1hot.size(0)
        num_alignments = msa1hot.size(1)
        num_symbols = 21

        if num_alignments == 1:
            # No alignments, predict from sequence alone
            f2d_dca = torch.zeros(
                batch_size, self.seqlen, self.seqlen, 442,
                dtype=torch.float,
                device=msa1hot.device)
            return f2d_dca

        # compute fast_dca
        # covariance
        x = msa1hot.view(batch_size, num_alignments, self.seqlen * num_symbols)
        num_points = weights.sum(1) - weights.mean(1).sqrt()
        mean = (x * weights.unsqueeze(2)).sum(1, keepdims=True) / num_points[:, None, None]
        x = (x - mean) * weights[:, :, None].sqrt()
        cov = torch.matmul(x.transpose(-1, -2), x) / num_points[:, None, None]

        # inverse covariance
        reg = torch.eye(self.seqlen * num_symbols,
                            device=weights.device,
                            dtype=weights.dtype)[None]
        reg = reg * penalty / weights.sum(1, keepdims=True).sqrt().unsqueeze(2)
        cov_reg = cov + reg
        inv_cov = torch.stack([torch.inverse(cr) for cr in cov_reg.unbind(0)], 0)

        x1 = inv_cov.view(batch_size, self.seqlen, num_symbols, self.seqlen, num_symbols)
        x2 = x1.permute(0, 1, 3, 2, 4)
        features = x2.reshape(batch_size, self.seqlen, self.seqlen, num_symbols * num_symbols)

        x3 = (x1[:, :, :-1, :, :-1] ** 2).sum((2, 4)).sqrt() * (
                1 - torch.eye(self.seqlen, device=weights.device, dtype=weights.dtype)[None])
        apc = x3.sum(1, keepdims=True) * x3.sum(2, keepdims=True) / x3.sum(
                (1, 2), keepdims=True)
        contacts = (x3 - apc) * (1 - torch.eye(
                self.seqlen, device=x3.device, dtype=x3.dtype).unsqueeze(0))

        f2d_dca = torch.cat([features, contacts[:, :, :, None]], axis=3)
        return f2d_dca
    
    def process(self, x):
        x = self._convert_ohe(x, self.ohe_dict).reshape(len(x),-1)
        x = self._one_hot_embedding(x, 21)
        w = self._reweight_py(x, self.wmin) 
        f1d = self._extract_features_1d(x, w)
        f2d = self._extract_features_2d(x, w)
        
        left = f1d.unsqueeze(2).repeat(1, 1, self.seqlen, 1)
        right = f1d.unsqueeze(1).repeat(1, self.seqlen, 1, 1)
        features = torch.cat((left, right, f2d), -1)
        features = features.permute(0, 3, 1, 2)
        
        return features


def tf_to_pytorch_weights(model_params, model_id):

    # check to see if previously downloaded weights, if not -> download
    if not os.path.exists('model_weights'):
        os.mkdir('model_weights')
            
    if len(os.listdir('model_weights')) == 0:
        print('grabbing weights from source...')
        wget.download('https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2', out = 'model_weights')
        model_file = tarfile.open('model_weights/' + os.listdir('model_weights')[0], mode = 'r:bz2')
        model_file.extractall('model_weights')
        model_file.close()
        
        
    # check to see if converted to pytorch weights yet
    if not os.path.exists('model_weights/pytorch_weights'):
        os.mkdir('model_weights/pytorch_weights')
        
    model_path = 'model_weights/pytorch_weights/pytorch_weights_' + model_id + '.pt'
    
    if not os.path.exists(model_path):
        print('converting model %s weights from tensorflow to pytorch...' % model_id)
    
        tf_path = next(os.walk('model_weights'))[1][0]
        ckpt = 'model_weights/'+ tf_path + '/' + 'model.xa' + model_id
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
        
        # convert into pytorch format
        torch_weight_dict = {}
        weights_idx = 0
        for name, param in model_params:
            if len(weights_list[weights_idx].shape) == 4:
                torch_weight_dict[name] = torch.from_numpy(weights_list[weights_idx]).to(torch.float64).permute(3,2,0,1)
            else:
                torch_weight_dict[name] = torch.from_numpy(weights_list[weights_idx]).to(torch.float64)
            weights_idx += 1
            
        torch.save(torch_weight_dict, model_path)


def parse_a3m(filename):
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename,"r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
    return seqs