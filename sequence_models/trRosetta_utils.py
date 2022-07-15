import torch
import torch.nn.functional as F
import numpy as np
import os
import tarfile
import string

from sequence_models.constants import WEIGHTS_DIR, trR_ALPHABET, DIST_BINS, PHI_BINS, OMEGA_BINS, THETA_BINS


def probs2value(array, property, mask2d):
    # input shape: batch, n_bins, ell, ell
    # output shape: batch, ell, ell
    if property == 'dist':
        bins = DIST_BINS
    elif property == 'phi':
        bins = PHI_BINS
    elif property == 'omega':
        bins = OMEGA_BINS
    elif property == 'theta':
        bins = THETA_BINS
    if property == 'dist':
        bins = torch.tensor(np.nan_to_num(bins), device=array.device, dtype=array.dtype)
        b = (bins[:-1] + bins[1:]) / 2
        diff = b[-1] - b[-2]
        b[0] = b[-1] + diff
    else:
        b = torch.tensor((bins[1:-1] + bins[2:]) / 2, device=array.device, dtype=array.dtype)
    b = b.view(1, -1, 1, 1)
    if property != 'dist':
        probs = array[:, 1:, :, :]
        den = torch.sum(probs, dim=1, keepdim=True)
        j = torch.where(den < 1e-9)
        den[j] = 1e-9
        probs = probs / den
    else:
        probs = array
    if property in ['dist', 'phi']:
        values = b * probs
        values = values.sum(dim=1)
    else:
        s = (torch.sin(b) * probs).sum(dim=1)
        c = (torch.cos(b) * probs).sum(dim=1)
        j = torch.where(s.abs() < 1e-9)
        s[j] = 1e-9
        j = torch.where(c.abs() < 1e-9)
        c[j] = 1e-9
        values = torch.atan2(s, c)

    values = values.masked_fill(~mask2d.bool().squeeze(), np.nan)
    ii, jj = np.diag_indices(values.shape[1])
    for i in range(len(values)):
        values[i, ii, jj] = values[i, ii, jj] + np.nan
    return values


# probably move this into a collate_fn 
class trRosettaPreprocessing():
    """Preprocessing a3m files to torch tensors for trRosetta"""

    def __init__(self, input_token_order, wmin=0.8):
        """
        Parameters:
        -----------
        input_token_order : str
            order of your amino acid alphabet

        wmin : float
            sequence identity value cutoff
        """
        if input_token_order == trR_ALPHABET:
            self.ohe_dict = None
        else:
            self.ohe_dict = self._build_ohe_dict(input_token_order)
        self.wmin = wmin
        self.seqlen = 0

    def _build_ohe_dict(self, input_order):
        """Convert your alphabet order to the one trRosetta uses

        Parameters:
        -----------
        input_token_order : str
            order of your amino acid alphabet

        Returns:
        --------
        ohe_dict : dict
            map between your alphabet order and trRosetta order
        """
        trR_order = trR_ALPHABET
        ohe_dict = {}
        for i in input_order:
            if i in trR_order:
                ohe_dict[input_order.index(i)] = trR_order.index(i)
            else:
                ohe_dict[input_order.index(i)] = trR_order.index('-')
        return ohe_dict

    def _convert_ohe(self, seqs):
        """Convert sequence to ohe

        Parameters:
        -----------
        seqs : list
            list of sequence from MSAs

        ohe_dict : dict
            map between your alphabet order and trRosetta order

        Returns:
        --------
        * : torch.Tensor
            one-hot-encodings of sequences, (num_of_seqs, len(seq))
        """

        processed_seqs = []
        for seq in seqs:
            processed_seqs.append([self.ohe_dict[i.item()] for i in seq])
        return torch.Tensor(np.array(processed_seqs)).long()

    def _reweight_py(self, msa1hot, cutoff, eps=1e-9):
        """Scatter one hot encoding

        Parameters:
        -----------
        msa1hot : torch.Tensor
            one hot encoded MSA seqs

        cutoff : float
            sequence identity value cutoff

        eps : float
            margin to prevent divide by 0

        Returns:
        --------
        * : torch.Tensor
            weights for sequence, (1, num_of_seq)
        """
        self.seqlen = msa1hot.size(2)
        id_min = self.seqlen * cutoff
        id_mtx = torch.stack([torch.tensordot(el, el, [[1, 2], [1, 2]]) for el in msa1hot], 0)
        id_mask = id_mtx > id_min
        weights = 1.0 / (id_mask.type_as(msa1hot).sum(-1) + eps)
        return weights

    def _extract_features_1d(self, msa1hot, weights):
        """Get 1d features

        Parameters:
        -----------
        msa1hot : torch.Tensor
            one hot encoded MSA seqs

        weights : torch.Tensor
            weights for sequences

        Returns:
        --------
        f1d : torch.Tensor
            1d features (1, len(seq), 42)
        """
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
        """Get 2d features

        Parameters:
        -----------
        msa1hot : torch.Tensor
            one hot encoded MSA seqs

        weights : torch.Tensor
            weights for sequences

        penalty : float
            penalty for inv. covariance

        Returns:
        --------
        f2d_dca : torch.Tensor
            2d features (1, len(seq), len(seq), 442)
        """
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
        chol = torch.cholesky(cov_reg.squeeze())
        inv_cov = torch.cholesky_inverse(chol).unsqueeze(0)
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
        """Do all preprocessing steps

        Parameters:
        -----------
        x : list
            list of sequences from MSA

        Returns:
        --------
        features : torch.Tensor, (1, 526, len(seq), len(seq))
            input for trRosetta
        """
        if self.ohe_dict is not None:
            x = self._convert_ohe(x).reshape(len(x), -1)
        x = F.one_hot(x, len(trR_ALPHABET)).unsqueeze(0).float()
        # x = self._one_hot_embedding(x, len(trR_ALPHABET))
        w = self._reweight_py(x, self.wmin)
        f1d = self._extract_features_1d(x, w)
        f2d = self._extract_features_2d(x, w)

        left = f1d.unsqueeze(2).repeat(1, 1, self.seqlen, 1)
        right = f1d.unsqueeze(1).repeat(1, self.seqlen, 1, 1)
        features = torch.cat((left, right, f2d), -1)
        features = features.permute(0, 3, 1, 2)
        return features

    def __call__(self, x):
        return self.process(x)


def tf_to_pytorch_weights(model_params, model_id):
    """Generate trRosetta weights for pytorch

    Parameters:
    -----------
    model_params : torch's model self.named_parameters()
        name of param and param

    model_id: str
        pretrained models a, b, c, d and/or e.

    """
    # check to see if previously downloaded weights, if not -> download
    if not os.path.exists(WEIGHTS_DIR):
        os.mkdir(WEIGHTS_DIR)
    tr_src_dir = WEIGHTS_DIR + 'trrosetta_tf_weights/'
    if not os.path.exists(tr_src_dir):
        os.mkdir(tr_src_dir)
    zip_fpath = tr_src_dir + 'model_weights.tar.bz2'
    tf_fpath = tr_src_dir + 'model2019_07/'
    if len(os.listdir(tr_src_dir)) == 0:
        print('grabbing weights from source...')
        import wget
        wget.download('https://files.ipd.uw.edu/pub/trRosetta/model2019_07.tar.bz2', out=zip_fpath)
        model_file = tarfile.open(zip_fpath, mode='r:bz2')
        model_file.extractall(tr_src_dir)
        model_file.close()

    # check to see if converted to pytorch weights yet
    tr_tgt_dir = WEIGHTS_DIR + 'trrosetta_pytorch_weights/'
    if not os.path.exists(tr_tgt_dir):
        os.mkdir(tr_tgt_dir)

    model_path = tr_tgt_dir + model_id + '.pt'

    if not os.path.exists(model_path):
        print('converting model %s weights from tensorflow to pytorch...' % model_id)

        ckpt = tf_fpath + 'model.xa' + model_id
        import tensorflow as tf
        w_vars = tf.train.list_variables(ckpt)  # get weight names

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
                torch_weight_dict[name] = torch.from_numpy(weights_list[weights_idx]).to(torch.float64).permute(3, 2, 0,
                                                                                                                1)
            else:
                torch_weight_dict[name] = torch.from_numpy(weights_list[weights_idx]).to(torch.float64)
            weights_idx += 1

        torch.save(torch_weight_dict, model_path)


def parse_a3m(filename):
    """Load a3m file to list of sequences

    Parameters:
    -----------
    filename : str
        path to a3m file

    Returns:
    --------
    seqs : list
        list of seqs in MSA

    """
    seqs = []
    table = str.maketrans(dict.fromkeys(string.ascii_lowercase))

    # read file line by line
    for line in open(filename, "r"):
        # skip labels
        if line[0] != '>':
            # remove lowercase letters and right whitespaces
            seqs.append(line.rstrip().translate(table))
    return seqs
