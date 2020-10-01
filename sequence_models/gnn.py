# import modules
from __future__ import print_function

import numpy as np
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

######################## UTILS FROM ORIGINAL PAPER ########################

# The following gather functions
def gather_edges(edges, neighbor_idx):
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_nodes_t(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor index [B,K] => Neighbor features[B,K,C]
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, idx_flat)
    return neighbor_features


def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    h_nn = torch.cat([h_neighbors, h_nodes], -1)
    return h_nn


class TransformerLayer(nn.Module):
    """Transformer block"""

    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        """
        Parameters:
        -----------
        num_hidden : int
            hidden dimension

        num_in : int
            input dimension

        num_heads : int
            number of attention heads

        dropout : float
            dropout
        """
        super(TransformerLayer, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)])

        self.attention = NeighborAttention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ 
        Parallel computation of full transformer layer 

        Parameters:
        -----------
        h_V : torch.Tensor, (N, L, in_channels) 
            node features

        h_E : torch.Tensor, (N, L, K_neighbors, in_channels)
            edge features 

        mask_V : torch.Tensor, (N, L)
            masks nodes with unknown features

        mask_attend : torch.Tensor, (N, L, K_neighbors)
            where to attend to

        Returns:
        --------
        h_V : torch.Tensor, (N, L, num_hidden)

        """
        # Self-attention
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V

    def step(self, t, h_V, h_E, mask_V=None, mask_attend=None):
        """ Sequential computation of step t of a transformer layer """
        # Self-attention
        h_V_t = h_V[:,t,:]
        dh_t = self.attention.step(t, h_V, h_E, mask_attend)
        h_V_t = self.norm[0](h_V_t + self.dropout(dh_t))

        # Position-wise feedforward
        dh_t = self.dense(h_V_t)
        h_V_t = self.norm[1](h_V_t + self.dropout(dh_t))

        if mask_V is not None:
            mask_V_t = mask_V[:,t].unsqueeze(-1)
            h_V_t = mask_V_t * h_V_t
        return h_V_t


class MPNNLayer(nn.Module):
    """
    MLP Layer - two layer perceptron, alt. to TransformerLayer
    """

    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):

        """
        Parameters:
        -----------
        num_hidden : int
            hidden dimension

        num_in : int
            in channels

        dropout : float
            dropout

        num_heads : int
            number of attention heads

        scale : int, 
            scaling factor for message

        """
        super(MPNNLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden, eps=1e-6) for _ in range(2)])

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)

        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        """ 
        Parallel computation of full transformer layer 

        Parameters:
        -----------
        h_V : torch.Tensor, (N, L, in_channels) 
            node features

        h_E : torch.Tensor, (N, L, K_neighbors, in_channels)
            edge features 

        mask_V : torch.Tensor, (N, L)
            masks nodes with unknown features

        mask_attend : torch.Tensor, (N, L)
            where to attend to

        Returns:
        --------
        h_V : torch.Tensor, (N, L, num_hidden)

        """

        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2), -1)
        if h_V_expand.dtype == torch.half:
            h_E = h_E.half()
        h_EV = torch.cat([h_V_expand, h_E], -1)

        h_message = self.W3(F.relu(self.W2(F.relu(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        """
        Parameters:
        -----------
        num_hidden : int
            in channel & out channel

        num_ff : int
            hidden dim

        """
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)

    def forward(self, h_V):
        """
        Parameters:
        -----------
        h_V : torch.Tensor, (N, L, in_channels) 
            node features

        Return:
        -------
        h : torch.Tensor, (N, L, out_channels)       
        """

        h = F.relu(self.W_in(h_V))
        h = self.W_out(h)
        return h


class NeighborAttention(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4):
        """
        Parameters:
        -----------
        num_hidden : int
            hidden dimension

        num_in : int
            in channels

        num_heads : int
            number of attention heads

        """
        super(NeighborAttention, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden

        # Self-attention layers: {queries, keys, values, output}
        self.W_Q = nn.Linear(num_hidden, num_hidden, bias=False)
        self.W_K = nn.Linear(num_in, num_hidden, bias=False)
        self.W_V = nn.Linear(num_in, num_hidden, bias=False)
        self.W_O = nn.Linear(num_hidden, num_hidden, bias=False)
        return

    def _masked_softmax(self, attend_logits, mask_attend, dim=-1):
        """ Numerically stable masked softmax """
        negative_inf = np.finfo(np.float32).min
        attend_logits = torch.where(mask_attend > 0, attend_logits, torch.tensor(negative_inf))
        attend = F.softmax(attend_logits, dim)
        attend = mask_attend * attend
        return attend

    def forward(self, h_V, h_E, mask_attend=None):
        """ Self-attention, graph-structured O(Nk)
        Parameters:
        -----------
        h_V : torch.Tensor, (N_batch, N_nodes, N_hidden)
            Node features

        h_E : torch.Tensor, (N_batch, N_nodes, K, N_hidden)
            Neighbor features

        mask_attend : torch.Tensor, (N_batch, N_nodes, K)
            Mask for attention
        
        Returns:
        --------
        h_V : torch.Tensor, (N_batch, N_nodes, N_hidden)
            Node update
        
        """

        # Queries, Keys, Values
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads

        d = int(self.num_hidden / n_heads)
        Q = self.W_Q(h_V).view([n_batch, n_nodes, 1, n_heads, 1, d])
        K = self.W_K(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d, 1])
        V = self.W_V(h_E).view([n_batch, n_nodes, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_nodes, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)
        
        if mask_attend is not None:
            # Masked softmax
            mask = mask_attend.unsqueeze(2).expand(-1,-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask)
        else:
            attend = F.softmax(attend_logits, -1)

        # Attentive reduction
        h_V_update = torch.matmul(attend.unsqueeze(-2), V.transpose(2,3))
        h_V_update = h_V_update.view([n_batch, n_nodes, self.num_hidden])
        h_V_update = self.W_O(h_V_update)
        return h_V_update

    def step(self, t, h_V, h_E, E_idx, mask_attend=None):
        """ Self-attention for a specific time step t

        Args:
            h_V:            Node features           [N_batch, N_nodes, N_hidden]
            h_E:            Neighbor features       [N_batch, N_nodes, K, N_in]
            E_idx:          Neighbor indices        [N_batch, N_nodes, K]
            mask_attend:    Mask for attention      [N_batch, N_nodes, K]
        Returns:
            h_V_t:            Node update
        """
        # Dimensions
        n_batch, n_nodes, n_neighbors = h_E.shape[:3]
        n_heads = self.num_heads
        d = self.num_hidden / n_heads

        # Per time-step tensors
        h_V_t = h_V[:,t,:]
        h_E_t = h_E[:,t,:,:]
        E_idx_t = E_idx[:,t,:]

        # Single time-step
        h_V_neighbors_t = gather_nodes_t(h_V, E_idx_t)
        E_t = torch.cat([h_E_t, h_V_neighbors_t], -1)

        # Queries, Keys, Values
        Q = self.W_Q(h_V_t).view([n_batch, 1, n_heads, 1, d])
        K = self.W_K(E_t).view([n_batch, n_neighbors, n_heads, d, 1])
        V = self.W_V(E_t).view([n_batch, n_neighbors, n_heads, d])

        # Attention with scaled inner product
        attend_logits = torch.matmul(Q, K).view([n_batch, n_neighbors, n_heads]).transpose(-2,-1)
        attend_logits = attend_logits / np.sqrt(d)

        if mask_attend is not None:
            # Masked softmax
            # [N_batch, K] -=> [N_batch, N_heads, K]
            mask_t = mask_attend[:,t,:].unsqueeze(1).expand(-1,n_heads,-1)
            attend = self._masked_softmax(attend_logits, mask_t)
        else:
            attend = F.softmax(attend_logits / np.sqrt(d), -1)

        # Attentive reduction
        h_V_t_update = torch.matmul(attend.unsqueeze(-2), V.transpose(1,2))
        return h_V_t_update


######################## OUR METHODS ########################

def argmax2value(array, bins, symmetric=False):
    processed = np.zeros(array.shape)
    if symmetric:
        for i in range(len(array)):
            for j in range(i+1):
                argmax_val = array[i,j]
                if argmax_val != 0:
                    argmax_val = np.random.uniform(bins[argmax_val], bins[argmax_val+1])
                else:
                    argmax_val = bins[argmax_val]
                    
                processed[i,j] = argmax_val
                processed[j,i] = argmax_val
                
    else:  
        for i in range(len(array)):
            for j in range(len(array)):
                argmax_val = array[i,j]
                if argmax_val != 0:
                    argmax_val = np.random.uniform(bins[argmax_val], bins[argmax_val+1])
                else:
                    bins[argmax_val]
                    
                processed[i,j] = argmax_val
                
    return processed

def load_npz(path):
    """
    Preprocess trRosetta output
    """
    data = np.load(path)
    
    dist = data['0']
    theta = data['1']
    phi = data['2']
    omega = data['3']

    # bins from trRosetta paper
    dist_bins = np.concatenate([np.array([np.nan]), np.linspace(2,20,37)])
    theta_bins = np.concatenate([np.array([np.nan]), np.linspace(0,360, 25)])
    phi_bins = np.concatenate([np.array([np.nan]), np.linspace(0,180, 13)])
    omega_bins = np.concatenate([np.array([np.nan]), np.linspace(0,360, 25)])
    
    # process
    dist = argmax2value(dist, dist_bins, symmetric=True)
    theta = argmax2value(theta, theta_bins, symmetric=False)
    phi = argmax2value(phi, phi_bins, symmetric=False)
    omega = argmax2value(omega, omega_bins, symmetric=True)
        
    return dist, omega, theta, phi
    

def get_node_features(omega, theta, phi):
    """
    Extract node features
    
    Parameters:
    -----------
    omega : torch.Tensor, (L, L)
        omega angles

    theta : torch.Tensor, (L, L)
        theta angles

    phi : torch.Tensor, (L, L)
        phi angles

    Returns:
    --------
    * : torch.Tensor, (L, 10)
        {sin, cos}×(ωi, φi, φ_ri, ψi, ψ_ri). 
    """

    # omega is symmetric, n1 is omega angle relative to prior
    n1 = torch.cat((torch.tensor([0.]), torch.diagonal(omega, offset=1)))
    
    # theta is asymmetric, n2 and n3 relative to prior
    n2 = torch.cat((torch.diagonal(theta, offset=1), torch.tensor([0.])))
    n3 = torch.cat((torch.tensor([0.]), torch.diagonal(theta, offset=-1)))
    
    # phi is asymmetric n4 and n5 relative to prior
    n4 = torch.cat((torch.diagonal(phi, offset=1), torch.tensor([0.])))
    n5 = torch.cat((torch.tensor([0.]), torch.diagonal(phi, offset=-1)))
    
    ns = torch.stack([n1, n2, n3, n4, n5], dim=1)
    
    # maybe add secondary structure
    return torch.cat([torch.sin(ns), torch.cos(ns)], dim=1)
    

def get_k_neighbors_idx(array, k):
    """
    Find k nearest neighbors for a node.

    Parameters:
    -----------
    array : torch.Tensor, (L, )
        distance matrix

    k : int
        number of neighbors  

    Returns:
    -------- 
    * : torch.Tensor, (1,k)
        Tensor with indices of nearest neighbors for node
    """
    if k > len(array):
        return torch.arange(len(array))
    return torch.topk(array, k, largest=False)[1]


def get_k_neighbors(dist, k):
    """
    Find k nearest neighbors for all nodes in a sequence

    Parameters:
    -----------
    dist : torch.Tensor, (L,L) 
        distance matrix

    k : int
        number of neighbors

    Returns:
    --------
    * : torch.Tensor, (L, k)
        Tensor with indices of nearest neighbors for node
    """
    E_idx = []
    if not isinstance(dist, int):
        ell = len(dist)
        k = min(k, ell)
        for i in range(len(dist)):
            val, idx = torch.topk(dist[i], len(dist), largest=False)
            idx_temp = torch.where(torch.isnan(val)*1 == 0)[0]
            E_idx_temp = idx[idx_temp]
            if len(E_idx_temp) < k:
                rem = k - len(E_idx_temp)
                closest = get_k_neighbors_idx(torch.abs(torch.Tensor(list(range(len(dist))))-i),
                    len(dist))[1:]
                closest = torch.tensor([i.item() for i in closest if i not in E_idx_temp][:rem]).long()
                E_idx_temp = torch.cat([E_idx_temp, closest])
            else:
                E_idx_temp = E_idx_temp[:k]
            E_idx.append(E_idx_temp)
    else:
        for i in range(dist):
            E_idx.append(get_k_neighbors_idx(torch.abs(torch.Tensor(list(range(dist)))-i), k+1)[1:])
    return torch.stack(E_idx)


def get_edge_features(dist, omega, theta, phi, E_idx):
    """
    Get edge features based on k neighbors
    
    Parameters:
    -----------
    dist : torch.Tensor (L, L)
        distance matrix

    omega : torch.Tensor, (L, L)
        omega angles

    theta : torch.Tensor, (L, L)
        theta angles

    phi : torch.Tensor, (L, L)
        phi angles

    connections : torch.Tensor (L, k_neighbors)
        indicies of k nearest neighbors of each node

    Returns:
    --------
    * : torch.Tensor, (L, k_neighbors, 6)
        Edge features 

    """

    dist_E = []
    omega_E = []
    theta_E = []
    theta_Er = []
    phi_E = []
    phi_Er = []
    
    for i in range(len(E_idx)):
        dist_E.append(dist[i, E_idx[i]])
        omega_E.append(omega[i, E_idx[i]])
        theta_E.append(theta[i, E_idx[i]])
        theta_Er.append(theta[E_idx[i], i])
        phi_E.append(phi[i, E_idx[i]])
        phi_Er.append(phi[E_idx[i], i])
        
    dist_E = torch.stack(dist_E)
    omega_E = torch.stack(omega_E)
    theta_E = torch.stack(theta_E)
    theta_Er = torch.stack(theta_Er)
    phi_E = torch.stack(phi_E)
    phi_Er = torch.stack(phi_Er)
    
    E = [dist_E, omega_E, theta_E, theta_Er, phi_E, phi_Er]
    return torch.stack(E, dim=2)


def get_mask(E):
    """
    Get mask to hide node with missing features

    Parameters:
    -----------
    edges : torch.Tensor, (L, k_neighbors, 6)
        edge features

    Returns:
    --------
    * : torch.Tensor, (L)
        mask to hide nodes with missing features
    """
    mask_E = torch.tensor(np.isfinite(np.sum(np.array(E), -1)).astype(np.float32)).view(E.shape[0], E.shape[1], 1)
    # mask_V = torch.tensor(np.isfinite(np.sum(np.array(nodes),-1)).astype(np.float32)).view(1,-1)
    # return mask_V, edge_mask
    return mask_E


def replace_nan(E):
    """
    Replace missing features with 0

    Parameters:
    -----------
    edges : torch.Tensor, (L, k_neighbors, 6)
        Edge features

    Returns:
    --------
    edges : torch.Tensor, (L, k_neighbors, 6)
        Edge features with imputed missing data
    """
    isnan = np.isnan(E).bool()
    E[isnan] = 0.
    return E


class Struct2SeqDecoder(nn.Module):
    """
    Decoder layers from "Generative models for graph-based protein design"
    Ingraham, J., et al : https://github.com/jingraham/neurips19-graph-protein-design/
    
    Decoder architecture:
        Input -> node features, edge features, k_neighbors, sequence if possible,
            otherwise if no structure automatically generate zero_like 
            tensor to replace node and edge features
        Embed features and sequence -> concat features according to k_neighbors
        Pass through TransformerLayer or MPNNLayer layers
        Pass through final output layer to predict residue 

    Example:
        # preprocessing
        # load in features dist, omega, theta, and phi
        nodes = get_node_features(omega, theta, phi)
        connections = get_k_neighbors(dist, 10)
        edges = get_edge_features(dist, omega, theta, phi, connections)
        mask = get_mask(edges)
        edges = replace_nan(edges)
        L = len(seq)
        src = get_S_enc(seq, tokenizer)

        model = Struct2SeqDecoder(num_letters=20, node_features=10,
                    edge_features=6, hidden_dim=16)

        outputs = model(nodes, edges, connections, src, L, mask)

    """
    def __init__(self, num_letters, node_features, edge_features,
                 hidden_dim, num_decoder_layers=3, dropout=0.1, use_mpnn=False):
        
        """
        Parameters:
        -----------
        num_letters : int
            len of protein alphabet

        node_features : int
            number of node features

        edge_features : int
            number of edge features

        hidden_dim : int
            hidden dim

        num_encoder_layers : int
            number of encoder layers
        
        num_decoder_layers : int
            number of decoder layers

        dropout : float
            dropout

        foward_attention_decoder : bool
            if True, use foward attention on encoder embeddings in decoder

        use_mpnn : bool
            if True, use MPNNLayer instead of TransformerLayer

        """

        super(Struct2SeqDecoder, self).__init__()

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim

        # Embedding layers
        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(num_letters, hidden_dim)
        layer = TransformerLayer if not use_mpnn else MPNNLayer

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            layer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)

        # Initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self, connections):
        N_nodes = connections.size(1)
        ii = torch.arange(N_nodes, device=connections.device)
        ii = ii.view((1, -1, 1))
        mask = connections - ii < 0
        mask = mask.type(torch.float32)
        return mask

    def forward(self, nodes, edges, connections, src, edge_mask):
        """
        Parameters:
        -----------
        nodes : torch.Tensor, (N_batch, L, in_channels)
            Node features

        edges : torch.Tensor, (N_batch, L, K_neighbors, in_channels)
            Edge features 

        connections : torch.Tensor, (N_batch, L, K_neighbors)
            Node neighbors 

        src : torch.Tensor, (N_batch, L)
            One-hot-encoded sequences

        L : array-like, (N_batch)
            Lengths of sequences

        edge_mask : torch.Tensor, (N_batch, L, k_neighbors)
            Mask to hide nodes with missing features

        Returns:
        --------
        log_probs : torch.Tensor, (N_batch, L, num_letters)
            Log probs of residue predictions 
        """
        
        # Prepare node and edge embeddings
        h_V = self.W_v(nodes)
        h_E = self.W_e(edges)

        # Concatenate sequence embeddings for autoregressive decoder
        h_S = self.W_s(src)
        h_ES = cat_neighbors_nodes(h_S, h_E, connections)

        # Build encoder embeddings
        h_ES_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, connections)
        h_ESV_encoder = cat_neighbors_nodes(h_V, h_ES_encoder, connections)

        # Decoder uses masked self-attention
        """
        mask_attend : autoregressive mask 
        mask : input mask, to mask the nodes/edges with nan values 
        mask_bw : applies both masks together
        """
        mask_attend = self._autoregressive_mask(connections).unsqueeze(-1)
        # mask_1D = mask_V.view([mask_V.size(0), mask_V.size(1), 1, 1])
        # mask_bw = mask_1D * edge_mask * mask_attend
        mask_bw = edge_mask * mask_attend
        
        # mask_fw = mask_1D * (1. - mask_attend)
        mask_fw = edge_mask * (1. - mask_attend)
        h_ESV_encoder_fw = mask_fw * h_ESV_encoder
        
        for layer in self.decoder_layers:
            # Masked positions attend to encoder information, unmasked see. 
            h_ESV = cat_neighbors_nodes(h_V, h_ES, connections)
            h_ESV = mask_bw * h_ESV + h_ESV_encoder_fw
            h_V = layer(h_V, h_ESV, mask_V=None)
        
        logits = self.W_out(h_V) 
        return logits

    
