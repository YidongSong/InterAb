# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel,AutoModel,RoFormerModel
from torch_geometric.nn import TransformerConv
from torch_scatter import scatter_mean,scatter_add
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pickle
import pdb
import sys
from torch.autograd import Variable
import os
from tfold.modules.af3_smod.utils import AdaptiveLayerNorm
from tfold.modules.af3_smod.utils import swish
from tfold.modules.attention import WindowAttention
from tfold.modules.af3_smod.utils import AdaptiveLayerNorm


class AtomInter(nn.Module):
    """
    AtomInter is an all-atom modeling approach designed to extract fine-grained structural features from antibody-antigen complexes.
    """
    def __init__(
        self,
        n_dims_atom_inputs=37,
        n_dims_atom=128,
        n_dims_atompair=16,
        atoms_per_window=27,
        n_dims_token=384,
        n_dims_sfea=384,
        n_dims_pfea=128,
        atom_module_blocks=3,
        atom_module_heads=4,
        atom_module_kwargs: dict = dict(),
    ):
        super().__init__()
        self.atoms_per_window = atoms_per_window
        self.atom_single_conditioning = nn.Linear(n_dims_atom_inputs, n_dims_atom, bias=False)
        self.valid_mask_embedding = nn.Linear(1, n_dims_atompair, bias=False)
        self.offset_embedding = nn.Linear(3, n_dims_atompair, bias=False)
        self.inverse_squared_distances_embedding = nn.Linear(1, n_dims_atompair, bias=False)
        self.single_to_atom_feat_cond = nn.Sequential(
            nn.LayerNorm(n_dims_sfea),
            nn.Linear(n_dims_sfea, n_dims_atom, bias=False)
        )
        self.pairwise_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(n_dims_pfea),
            nn.Linear(n_dims_pfea, n_dims_atompair, bias=False)
        )
        self.atom_pos_to_atom_feat = nn.Linear(3, n_dims_atom, bias=False)
        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(n_dims_atom),
            nn.ReLU(),
            nn.Linear(n_dims_atom, n_dims_atompair * 2, bias=False),
        )
        self.atompair_feats_mlp = nn.Sequential(
            nn.Linear(n_dims_atompair, n_dims_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(n_dims_atompair, n_dims_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(n_dims_atompair, n_dims_atompair, bias=False),
        )
        self.atom_transformer = DiffusionModule(
            n_lyrs=atom_module_blocks,
            n_heads=atom_module_heads,
            dim=n_dims_atom,
            n_dims_cond=n_dims_atom,
            n_dims_pfea=n_dims_atompair,
            attn_window_size=atoms_per_window,
            **atom_module_kwargs
        )
        self.atom_feats_to_pooled_token = AtomPooler(
            dim=n_dims_atom,
            dim_out=n_dims_token
        )
    
    def enable_activation_checkpoint(self, enabled=True):
        self.atom_transformer.enable_activation_checkpoint(enabled)

    def forward(self, atom_inputs, atom_mask=None):
        """Perform the forward pass.

        Args:
            * atom_inputs:
              - atom_feats  B x M x c
              - atom_ref_pos B x M x 3
              - atom_ref_space_uid B x M
              - molecule_atom_lens B x N
            * nfea_tns: noise coordinate, B x M x 3
            * sfea_tns: sequence representation: B x N x c_s
            * pfea_tns: pairwise representation: B x N x N x c_z

        Returns:
            tfea_tns: token representation B x N x c_t
            afea_tns: atom representation B x M x c_a
            atom_feat_cond: B x M x c_a
            atompair_feat_cond: B x M x M x c_p
        """

        dtype = next(self.parameters()).dtype
        atom_feat_cond = self.atom_single_conditioning(atom_inputs['atom_feats'].to(dtype))
        atom_ref_space_uid = atom_inputs['atom_ref_space_uid']
        same_ref_space_mask = (atom_ref_space_uid.unsqueeze(1) == atom_ref_space_uid.unsqueeze(2)).to(dtype)
        atom_ref_pos = atom_inputs['atom_ref_pos'].to(atom_feat_cond.dtype)
        pairwise_rel_pos = atom_ref_pos.unsqueeze(1) - atom_ref_pos.unsqueeze(2)
        atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim=-1, p=2) ** 2) ** -1
        atompair_feat_cond = self.offset_embedding(pairwise_rel_pos) * same_ref_space_mask.unsqueeze(-1)
        atompair_feat_cond += self.inverse_squared_distances_embedding(
            atom_inv_square_dist.unsqueeze(-1)) * same_ref_space_mask.unsqueeze(-1)
        atompair_feat_cond += self.valid_mask_embedding(same_ref_space_mask.unsqueeze(-1))
        afea_tns = atom_feat_cond
        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feat_cond)
        
        atom_repr_cond_r, atom_repr_cond_c = atom_repr_cond.chunk(2, dim=-1)
        atompair_feat_cond = atompair_feat_cond + atom_repr_cond_r.unsqueeze(-2) + atom_repr_cond_c.unsqueeze(-3)
        atompair_feat_cond = self.atompair_feats_mlp(atompair_feat_cond) + atompair_feat_cond
        afea_tns = self.atom_transformer(afea_tns, atom_feat_cond, atompair_feat_cond, mask=atom_mask)
        sfea_tns = self.atom_feats_to_pooled_token(
            atom_feats=afea_tns,
            molecule_atom_lens=atom_inputs['molecule_atom_lens']
        )

        return sfea_tns

class AtomPooler(nn.Module):
    """Convert atomic-level features into token-level features"""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out
        self.proj = nn.Sequential(
            nn.Linear(dim, dim_out, bias=False),
            nn.ReLU()
        )

    def forward(self, atom_feats, molecule_atom_lens):
        """Perform the forward pass.

        Args:
        atom_feats: B x M x c
        molecule_atom_lens: B x N

        Returns:
        token_feats: B x N x c
        """
        atom_feats = self.proj(atom_feats)
        atom_len = atom_feats.shape[1]
        token_mask = molecule_atom_lens > 0
        cumsum_feats = atom_feats.cumsum(dim=1)
        cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value=0)
        cumsum_indices = molecule_atom_lens.cumsum(dim=1)
        cumsum_indices = F.pad(cumsum_indices, (1, 0), value=0)
        cumsum_indices = repeat(cumsum_indices, 'b n -> b n d', d=cumsum_feats.shape[-1])
        sel_cumsum = cumsum_feats.gather(-2, cumsum_indices.to(torch.int64))

        # subtract cumsum at one index from the previous one
        summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]
        token_feats = summed / molecule_atom_lens.clamp(min=1).unsqueeze(-1)
        token_feats = torch.where(token_mask.unsqueeze(-1), token_feats, torch.zeros_like(token_feats))

        return token_feats

class Transition(nn.Module):
    def __init__(
        self,
        dim,
        dim_cond,
        expansion_factor=2
    ):
        super().__init__()
        self.adaptive_norm = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.linear_1 = nn.Linear(dim, dim * expansion_factor, bias=False)
        self.linear_2 = nn.Linear(dim, dim * expansion_factor, bias=False)
        self.linear_3 = nn.Linear(dim * expansion_factor, dim, bias=False)
        adaln_zero_gamma_linear = nn.Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, -2)
        self.to_adaln_zero_gamma = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    def forward(
        self,
        a,
        cond,
        **kwargs
    ):
        a = self.adaptive_norm(a, cond=cond)
        b = swish(self.linear_1(a)) * self.linear_2(a)
        a = self.to_adaln_zero_gamma(cond) * self.linear_3(b)

        return a

class DiffusionBlock(nn.Module):
    def __init__(
        self,
        n_heads,
        dim=384,
        n_dims_cond=None,
        n_dims_pfea=128,
        attn_window_size=None,
        attn_pair_bias_kwargs: dict = dict(),
    ):
        super().__init__()

        # InterAttention
        self.pair_bias_attn = InterAttention(
            dim=dim,
            n_dims_sfea=n_dims_cond,
            n_dims_pfea=n_dims_pfea,
            n_heads=n_heads,
            window_size=attn_window_size,
            **attn_pair_bias_kwargs
        )

        # Transition
        self.condition_trans = Transition(
            dim=dim,
            dim_cond=n_dims_cond,
        )

    def forward(self, nfea_tns, sfea_tns, pfea_tns, mask=None):
        attn_out = self.pair_bias_attn(nfea_tns, sfea_tns=sfea_tns, pfea_tns=pfea_tns, mask=mask)
        ff_out = self.condition_trans(nfea_tns, cond=sfea_tns)
        nfea_tns = attn_out + ff_out + nfea_tns

        return nfea_tns

class DiffusionModule(nn.Module):

    def __init__(
        self,
        n_lyrs,
        n_heads,
        dim=384,
        n_dims_cond=None,
        n_dims_pfea=128,
        attn_window_size=None,
        attn_pair_bias_kwargs: dict = dict(),
        activation_checkpoint_fn=None
    ):
        super().__init__()
        n_dims_cond = n_dims_cond if n_dims_cond is not None else dim
        self.n_lyrs = n_lyrs

        if activation_checkpoint_fn is None:
            self.activation_checkpoint_fn = torch.utils.checkpoint.checkpoint

        self.activation_checkpoint = False
        self.blocks = nn.ModuleList([
            DiffusionBlock(
                n_heads,
                dim,
                n_dims_cond,
                n_dims_pfea,
                attn_window_size,
                attn_pair_bias_kwargs,
            )
            for _ in range(self.n_lyrs)
        ])

    def enable_activation_checkpoint(self, enabled=True):
        self.activation_checkpoint = enabled

    def forward(
        self,
        nfea_tns,
        sfea_tns,
        pfea_tns,
        mask=None,  # [B, L]
    ):

        for block in self.blocks:
            if self.activation_checkpoint:
                nfea_tns = self.activation_checkpoint_fn(block, nfea_tns, sfea_tns, pfea_tns, mask)
            else:
                nfea_tns = block(nfea_tns, sfea_tns, pfea_tns, mask)
        return nfea_tns

class InterAttention(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        n_dims_sfea,
        n_dims_pfea,
        window_size=None,
        num_memory_kv=0,
        **attn_kwargs
    ):
        super().__init__()

        self.window_size = window_size
        self.attn = WindowAttention(
            c_q=dim,
            c_k=dim,
            c_v=dim,
            window_size=window_size,
            num_heads=n_heads,
            **attn_kwargs
        )
        self.adaptive_norm = AdaptiveLayerNorm(dim=dim, dim_cond=n_dims_sfea)
        to_attn_bias_linear = nn.Linear(n_dims_pfea, n_heads, bias=False)
        nn.init.zeros_(to_attn_bias_linear.weight)
        self.to_attn_bias = nn.Sequential(
            nn.LayerNorm(n_dims_pfea),
            to_attn_bias_linear,
            Rearrange('b ... h -> b h ...')
        )
        adaln_zero_gamma_linear = nn.Linear(n_dims_sfea, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, -2)
        self.to_out = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    def forward(
        self,
        nfea_tns,
        sfea_tns,
        pfea_tns,
        mask=None,   # [*, L]
        attn_bias=None,
        **kwargs
    ):

        # attention preparation with further addition from pairwise repr
        if attn_bias is not None:
            attn_bias = rearrange(attn_bias, 'b ... -> b 1 ...')
        else:
            attn_bias = 0.
        nfea_tns = self.adaptive_norm(nfea_tns, sfea_tns)
        attn_bias = self.to_attn_bias(pfea_tns) + attn_bias
        if mask is not None:
            mask = (mask[..., None] * mask[..., None, :])[..., None, :, :]

        # Attention
        nfea_tns = self.attn(nfea_tns, attn_mask=mask, attn_bias=attn_bias, **kwargs)
        nfea_tns = self.to_out(sfea_tns) * nfea_tns

        return nfea_tns

def _positional_embeddings(edge_index, num_embeddings=16):
        d = edge_index[0] - edge_index[1]
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        PE = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return PE

def _get_angle(X, eps=1e-7):
    # psi, omega, phi
    X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
    dX = X[1:] - X[:-1]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:-2]
    u_1 = U[1:-1]
    u_0 = U[2:]

    # Backbone normals
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    # Angle between normals
    cosD = torch.sum(n_2 * n_1, -1)
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)
    D = F.pad(D, [1, 2]) # This scheme will remove phi[0], psi[-1], omega[-1]
    D = torch.reshape(D, [-1, 3])
    dihedral = torch.cat([torch.cos(D), torch.sin(D)], 1)

    # alpha, beta, gamma
    cosD = (u_2 * u_1).sum(-1) # alpha_{i}, gamma_{i}, beta_{i+1}
    cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
    D = torch.acos(cosD)
    D = F.pad(D, [1, 2])
    D = torch.reshape(D, [-1, 3])
    bond_angles = torch.cat((torch.cos(D), torch.sin(D)), 1)

    node_angles = torch.cat((dihedral, bond_angles), 1)
    return node_angles # dim = 12

def _rbf(D, D_min=0., D_max=20., D_count=16):
    '''
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=D.device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _get_distance(X, edge_index):
    atom_N = X[:,0]  # [L, 3]
    atom_Ca = X[:,1]
    atom_C = X[:,2]
    atom_O = X[:,3]
    atom_R = X[:,4]
    node_list = ['Ca-N', 'Ca-C', 'Ca-O', 'N-C', 'N-O', 'O-C', 'R-N', 'R-Ca', "R-C", 'R-O']
    node_dist = []
    for pair in node_list:
        atom1, atom2 = pair.split('-')
        E_vectors = vars()['atom_' + atom1] - vars()['atom_' + atom2]
        rbf = _rbf(E_vectors.norm(dim=-1))
        node_dist.append(rbf)
    node_dist = torch.cat(node_dist, dim=-1) # dim = [N, 10 * 16]

    atom_list = ["N", "Ca", "C", "O", "R"]
    edge_dist = []
    for atom1 in atom_list:
        for atom2 in atom_list:
            try:
                E_vectors = vars()['atom_' + atom1][edge_index[0]] - vars()['atom_' + atom2][edge_index[1]]
                rbf = _rbf(E_vectors.norm(dim=-1))
                edge_dist.append(rbf)
            except:
                pdb.set_trace()
    edge_dist = torch.cat(edge_dist, dim=-1) # dim = [E, 25 * 16]

    return node_dist, edge_dist

def _quaternions(R):
    """ Convert a batch of 3D rotations [R] to quaternions [Q]
        R [N,3,3]
        Q [N,4]
    """
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
          Rxx - Ryy - Rzz,
        - Rxx + Ryy - Rzz,
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    # The relu enforces a non-negative trace
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    Q = F.normalize(Q, dim=-1)

    return Q

def _get_direction_orientation(X, edge_index): # N, CA, C, O, R
    X_N = X[:,0]  # [L, 3]
    X_Ca = X[:,1]
    X_C = X[:,2]
    u = F.normalize(X_Ca - X_N, dim=-1)
    v = F.normalize(X_C - X_Ca, dim=-1)
    b = F.normalize(u - v, dim=-1)
    n = F.normalize(torch.cross(u, v), dim=-1)
    local_frame = torch.stack([b, n, torch.cross(b, n)], dim=-1) # [L, 3, 3] (3 column vectors)
    node_j, node_i = edge_index
    t = F.normalize(X[:, [0,2,3,4]] - X_Ca.unsqueeze(1), dim=-1) # [L, 4, 3]
    try:
        node_direction = torch.matmul(t, local_frame).reshape(t.shape[0], -1) # [L, 4 * 3]
    except Exception as ex:
        print(t.size())
        print(local_frame.size())
        print('except')
        print(ex)

    t = F.normalize(X[node_j] - X_Ca[node_i].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ji = torch.matmul(t, local_frame[node_i]).reshape(t.shape[0], -1) # [E, 5 * 3]
    t = F.normalize(X[node_i] - X_Ca[node_j].unsqueeze(1), dim=-1) # [E, 5, 3]
    edge_direction_ij = torch.matmul(t, local_frame[node_j]).reshape(t.shape[0], -1) # [E, 5 * 3]
    edge_direction = torch.cat([edge_direction_ji, edge_direction_ij], dim = -1) # [E, 2 * 5 * 3]

    r = torch.matmul(local_frame[node_i].transpose(-1,-2), local_frame[node_j]) # [E, 3, 3]
    edge_orientation = _quaternions(r) # [E, 4]

    return node_direction, edge_direction, edge_orientation

def get_geo_feat(X, edge_index):
    pos_embeddings = _positional_embeddings(edge_index)
    node_angles = _get_angle(X)
    node_dist, edge_dist = _get_distance(X, edge_index)
    node_direction, edge_direction, edge_orientation = _get_direction_orientation(X, edge_index)

    geo_node_feat = torch.cat([node_angles, node_dist, node_direction], dim=-1)
    geo_edge_feat = torch.cat([pos_embeddings, edge_orientation, edge_dist, edge_direction], dim=-1)

    return geo_node_feat, geo_edge_feat

def padding_ver1(x, batch_id, feature_dim):
        batch_size = max(batch_id) + 1
        max_len= max(torch.unique(batch_id,return_counts=True)[1])
        batch_data = torch.zeros([batch_size,max_len,feature_dim])
        mask = torch.zeros([batch_size,max_len])
        len_0 = 0
        len_1 = 0
        for i in range(batch_size):
            len_1 = len_0 + torch.unique(batch_id,return_counts=True)[1][i]
            batch_data[i][:torch.unique(batch_id,return_counts=True)[1][i]] = x[len_0:len_1]
            mask[i][:torch.unique(batch_id,return_counts=True)[1][i]] = 1
            len_0 += torch.unique(batch_id,return_counts=True)[1][i]
        return batch_data, mask

class Context(nn.Module):
    def __init__(self, num_hidden):
        super(Context, self).__init__()

        self.V_MLP_g = nn.Sequential(
                                nn.Linear(num_hidden,num_hidden),
                                nn.ReLU(),
                                nn.Linear(num_hidden,num_hidden),
                                nn.Sigmoid()
                                )

    def forward(self, h_V, batch_id):
        c_V = scatter_mean(h_V, batch_id, dim=0)
        h_V = h_V * self.V_MLP_g(c_V[batch_id])
        return h_V

class EdgeMLP(nn.Module):
    def __init__(self, num_hidden, dropout=0.2):
        super(EdgeMLP, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.BatchNorm1d(num_hidden)
        self.W11 = nn.Linear(3*num_hidden, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()

    def forward(self, h_V, edge_index, h_E):
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        h_EV = torch.cat([h_V[src_idx], h_E, h_V[dst_idx]], dim=-1)
        h_message = self.W12(self.act(self.W11(h_EV)))
        h_E = self.norm(h_E + self.dropout(h_message))
        return h_E
    
class GNNLayer(nn.Module):
    def __init__(self, num_hidden, dropout=0.2, num_heads=4):
        super(GNNLayer, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([nn.LayerNorm(num_hidden) for _ in range(2)])

        self.attention = TransformerConv(in_channels=num_hidden, out_channels=int(num_hidden / num_heads), heads=num_heads, dropout = dropout, edge_dim = num_hidden, root_weight=False)
        self.PositionWiseFeedForward = nn.Sequential(
            nn.Linear(num_hidden, num_hidden*4),
            nn.ReLU(),
            nn.Linear(num_hidden*4, num_hidden)
        )
        self.edge_update = EdgeMLP(num_hidden, dropout)
        self.context = Context(num_hidden)

    def forward(self, h_V, edge_index, h_E, batch_id):
        dh = self.attention(h_V, edge_index, h_E)
        h_V = self.norm[0](h_V + self.dropout(dh))

        # Position-wise feedforward
        dh = self.PositionWiseFeedForward(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        # update edge
        h_E = self.edge_update(h_V, edge_index, h_E)

        # context node update
        h_V = self.context(h_V, batch_id)

        return h_V, h_E
    
class Graph_encoder(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim,
                 seq_in=False, num_layers=4, drop_rate=0.2):
        super(Graph_encoder, self).__init__()

        self.seq_in = seq_in
        if self.seq_in:
            self.W_s = nn.Embedding(20, 20)
            node_in_dim += 20
        
        self.node_embedding = nn.Linear(node_in_dim, hidden_dim, bias=True)
        self.edge_embedding = nn.Linear(edge_in_dim, hidden_dim, bias=True)
        self.norm_nodes = nn.BatchNorm1d(hidden_dim)
        self.norm_edges = nn.BatchNorm1d(hidden_dim)
        
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.W_e = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.layers = nn.ModuleList(
                GNNLayer(num_hidden=hidden_dim, dropout=drop_rate, num_heads=4)
            for _ in range(num_layers))


    def forward(self, h_V, edge_index, h_E, seq, batch_id):
        if self.seq_in and seq is not None:
            seq = self.W_s(seq)
            h_V = torch.cat([h_V, seq], dim=-1)
        
        h_V = self.W_v(self.norm_nodes(self.node_embedding(h_V)))
        h_E = self.W_e(self.norm_edges(self.edge_embedding(h_E)))

        for layer in self.layers:
            h_V, h_E = layer(h_V, edge_index, h_E, batch_id)
        
        return h_V

class AvgPoolOfExperts(nn.Module):
    """Compute joint posterior from single-modality posteriors
    using 1D average pooling.
    """

    def __init__(self):
        super(AvgPoolOfExperts, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)

    def forward(self, mu, logvar):
        """
        :param mu: M x B x D for M experts, batch size B, dimension D
        :param logvar: M x B x D for M experts, batch size B, dimension D
        """
        mu = mu.transpose(0, 1)
        logvar = logvar.transpose(0, 1)
        mu = self.pool(mu.transpose(1, 2)).squeeze()
        logvar = self.pool(logvar.transpose(1, 2)).squeeze()

        return mu, logvar

class MixtureOfExperts(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(MixtureOfExperts, self).__init__()
        self.experts = nn.ModuleList([Classifier(input_dim, 1) for i in range(num_experts)])
        self.gating_network = nn.Sequential(nn.Linear(input_dim, num_experts), nn.Softmax(dim=1))

    def forward(self, x):
        if x.dim() == 1:
                x = x.unsqueeze(0)
        expert_outputs = [expert(x) for expert in self.experts]
        expert_outputs = torch.stack(expert_outputs, dim=1)
        
        gating_weights = self.gating_network(x)
        final_output = torch.sum(expert_outputs * gating_weights.unsqueeze(2), dim=1)
        return final_output
    
class Classifier(nn.Module):
    """ Classifier/decoder.
    Models p(y|z).
    """
    _DROPOUT = 0.3
    def __init__(self, z_dim, out_dim, drop: int = _DROPOUT, layer_norm: bool = False):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, out_dim)

        self.ln1 = nn.LayerNorm(z_dim)
        self.ln2 = nn.LayerNorm(z_dim)
        self.layer_norm = layer_norm

        self.drop = nn.Dropout(p=drop)

    def forward(self, x):
        if self.layer_norm:
            h = self.drop(F.relu(self.ln1(self.fc1(x))))
            h = self.drop(F.relu(self.ln2(self.fc2(h))))
        else:
            h = self.drop(F.relu(self.fc1(x)))
            h = self.drop(F.relu(self.fc2(h)))

        h = self.fc3(h)
        return h
      
class InterAb(nn.Module):
    def __init__(self, heavy_dir,light_dir, antigen_dir, emb_dim=256):
        super().__init__()
        self.HeavyModel = AutoModel.from_pretrained(heavy_dir, output_hidden_states=True, return_dict=True)
        self.LightModel = AutoModel.from_pretrained(light_dir, output_hidden_states=True, return_dict=True)
        self.AntigenModel = AutoModel.from_pretrained(antigen_dir, output_hidden_states=True, return_dict=True,cache_dir = "./ESM_models")

        self.cnn1 = CNN_Module(in_channel= 256) # 256
        self.cnn2 = CNN_Module(in_channel = 256) # 256
        self.cnn3 = CNN_Module(in_channel = 700,hidden_size=76) # 700

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=128 * 4 + 150, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=emb_dim),
            nn.ReLU(),
            nn.Linear(in_features=emb_dim, out_features=1)
        )

        self.FC_1 = nn.Sequential(
            nn.Linear(in_features=150*5, out_features=150),
            nn.ReLU(),
            nn.LayerNorm(150, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=150, out_features=32),
            nn.ReLU(),
            nn.LayerNorm(32, eps=1e-6),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=32, out_features=1),
        )
        self.FC_2 = nn.Linear(150 * 5, 150) 
        self.FC_3 = nn.Linear(128 * 3, 128)
        self.LN_1 = nn.LayerNorm(32)
        self.LN_2 = nn.LayerNorm(150)
        self.LN_3 = nn.LayerNorm(128)

        joint_posterior = 'avg_pool'
        self.experts = AvgPoolOfExperts()
        z_dim  =150
        moe_dim = 15
        self.experts_domain = MixtureOfExperts(z_dim, moe_dim)  # 1,5,10,15,20
        self.cnn5 = CNN_Module_ver1(in_channel= 384)
        self.predict = nn.Sequential(
            nn.Linear(in_features=128, out_features=1)
        )
        
        # geometric graph learning
        node_input_dim = 1280+9+184
        edge_input_dim = 450
        hidden_dim = 256
        num_layers = 3
        dropout = 0.1
        attention_heads = 8
        self.ab_h_Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)
        self.ab_l_Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)
        self.ag_Graph_encoder = Graph_encoder(node_in_dim=node_input_dim, edge_in_dim=edge_input_dim, hidden_dim=hidden_dim, seq_in=False, num_layers=num_layers, drop_rate=dropout)
        self.ab_h_ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, attention_heads) # num_heads
                                 )
        self.ab_l_ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, attention_heads) # num_heads
                                 )
        self.ag_ATFC = nn.Sequential(
                                  nn.Linear(hidden_dim, 64)
                                 ,nn.LeakyReLU()
                                 ,nn.LayerNorm(64, eps=1e-6)
                                 ,nn.Linear(64, attention_heads) # num_heads
                                 )
        self.ab_h_output = nn.Sequential(
                                        #  nn.Linear(2*attention_heads*hidden_dim, 2*attention_heads*hidden_dim)
                                        #  ,nn.LeakyReLU()
                                        #  ,nn.LayerNorm(2*attention_heads*hidden_dim, eps=1e-6)
                                        #  ,nn.Dropout(dropout)
                                         nn.Linear(attention_heads*hidden_dim, attention_heads*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(attention_heads*hidden_dim, eps=1e-6)
                                         ,nn.Linear(attention_heads*hidden_dim, (attention_heads//2)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads//2)*hidden_dim, eps=1e-6)
                                         ,nn.Linear((attention_heads//2)*hidden_dim, (attention_heads//4)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads//4)*hidden_dim, eps=1e-6)
                                         ,nn.Linear((attention_heads//4)*hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Linear(hidden_dim, hidden_dim//2)
                                         )
        self.ab_l_output = nn.Sequential(
                                        #  nn.Linear(2*attention_heads*hidden_dim, 2*attention_heads*hidden_dim)
                                        #  ,nn.LeakyReLU()
                                        #  ,nn.LayerNorm(2*attention_heads*hidden_dim, eps=1e-6)
                                        #  ,nn.Dropout(dropout)
                                         nn.Linear(attention_heads*hidden_dim, attention_heads*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(attention_heads*hidden_dim, eps=1e-6)
                                         ,nn.Linear(attention_heads*hidden_dim, (attention_heads//2)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads//2)*hidden_dim, eps=1e-6)
                                         ,nn.Linear((attention_heads//2)*hidden_dim, (attention_heads//4)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads//4)*hidden_dim, eps=1e-6)
                                         ,nn.Linear((attention_heads//4)*hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Linear(hidden_dim, hidden_dim//2)
                                         )
        self.ag_output = nn.Sequential(
                                        #  nn.Linear(2*attention_heads*hidden_dim, 2*attention_heads*hidden_dim)
                                        #  ,nn.LeakyReLU()
                                        #  ,nn.LayerNorm(2*attention_heads*hidden_dim, eps=1e-6)
                                        #  ,nn.Dropout(dropout)
                                         nn.Linear(attention_heads*hidden_dim, attention_heads*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(attention_heads*hidden_dim, eps=1e-6)
                                         ,nn.Linear(attention_heads*hidden_dim, (attention_heads//2)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads//2)*hidden_dim, eps=1e-6)
                                         ,nn.Linear((attention_heads//2)*hidden_dim, (attention_heads//4)*hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm((attention_heads//4)*hidden_dim, eps=1e-6)
                                         ,nn.Linear((attention_heads//4)*hidden_dim, hidden_dim)
                                         ,nn.LeakyReLU()
                                         ,nn.LayerNorm(hidden_dim, eps=1e-6)
                                         ,nn.Linear(hidden_dim, hidden_dim//2)
                                         )
        
        # AtomInter
        self.atom_attn_encoder = AtomInter(n_dims_atom_inputs=42, n_dims_atom=64, n_dims_atompair=16, atoms_per_window=512, n_dims_token=384, n_dims_sfea=384, n_dims_pfea=128, atom_module_blocks=2, atom_module_heads=4)
        self.conv_1 = nn.Conv1d(in_channels=384, out_channels=150, kernel_size=1, padding="same")
        self.conv_3 = nn.Conv1d(in_channels=384, out_channels=150, kernel_size=3, padding="same")
        self.conv_5 = nn.Conv1d(in_channels=384, out_channels=150, kernel_size=5, padding="same")
        self.conv_7 = nn.Conv1d(in_channels=384, out_channels=150, kernel_size=7, padding="same")
        self.conv_9 = nn.Conv1d(in_channels=384, out_channels=150, kernel_size=9, padding="same")
        self.drop = nn.Dropout(p=0.1)
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)


    def forward(self, heavy, light, antigen, data_dict, data_batch, device):
        # Inner-chain sequence module
        heavy_emb = self.HeavyModel(**heavy).last_hidden_state 
        light_emb = self.LightModel(**light).last_hidden_state 
        antigen_emb = self.AntigenModel(**antigen).last_hidden_state 
        heavy_InnerSeq = self.cnn1(heavy_emb)
        light_InnerSeq = self.cnn2(light_emb)
        antigen_InnerSeq = self.cnn3(antigen_emb)

        # Inner-chain structure module
        ab_h_h_V_geo, ab_h_h_E = get_geo_feat(data_dict['ab_h_X'], data_dict['ab_h_edge_index'])
        ab_l_h_V_geo, ab_l_h_E = get_geo_feat(data_dict['ab_l_X'], data_dict['ab_l_edge_index'])
        ag_h_V_geo, ag_h_E = get_geo_feat(data_dict['ag_X'], data_dict['ag_edge_index'])
        ab_h_h_V = torch.cat([data_dict['ab_h_node_feat'], ab_h_h_V_geo], dim=-1)
        ab_l_h_V = torch.cat([data_dict['ab_l_node_feat'], ab_l_h_V_geo], dim=-1)  
        ag_h_V = torch.cat([data_dict['ag_node_feat'], ag_h_V_geo], dim=-1)
        ab_h_h_V = self.ab_h_Graph_encoder(ab_h_h_V.to(device), data_dict['ab_h_edge_index'].to(device), ab_h_h_E.to(device), data_dict['ab_h_seq'].to(device), data_dict['ab_h_batch_id'].to(device)) # [num_residue, hidden_dim]
        ab_l_h_V = self.ab_l_Graph_encoder(ab_l_h_V.to(device), data_dict['ab_l_edge_index'].to(device), ab_l_h_E.to(device), data_dict['ab_l_seq'].to(device), data_dict['ab_l_batch_id'].to(device)) # [num_residue, hidden_dim]
        ag_h_V = self.ag_Graph_encoder(ag_h_V.to(device), data_dict['ag_edge_index'].to(device), ag_h_E.to(device), data_dict['ag_seq'].to(device), data_dict['ag_batch_id'].to(device))
        ab_h_h_V_stru, ab_h_mask_baseline = padding_ver1(ab_h_h_V.cpu(), data_dict['ab_h_batch_id'].cpu(), ab_h_h_V.shape[1])
        ab_l_h_V_stru, ab_l_mask_baseline = padding_ver1(ab_l_h_V.cpu(), data_dict['ab_l_batch_id'].cpu(), ab_l_h_V.shape[1])
        ag_h_V_stru, ag_mask_baseline = padding_ver1(ag_h_V.cpu(), data_dict['ag_batch_id'].cpu(), ag_h_V.shape[1])
        
        ab_h_att = self.ab_h_ATFC(ab_h_h_V_stru.to(device))    
        ab_h_att = ab_h_att.masked_fill(ab_h_mask_baseline[:, :, None].to(device) == 0, -1e9)
        ab_h_att = F.softmax(ab_h_att, dim=1).to(device)

        ab_l_att = self.ab_l_ATFC(ab_l_h_V_stru.to(device))    
        ab_l_att = ab_l_att.masked_fill(ab_l_mask_baseline[:, :, None].to(device) == 0, -1e9)
        ab_l_att = F.softmax(ab_l_att, dim=1).to(device)

        ag_att = self.ag_ATFC(ag_h_V_stru.to(device))    
        ag_att = ag_att.masked_fill(ag_mask_baseline[:, :, None].to(device) == 0, -1e9)
        ag_att = F.softmax(ag_att, dim=1).to(device)
       
        ab_h_att = ab_h_att.transpose(1,2)  
        ab_h_h_V_stru = ab_h_att@ab_h_h_V_stru.to(device)
        ab_h_h_V_stru = torch.flatten(ab_h_h_V_stru, start_dim=1)

        ab_l_att = ab_l_att.transpose(1,2)  
        ab_l_h_V_stru = ab_l_att@ab_l_h_V_stru.to(device)
        ab_l_h_V_stru = torch.flatten(ab_l_h_V_stru, start_dim=1)

        ag_att = ag_att.transpose(1,2)  
        ag_h_V_stru = ag_att@ag_h_V_stru.to(device)
        ag_h_V_stru = torch.flatten(ag_h_V_stru, start_dim=1)

        ab_h_h_V_stru = self.ab_h_output(ab_h_h_V_stru)
        ab_l_h_V_stru = self.ab_l_output(ab_l_h_V_stru)
        ag_h_V_stru = self.ag_output(ag_h_V_stru)
        h_V_con = torch.concat((ab_h_h_V_stru, ab_l_h_V_stru, ag_h_V_stru), dim = 1)
        h_V_con = self.FC_3(h_V_con)
        h_V_con = self.LN_3(h_V_con)


        # Inter-chain module
        sfea_tns = self.atom_attn_encoder(data_batch)
        x = sfea_tns.transpose(2, 1)
        c1 = self.drop(torch.relu(self.pool(self.conv_1(x)).squeeze()))  # B, 150
        c3 = self.drop(torch.relu(self.pool(self.conv_3(x)).squeeze()))  # B, 150
        c5 = self.drop(torch.relu(self.pool(self.conv_5(x)).squeeze()))  # B, 150
        c7 = self.drop(torch.relu(self.pool(self.conv_7(x)).squeeze()))  # B, 150
        c9 = self.drop(torch.relu(self.pool(self.conv_9(x)).squeeze()))  # B, 150
       
        conv_out = torch.concat([c1, c3, c5, c7, c9], dim = -1)
        atom_enc = self.FC_2(conv_out)
        atom_enc = self.LN_2(atom_enc)
        if len(atom_enc.shape) == 1:
            atom_enc = atom_enc.unsqueeze(0)
        # atom_enc = self.predict(atom_enc)
        try:
            concated_encoded = torch.concat((heavy_InnerSeq,light_InnerSeq,antigen_InnerSeq, h_V_con, atom_enc) , dim = 1)
        except:
            pdb.set_trace()

        # concated_encoded = torch.concat((heavy_InnerSeq,light_InnerSeq,antigen_InnerSeq, h_V_con), dim = 1)

        try:
            output = self.binding_predict(concated_encoded)
        except:
            pdb.set_trace()
        
        return  output
    
    def reparametrize(self, mu, logvar):
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
    
    def prior_expert(self, size, device):
        """Universal prior expert. Here we use a spherical
        Gaussian: N(0, 1).
        :param size: dimensionality of Gaussian
        """
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.zeros(size))
        mu, logvar = mu.to(device), logvar.to(device)
        return mu, logvar


class CNN_Module(nn.Module):
    def __init__(self, in_channel=118,emb_size = 20,hidden_size = 92):#189):
        super(CNN_Module, self).__init__()
        
        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn(in_channel = in_channel,hidden_channel = 64)   # 118*64
        self.conv2 = cnn(in_channel = 64,hidden_channel = 32) # 64*32

        self.conv3 = cnn(in_channel = 32,hidden_channel = 32)

        self.fc1 = nn.Linear(32*hidden_size , 128) # 32*29*512
        self.fc2 = nn.Linear(128 , 128)

        self.fc3 = nn.Linear(128 , 128)

    def forward(self, x):
        #x = x
        # x = self.emb(x)
        
        x = self.conv1(x)
        
        x = self.conv2(x)

        x = self.conv3(x)
        
        x = x.view(x.shape[0] ,-1)
        
        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x +sk

class CNN_Module_ver1(nn.Module):
    def __init__(self, in_channel=118,emb_size = 20,hidden_size = 92):#189):
        super(CNN_Module_ver1, self).__init__()
        
        # self.emb = nn.Embedding(emb_size,128)  # 20*128
        self.conv1 = cnn(in_channel = in_channel,hidden_channel = 64)   # 118*64
        self.conv2 = cnn(in_channel = 64,hidden_channel = 32) # 64*32

        self.conv3 = cnn(in_channel = 32,hidden_channel = 32)

        self.fc1 = nn.Linear(32*5 , 128) # 32*29*512
        self.fc2 = nn.Linear(128 , 128)

        self.fc3 = nn.Linear(128 , 128)

    def forward(self, x):
        #x = x
        # x = self.emb(x)
        
        x = self.conv1(x)
        
        x = self.conv2(x)
        # pdb.set_trace()
        x = x.view(x.shape[0] ,-1)

        x = nn.ReLU()(self.fc1(x))
        sk = x
        x = self.fc2(x)

        x = self.fc3(x)
        return x +sk
  
class cnn(nn.Module):
    def __init__(self, in_channel=2, hidden_channel=2, out_channel=2):
        super(cnn, self).__init__()
        
        self.cnn = nn.Conv1d(in_channel , hidden_channel , kernel_size = 5 , stride = 1) # bs * 64*60
        self.max_pool = nn.MaxPool1d(kernel_size = 2 , stride=2)# bs * 32*30
                               
        self.relu = nn.ReLU()
    
    def forward(self, x):
        
        #x = self.emb(x)
        x = self.cnn(x)
        x = self.max_pool(x)
        x = self.relu(x)
        return x