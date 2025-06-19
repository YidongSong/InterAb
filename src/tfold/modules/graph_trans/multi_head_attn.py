"""Multi-head attention for densely-connected graphs.

Inputs:
h_i: node features
g_ij: edge features
m_ij: edge validness masks

Multi-head attention w/o masks:
q_i = MLP ( h_i )
k_i = MLP ( h_i )
v_i = MLP ( h_i )
e_ij = MLP ( g_ij )
z_ij = < q_i, k_j + e_ij >
p_ij = m_ij * EXP ( z_ij )
a_ij = p_ij / sum_j' p_ij'
"""

import numpy as np
import torch
from torch import nn


class MultiHeadAttn(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-head attention for densely-connected graphs."""

    def __init__(self, n_dims_node, n_dims_edge, n_heads, n_dims_attn):
        """Constructor function."""

        super().__init__()

        self.n_dims_node = n_dims_node
        self.n_dims_edge = n_dims_edge
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn

        self.eps = 1e-6
        self.linear_q = nn.Linear(n_dims_node, n_heads * n_dims_attn)
        self.linear_k = nn.Linear(n_dims_node, n_heads * n_dims_attn)
        self.linear_v = nn.Linear(n_dims_node, n_heads * n_dims_attn)
        self.linear_e = nn.Linear(n_dims_edge, n_heads * n_dims_attn)


    def forward(self, node_feats, edge_feats, edge_masks=None):  # pylint: disable=too-many-locals
        """Perform the forward pass.

        Args:
        * node_feats: node features of size BS x N x D_v
        * edge_feats: edge features of size BS x N x N x D_e
        * edge_masks: edge validness masks of size BS x N x N

        Returns:
        * node_msgs: multi-head messages of size BS x N x H x D_a
        """

        # initialization
        device = node_feats.device
        n_smpls = node_feats.shape[0]
        n_nodes = node_feats.shape[1]
        assert list(node_feats.shape) == [n_smpls, n_nodes, self.n_dims_node]
        assert list(edge_feats.shape) == [n_smpls, n_nodes, n_nodes, self.n_dims_edge]
        if edge_masks is None:
            edge_masks = torch.ones((n_smpls, n_nodes, n_nodes), dtype=torch.float32).to(device)

        # calculate multi-head attention coefficients
        # > q_tns: BS x N x 1 x H x D_a - query embeddings
        # > k_tns: BS x N x 1 x H x D_a - key embeddings
        # > v_tns: BS x N x 1 x H x D_a - value embeddings
        # > e_tns: BS x N x N x H x D_a - edge embeddings
        # > z_tns: BS x N x N x H       - unnormalized attentions (raw)
        # > p_tns: BS x N x N x H       - unnormalized attentions (max value subtracted)
        # > a_tns: BS x N x N x H       - normalized attentions
        q_tns = self.linear_q(node_feats).view(n_smpls, n_nodes, 1, self.n_heads, -1)
        k_tns = self.linear_k(node_feats).view(n_smpls, n_nodes, 1, self.n_heads, -1)
        v_tns = self.linear_v(node_feats).view(n_smpls, n_nodes, 1, self.n_heads, -1)
        e_tns = self.linear_e(edge_feats).view(n_smpls, n_nodes, n_nodes, self.n_heads, -1)
        z_tns = torch.sum(q_tns * (k_tns + e_tns), dim=4) / np.sqrt(self.n_dims_attn)
        z_tns_max = torch.max(z_tns, dim=2, keepdim=True)[0]
        p_tns = edge_masks.unsqueeze(dim=3) * torch.exp(z_tns - z_tns_max)
        a_tns = p_tns / (torch.sum(p_tns, dim=2, keepdim=True) + self.eps)

        # calculate multi-head messages, which are aggregated at each node
        node_msgs = torch.sum(a_tns.unsqueeze(dim=-1) * (v_tns + e_tns), dim=2)

        return node_msgs


    def __repr__(self):
        """Get the string representation."""

        config_str = ', '.join([
            f'n_dims_node={self.n_dims_node}',
            f'n_dims_edge={self.n_dims_edge}',
            f'n_heads={self.n_heads}',
            f'n_dims_attn={self.n_dims_attn}',
        ])
        repr_str = f'MultiHeadAttn({config_str})'

        return repr_str
