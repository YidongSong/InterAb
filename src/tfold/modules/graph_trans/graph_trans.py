"""Graph transformer.

Reference:
Shi et al., Masked Label Prediction: Unified Message Passing Model for Semi-Supervised
  Classification. IJCAI 2021.
"""

import torch
from torch import nn

from tfold.modules.graph_trans.multi_head_attn import MultiHeadAttn


class GraphTrans(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Graph transformer."""

    def __init__(
            self,
            n_dims_node,  # number of dimensions in node features
            n_dims_edge,  # number of dimensions in edge features
            n_heads,      # number of heads in multi-head attentions
            n_dims_attn,  # number of dimensions in attention embeddings (query & key)
            aggr_fn,      # aggregation function (choices: 'mean' / 'concat')
            use_nlnr,     # whether to use the final non-linearity transformation
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        self.n_dims_node = n_dims_node
        self.n_dims_edge = n_dims_edge
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn
        self.aggr_fn = aggr_fn
        self.use_nlnr = use_nlnr

        n_dims_out = n_dims_attn if self.aggr_fn == 'mean' else n_heads * n_dims_attn
        self.mha = MultiHeadAttn(n_dims_node, n_dims_edge, n_heads, n_dims_attn)
        self.linear_r = nn.Linear(n_dims_node, n_dims_out)
        self.linear_b = nn.Linear(3 * n_dims_out, n_dims_out)
        if use_nlnr:
            self.nlnr = nn.Sequential(
                nn.LayerNorm(n_dims_out),
                nn.ReLU(),
            )


    def forward(self, node_feats, edge_feats, edge_masks=None):
        """Perform the forward pass.

        Args:
        * node_feats: node features of size BS x N x D_v
        * edge_feats: edge features of size BS x N x N x D_e
        * edge_masks: edge validness masks of size BS x N x N

        Returns:
        * node_feats_out: updated node features of size BS x N x D_o

        Note
        * D_o = D_a if <self.aggr_fn> == 'mean' else H x D_a
        """

        # initialization
        n_smpls = node_feats.shape[0]
        n_nodes = node_feats.shape[1]
        assert list(node_feats.shape) == [n_smpls, n_nodes, self.n_dims_node]
        assert list(edge_feats.shape) == [n_smpls, n_nodes, n_nodes, self.n_dims_edge]

        # calculate multi-head messages, which are aggregated at each node
        node_msgs = self.mha(node_feats, edge_feats, edge_masks)  # BS x N x H x D_a
        if self.aggr_fn == 'concat':
            m_tns = torch.reshape(node_msgs, [n_smpls, n_nodes, -1])  # BS x N x (H x D_a)
        elif self.aggr_fn == 'mean':
            m_tns = torch.mean(node_msgs, dim=2)  # BS x N x D_a
        else:
            raise ValueError(f'unrecognized aggregation function: {self.aggr_fn}')

        # calculate coefficients for gated residual connections (to prevent over-smoothing)
        r_tns = self.linear_r(node_feats)  # BS x N x D_o
        b_tns = torch.sigmoid(self.linear_b(torch.cat([m_tns, r_tns, m_tns - r_tns], dim=2)))
        node_feats_out = (1.0 - b_tns) * m_tns + b_tns * r_tns  # BS x N x D_o

        # (optional) final non-linear transformation
        if self.use_nlnr:
            node_feats_out = self.nlnr(node_feats_out)  # BS x N x D_o

        return node_feats_out


    def __repr__(self):
        """Get the string representation."""

        config_str = ', '.join([
            f'n_dims_node={self.n_dims_node}',
            f'n_dims_edge={self.n_dims_edge}',
            f'n_heads={self.n_heads}',
            f'n_dims_attn={self.n_dims_attn}',
            f'aggr_fn={self.aggr_fn}',
            f'use_nlnr={self.use_nlnr}',
        ])
        repr_str = f'GraphTrans({config_str})'

        return repr_str
