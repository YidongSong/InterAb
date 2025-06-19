"""The network consists of stacked graph transformer layers.

Reference:
Shi et al., Masked Label Prediction: Unified Message Passing Model for Semi-Supervised
  Classification. IJCAI 2021.
"""

from torch import nn

from tfold.modules.graph_trans.graph_trans import GraphTrans


class GraphTransNet(nn.Module):
    """The network consists of stacked graph transformer layers."""

    def __init__(
            self,
            n_lyrs,           # number of layers
            n_dims_node_in,   # number of dimensions in input node features
            n_dims_node_out,  # number of dimensions in output node embeddings
            n_dims_edge,      # number of dimensions in edge features
            n_heads=1,        # number of heads in multi-head attentions
            n_dims_attn=16,   # number of dimensions in attention embeddings (query & key)
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        self.n_lyrs = n_lyrs
        self.n_dims_node_in = n_dims_node_in
        self.n_dims_node_out = n_dims_node_out
        self.n_dims_edge = n_dims_edge
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn

        n_dims_node_hid = n_heads * n_dims_attn
        self.layers = nn.ModuleList()
        for idx_lyr in range(n_lyrs):
            n_dims_node = n_dims_node_in if idx_lyr == 0 else n_dims_node_hid
            aggr_fn = 'concat' if idx_lyr != n_lyrs - 1 else 'mean'
            use_nlnr = (idx_lyr != n_lyrs - 1)  # do not use non-linearity in the final layer
            self.layers.append(
                GraphTrans(n_dims_node, n_dims_edge, n_heads, n_dims_attn, aggr_fn, use_nlnr))
        self.layers.append(nn.Linear(n_dims_attn, n_dims_node_out))


    def forward(self, node_feats, edge_feats, edge_masks=None):
        """Perform the forward pass.

        Args:
        * node_feats: node features of size BS x N x D_v
        * edge_feats: edge features of size BS x N x N x D_e
        * edge_masks: edge validness masks of size BS x N x N

        Returns:
        * node_feats: updated node features of size BS x N x D_o
        """

        for layer in self.layers:
            if isinstance(layer, GraphTrans):
                node_feats = layer(node_feats, edge_feats, edge_masks)
            elif isinstance(layer, nn.Linear):
                node_feats = layer(node_feats)
            else:
                raise ValueError(f'unrecognized layer type: {type(layer)}')

        return node_feats


    def __repr__(self):
        """Get the string representation."""

        config_str = ', '.join([
            f'n_lyrs={self.n_lyrs}',
            f'n_dims_node_in={self.n_dims_node_in}',
            f'n_dims_node_out={self.n_dims_node_out}',
            f'n_dims_edge={self.n_dims_edge}',
            f'n_heads={self.n_heads}',
            f'n_dims_attn={self.n_dims_attn}',
        ])
        repr_str = f'GraphTransNet({config_str})'

        return repr_str
