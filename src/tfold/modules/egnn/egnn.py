"""E(n)-equivariant Graph Neural Network."""

from torch import nn

from tfold.modules.egnn.egcl import EGCL
from tfold.modules.egnn.mha_egcl import MhaEGCL


class EGNN(nn.Module):
    """E(n)-equivariant graph neural network."""

    def __init__(
            self,
            n_lyrs,              # number of layers
            n_dims_nfea,         # number of dimensions in node features
            n_dims_embd,         # number of dimensions in node output embeddings
            n_dims_efea,         # number of dimensions in edge features
            lyr_type='MhaEGCL',  # layer type (choices: 'EGCL' / 'MhaEGCL')
            n_dims_emsg=16,      # number of dimensions in edge-wise messages
            n_dims_nhid=16,      # number of dimensions in node hidden embeddings
            n_grps_cord=1,       # number of groups of 3D coordinates per node
            n_heads=1,           # number of heads in multi-head attentions
            n_dims_attn=16,      # number of dimensions in multi-head attention embeddings
            drop_prob=0.0,       # drop-out probability (how likely one entry is reset to zero)
        ):  # pylint: disable=too-many-arguments,too-many-locals
        """Constructor function."""

        super().__init__()

        # additional configurations
        self.enbl_denc = True  # whether to enable distance encodings
        self.denc_base = 2.0   # exp. base for distance encodings
        self.n_dims_denc = 11  # number of dimensions in distance encodings
        self.enbl_norm = True  # whether to enable layer normalization

        # create <EGCL> / <MhaEGCL> layers
        self.layers = nn.ModuleList()
        for idx_lyr in range(n_lyrs):
            # determine the input/output dimension of node features
            merge_fn = 'cat' if idx_lyr != n_lyrs - 1 else 'avg'
            n_dims_node_src = n_dims_nhid if idx_lyr != 0 else n_dims_nfea
            n_dims_node_dst = n_dims_nhid if idx_lyr != n_lyrs - 1 else n_dims_embd

            # create a <EGCL> or <MhaEGCL> layer
            if lyr_type == 'EGCL':
                self.layers.append(EGCL(
                    n_dims_node_src, n_dims_node_dst, n_dims_efea,
                    n_dims_emsg=n_dims_emsg, enbl_denc=self.enbl_denc, denc_base=self.denc_base,
                    n_dims_denc=self.n_dims_denc,
                ))
            elif lyr_type == 'MhaEGCL':
                self.layers.append(MhaEGCL(
                    n_dims_node_src, n_dims_node_dst, n_dims_efea,
                    n_dims_emsg=n_dims_emsg, enbl_denc=self.enbl_denc, denc_base=self.denc_base,
                    n_dims_denc=self.n_dims_denc, n_grps_cord=n_grps_cord, n_heads=n_heads,
                    n_dims_attn=n_dims_attn, merge_fn=merge_fn,
                ))
            else:
                raise ValueError(f'unrecognized layer type: {lyr_type}')

            # create a normalization layer for node features
            if self.enbl_norm and idx_lyr != n_lyrs - 1:
                self.layers.append(nn.LayerNorm(n_dims_node_dst))

            # create a drop-out layer for node features
            if drop_prob > 0.0 and idx_lyr != n_lyrs - 1:
                self.layers.append(nn.Dropout(p=drop_prob))


    def forward(self, graph, node_feats, node_cords, edge_feats, node_masks=None):  # pylint: disable=too-many-arguments
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * node_feats: node features of size Nv x Dv
        * node_cords: node coordinates of size Nv x 3 or Nv x G x 3
        * edge_feats: edge features of size Ne x De
        * (optional) node_masks: node coordinates' validness masks of size Nv

        Returns:
        * node_feats: updated node features of size Nv x Dv
        * node_cords: updated node coordinates of size Nv x 3 or Nv x G x 3
        """

        # perform the forward pass
        for layer in self.layers:
            if isinstance(layer, (EGCL, MhaEGCL)):
                node_feats, node_cords = layer(
                    graph, node_feats, node_cords, edge_feats, node_masks)
            elif isinstance(layer, (nn.LayerNorm, nn.Dropout)):
                node_feats = layer(node_feats)
            else:
                raise ValueError(f'unrecognized layer: {layer}')

        return node_feats, node_cords
