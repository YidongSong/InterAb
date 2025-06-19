"""Geometric frame perceptron network."""

from torch import nn

from tfold.tools import DistEncoder
from tfold.tools import FrcdEncoder
from tfold.modules.egnn.gfp import GFP


class GFPNet(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Geometric frame perceptron network."""

    def __init__(
            self,
            n_lyrs,            # number of layers
            n_dims_nfea,       # number of dimensions in node features
            n_dims_efea,       # number of dimensions in edge features
            n_grps_cord,       # number of groups of 3D coordinates per node
            n_dims_nemb,       # number of dimensions in node output embeddings
            encd_type='frcd',  # encoding type (choices: 'dist-s' / 'dist-m' / 'frcd')
            n_dims_nhid=32,    # number of dimensions in node hidden embeddings
            n_dims_emsg=32,    # number of dimensions in edge-wise messages
            n_heads=8,         # number of heads in multi-head attentions
            n_dims_attn=32,    # number of dimensions in multi-head attention embeddings
            updt_cord=False,   # whether to update 3D coordinates
            drop_prob=0.0,     # drop-out probability (how likely one entry is reset to zero)
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # initialization
        self.n_lyrs = n_lyrs
        self.n_dims_nfea = n_dims_nfea
        self.n_dims_efea = n_dims_efea
        self.n_grps_cord = n_grps_cord
        self.n_dims_nemb = n_dims_nemb
        self.encd_type = encd_type
        self.n_dims_nhid = n_dims_nhid
        self.n_dims_emsg = n_dims_emsg
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn
        self.updt_cord = updt_cord
        self.drop_prob = drop_prob

        # build encoders for distance / frame & coordinate
        self.dist_encoder = DistEncoder()
        self.frcd_encoder = FrcdEncoder(self.dist_encoder, self.n_grps_cord)

        # create <EGCL> / <MhaEGCL> layers
        self.layers = nn.ModuleList()
        for idx_lyr in range(self.n_lyrs):
            # determine the merge function for node hidden embeddings
            merge_fn = 'cat' if idx_lyr != self.n_lyrs - 1 else 'avg'

            # input linear layer
            if idx_lyr == 0:
                self.layers.append(nn.Linear(self.n_dims_nfea, self.n_dims_nhid))

            # geometric frame perceptron layer
            self.layers.append(GFP(
                n_dims_nhid, n_dims_efea, n_grps_cord,
                encd_type=self.encd_type, n_dims_emsg=self.n_dims_emsg,
                dist_encoder=self.dist_encoder, frcd_encoder=self.frcd_encoder,
                n_heads=self.n_heads, n_dims_attn=self.n_dims_attn,
                updt_cord=self.updt_cord, merge_fn=merge_fn,
            ))

            # normalization layer for node features
            if idx_lyr != self.n_lyrs - 1:
                self.layers.append(nn.LayerNorm(n_dims_nhid))

            # drop-out layer for node features
            if (idx_lyr != self.n_lyrs - 1) and (self.drop_prob > 0.0):
                self.layers.append(nn.Dropout(p=self.drop_prob))

            # output linear layer
            if idx_lyr == self.n_lyrs - 1:
                self.layers.append(nn.Linear(self.n_dims_nhid, self.n_dims_nemb))


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
            if isinstance(layer, GFP):
                node_feats, node_cords = layer(
                    graph, node_feats, node_cords, edge_feats, node_masks)
            elif isinstance(layer, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                node_feats = layer(node_feats)
            else:
                raise ValueError(f'unrecognized layer: {layer}')

        return node_feats, node_cords
