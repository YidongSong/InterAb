"""The E(n) equivariant convolutional layer (EGCL) for bipartite graphs.

Notes:
* Only secondary nodes' features, coordinates, velocities, and forces are updated.
* To update primary nodes, create another layer w/ primary and secondary nodes swapped.
* Please refer to <EqGraphConv> for details on how 3D coordinates are updated.
"""

import torch
from torch import nn
from torch_geometric.nn import MessagePassing

from tfold.modules.common import SiLU
from tfold.modules.ra_mpnn.cord_norm import CordNorm
from tfold.modules.ra_mpnn.fourier_dist_encoder import FourierDistEncoder


class EqGraphConvBG(MessagePassing):
    """The E(n) equivariant convolutional layer (EGCL) for bipartite graphs."""

    def __init__(
            self,
            n_dims_nfea_pri,  # number of dimensions in primary node features
            n_dims_nfea_sec,  # number of dimensions in secondary node features
            n_dims_emsg=64,  # number of dimensions in edge-wise messages
            n_dims_efea=0,  # number of dimensions in edge features
            cord_scale=1.0,  # scaling factor of 3D coordinates (in Angstrom)
            updt_nvlc=True,  # whether to update node velocities
            updt_nfrc=True,  # whether to update node forces
            norm_cord=True,  # whether to normalize relative coordinates between nodes
            rezero_init=True,  # whether to use re-zero initialization
            mp_aggr_op='add',  # aggregation operation (choices: 'add' / 'mean' / 'max')
        ):
        """Constructor function."""

        super().__init__(aggr=mp_aggr_op)

        # setup configurations
        self.n_dims_nfea_pri = n_dims_nfea_pri
        self.n_dims_nfea_sec = n_dims_nfea_sec
        self.n_dims_emsg = n_dims_emsg
        self.n_dims_efea = n_dims_efea
        self.cord_scale = cord_scale
        self.updt_nvlc = updt_nvlc
        self.updt_nfrc = updt_nfrc
        self.norm_cord = norm_cord
        self.rezero_init = rezero_init

        # additional configurations
        self.eps = 1e-6
        self.dropout = 0.1
        self.n_dims_denc = 32  # number of dimensions in distance encodings
        self.denc_base = 1.5
        self.dist_encoder = FourierDistEncoder(
            n_dims=self.n_dims_denc, base=self.denc_base, cord_scale=self.cord_scale)
        self.n_dims_emsg_init = \
            self.n_dims_nfea_pri + self.n_dims_nfea_sec + self.n_dims_denc + self.n_dims_efea

        # build the network
        self.net = nn.ModuleDict()
        self.net['mlp-e'] = nn.Sequential(
            nn.LayerNorm(self.n_dims_emsg_init),
            nn.Linear(self.n_dims_emsg_init, self.n_dims_emsg),
            nn.Dropout(p=self.dropout),
            SiLU(),
            nn.Linear(self.n_dims_emsg, self.n_dims_emsg),
            nn.Dropout(p=self.dropout),
            SiLU(),
        )
        if self.updt_nvlc:
            self.net['mlp-v'] = nn.Sequential(
                nn.Linear(self.n_dims_emsg, self.n_dims_emsg),
                nn.Dropout(p=self.dropout),
                SiLU(),
                nn.Linear(self.n_dims_emsg, 1),
            )
        if self.updt_nfrc:
            self.net['mlp-f'] = nn.Sequential(
                nn.Linear(self.n_dims_emsg, self.n_dims_emsg),
                nn.Dropout(p=self.dropout),
                SiLU(),
                nn.Linear(self.n_dims_emsg, 1),
            )
        self.net['mlp-h'] = nn.Sequential(
            nn.Linear(self.n_dims_nfea_sec + self.n_dims_emsg, self.n_dims_nfea_sec),
            nn.Dropout(p=self.dropout),
            SiLU(),
            nn.Linear(self.n_dims_nfea_sec, self.n_dims_nfea_sec),
        )
        if self.norm_cord:
            self.net['norm-c'] = CordNorm()

        # setup the re-zero initialization
        if not self.rezero_init:
            self.alpha_h = 1.0
            self.alpha_v = 1.0
            self.alpha_f = 1.0
        else:
            self.alpha_h = nn.Parameter(data=torch.tensor([1e-3], dtype=torch.float32))
            self.alpha_v = nn.Parameter(data=torch.tensor([1e-3], dtype=torch.float32))
            self.alpha_f = nn.Parameter(data=torch.tensor([1e-3], dtype=torch.float32))

        # initialize all the linear layers
        def _init_weights(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

        self.net.apply(_init_weights)


    def forward(
            self, nfea_mat_pri, nfea_mat_sec, ncrd_mat_pri, ncrd_mat_sec, eidx_mat,
            efea_mat=None, nvlc_mat_sec=None, nfrc_mat_sec=None, ncum_vec_sec=None,
        ):
        """Perform the forward pass.

        Args:
        * nfea_mat_pri: primary node features of size N_vp x D_vp
        * nfea_mat_sec: secondary node features of size N_vs x D_vs
        * ncrd_mat_pri: primary node coordinates of size N_vp x 3
        * ncrd_mat_sec: secondary node coordinates of size N_vs x 3
        * eidx_mat: edge indices of size 2 x N_e (following PyG tradition)
        * efea_mat: (optional) edge features of size N_e x D_e
        * nvlc_mat_sec: (optional) secondary node velocities of size N_vs x 3
        * nfrc_mat_sec: (optional) secondary node forces of size N_vs x 3
        * ncum_vec_sec: (optional) secondary node coordinates' update-or-not masks of size N_vs

        Returns:
        * nfea_mat_sec: updated node features of size N_vs x D_vs
        * ncrd_mat_sec: updated node coordinates of size N_vs x 3
        * nvlc_mat_sec: updated node velocities of size N_vs x 3
        * nfrc_mat_sec: updated node forces of size N_vs x 3

        Notes:
        * The first row in <eidx_mat> consists of primary node indices (as source).
        * The second row in <eidx_mat> consists of secondary node indices (as destination).
        """

        # initialization
        dtype = nfea_mat_pri.dtype
        device = nfea_mat_pri.device
        n_nodes_pri = nfea_mat_pri.shape[0]
        n_nodes_sec = nfea_mat_sec.shape[0]
        n_edges = eidx_mat.shape[1]

        # initialize optional arguments
        if efea_mat is None:
            efea_mat = torch.zeros((n_edges, 0), dtype=dtype, device=device)
        if nvlc_mat_sec is None:
            nvlc_mat_sec = torch.zeros((n_nodes_sec, 3), dtype=dtype, device=device)
        if nfrc_mat_sec is None:
            nfrc_mat_sec = torch.zeros((n_nodes_sec, 3), dtype=dtype, device=device)
        if ncum_vec_sec is None:
            ncum_vec_sec = torch.ones(n_nodes_sec, dtype=torch.int8, device=device)

        # perform the forward pass
        nfea_mat_sec, ncrd_mat_sec, nvlc_mat_sec, nfrc_mat_sec = self.propagate(
            eidx_mat, size=(n_nodes_pri, n_nodes_sec),
            nfea_mat=(nfea_mat_pri, nfea_mat_sec),
            ncrd_mat=(ncrd_mat_pri, ncrd_mat_sec),
            efea_mat=efea_mat,
            nvlc_mat_sec=nvlc_mat_sec, nfrc_mat_sec=nfrc_mat_sec, ncum_vec_sec=ncum_vec_sec,
        )

        return nfea_mat_sec, ncrd_mat_sec, nvlc_mat_sec, nfrc_mat_sec


    def message(self, nfea_mat_i, nfea_mat_j, ncrd_mat_i, ncrd_mat_j, efea_mat):
        """Calculate edge-wise messages.

        Notes:
        * If <flow> is 'source_to_target', then an edge-wise message is constructed if there exists
            an edge starting from <j> and ending at <i>.
        * If <flow> is 'target_to_source', then an edge-wise message is constructed if there exists
            an edge starting from <i> and ending at <j>.
        * Edge-wise messages are always aggregated at node <i>, a.k.a. central nodes, regardless of
            the specified flow direction of message passing.
        """

        # encode relative distance between nodes
        dcrd_mat = ncrd_mat_j - ncrd_mat_i  # relative coordinates
        if self.norm_cord:
            dcrd_mat = self.net['norm-c'](dcrd_mat)
        dist_vec = torch.norm(dcrd_mat, dim=1)  # relative distance
        denc_mat = self.dist_encoder.run(dist_vec)  # distance encodings

        # calculate edge-wise messages
        emsg_mat_init = torch.cat([nfea_mat_j, nfea_mat_i, denc_mat, efea_mat], dim=1)
        emsg_mat = self.net['mlp-e'](emsg_mat_init)  # edge-wise messages

        # calculate edge-wise update terms of node velocities
        if self.updt_nvlc:
            coef_mat_nvlc = self.net['mlp-v'](emsg_mat)
            emsg_mat_nvlc = coef_mat_nvlc * dcrd_mat
        else:
            emsg_mat_nvlc = torch.zeros_like(ncrd_mat_i)

        # calculate edge-wise update terms of node forces
        if self.updt_nfrc:
            coef_mat_nfrc = self.net['mlp-f'](emsg_mat)
            emsg_mat_nfrc = coef_mat_nfrc * dcrd_mat
        else:
            emsg_mat_nfrc = torch.zeros_like(ncrd_mat_i)

        # concatenate edge-wise messages & update terms for aggregation
        emsg_mat_ext = torch.cat([emsg_mat, emsg_mat_nvlc, emsg_mat_nfrc], dim=1)

        return emsg_mat_ext


    def update(self, aggr_mat_ext, nfea_mat, ncrd_mat, nvlc_mat_sec, nfrc_mat_sec, ncum_vec_sec):
        """Update secondary node features (and coordinates)."""

        # fetch secondary node features, coordinates, and forces
        _, nfea_mat_sec = nfea_mat  # please note that <nfea_mat> is actually a tuple
        _, ncrd_mat_sec = ncrd_mat

        # initialization
        aggr_mat, aggr_mat_nvlc, aggr_mat_nfrc = \
            torch.split(aggr_mat_ext, [self.n_dims_emsg, 3, 3], dim=1)

        # update secondary node features, velocities, and forces
        nfea_mat_sec = nfea_mat_sec \
            + self.alpha_h * self.net['mlp-h'](torch.cat([nfea_mat_sec, aggr_mat], dim=1))
        nvlc_mat_sec = nvlc_mat_sec + self.alpha_v * aggr_mat_nvlc
        nfrc_mat_sec = nfrc_mat_sec + self.alpha_f * aggr_mat_nfrc

        # update node coordinates
        ncrd_mat_sec = ncrd_mat_sec + ncum_vec_sec.unsqueeze(dim=1) * nvlc_mat_sec

        return nfea_mat_sec, ncrd_mat_sec, nvlc_mat_sec, nfrc_mat_sec
