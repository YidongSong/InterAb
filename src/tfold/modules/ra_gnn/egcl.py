"""E(n)-equivariant graph convolutional layer."""

import torch
from torch import nn
import dgl.function as dgl_fn

from tfold.modules.common import SiLU


class EGCL(nn.Module):
    """E(n)-equivariant graph convolutional layer."""

    def __init__(
            self,
            n_dims_nfea,  # number of dimensions in node features
            n_dims_efea,  # number of dimensions in edge features
            n_dims_emsg=32,   # number of dimensions in hidden edge messages
            dist_encoder=None,  # distance encoder
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_nfea = n_dims_nfea
        self.n_dims_efea = n_dims_efea
        self.n_dims_emsg = n_dims_emsg
        self.dist_encoder = dist_encoder

        # additional configurations
        self.n_dims_dist = 1 if self.dist_encoder is None else self.dist_encoder.n_dims
        self.n_dims_ehid = 2 * self.n_dims_nfea + self.n_dims_dist + self.n_dims_efea

        # sub-network: edge function
        self.net = nn.ModuleDict()
        self.net['e'] = nn.Sequential(
            nn.Linear(self.n_dims_ehid, self.n_dims_emsg),
            SiLU(),
            nn.Linear(self.n_dims_emsg, self.n_dims_emsg),
            SiLU(),
        )

        # sub-network: coordinate function
        self.net['x'] = nn.Sequential(
            nn.Linear(self.n_dims_emsg, self.n_dims_emsg),
            SiLU(),
            nn.Linear(self.n_dims_emsg, 1),
        )

        # sub-network: node function
        self.net['h'] = nn.Sequential(
            nn.Linear(self.n_dims_nfea + self.n_dims_emsg, self.n_dims_nfea),
            SiLU(),
            nn.Linear(self.n_dims_nfea, self.n_dims_nfea),
        )

        # manual initialization for all the linear layers
        def _init_fn(module):
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0.0, 0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.net.apply(_init_fn)


    def forward(self, graph, nfea_mat, efea_mat, cord_mat, cmsk_vec, umsk_vec):  # pylint: disable=too-many-arguments
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * nfea_mat: node features of size Nv x Dv
        * efea_mat: edge features of size Ne x De
        * cord_mat: per-atom 3D coordinates of size Nv x 3
        * cmsk_vec: per-atom 3D coordinates' validness masks of size Nv
        * umsk_vec: per-atom 3D coordinates' update-or-not masks of size Nv

        Returns:
        * nfea_mat_out: updated node features of size Nv x Dv
        * cord_mat_out: updated per-atom 3D coordinates of size Nv x 3
        """

        with graph.local_scope():
            # initialization
            graph.ndata['f'] = nfea_mat
            graph.edata['g'] = efea_mat
            graph.ndata['x'] = cord_mat
            graph.ndata['v'] = cmsk_vec.to(torch.float32)  # valid-or-not
            graph.ndata['u'] = umsk_vec.to(torch.float32)  # update-or-not

            # calculate relative coordinates & validness masks
            graph.apply_edges(dgl_fn.u_sub_v('x', 'x', 'ex'))
            graph.apply_edges(dgl_fn.u_mul_v('v', 'v', 'ev'))

            # (h_i, h_j, r_ij, a_ij) => m_ij
            graph.apply_edges(self.__calc_emsg)

            # (x_i, r_ij, m_ij) => x'_i
            graph.edata['xm'] = self.net['x'](graph.edata['m']) * \
                graph.edata['ev'].unsqueeze(dim=1) * graph.edata['ex']
            graph.update_all(dgl_fn.copy_e('xm', 'xm_t'), dgl_fn.sum('xm_t', 'xd'))
            graph.ndata['xo'] = graph.ndata['x'] + \
                graph.ndata['u'].unsqueeze(dim=1) * graph.ndata['xd']

            # (h_i, m_ij) => h'_i
            graph.update_all(dgl_fn.copy_e('m', 'm_t'), dgl_fn.sum('m_t', 'm'))
            graph.ndata['fo'] = graph.ndata['f'] + self.net['h'](
                torch.cat([graph.ndata['f'], graph.ndata['m']], dim=1))

            # fetch the updated node features & delta term of node coordinates
            nfea_mat_out = graph.ndata['fo']
            cord_mat_out = graph.ndata['xo']

        return nfea_mat_out, cord_mat_out


    def __calc_emsg(self, edges):
        """Calculate hidden edge messages."""

        # calculate additional edge-wise features from radial distance
        dist_vec = torch.norm(edges.data['ex'], dim=1)
        if self.dist_encoder is None:
            efea_mat_radi = dist_vec.unsqueeze(dim=1)  # Ne x 1
        else:
            efea_mat_radi = self.dist_encoder.run(dist_vec)  # Ne x Dr

        # calculate hidden edge messages
        ehid_tns = torch.cat([
            edges.src['f'], edges.dst['f'], edges.data['g'],
            edges.data['ev'].unsqueeze(dim=1) * efea_mat_radi,
        ], dim=1)
        emsg_tns = self.net['e'](ehid_tns)

        return {'m': emsg_tns}
