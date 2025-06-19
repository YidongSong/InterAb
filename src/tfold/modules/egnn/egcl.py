"""E(n)-equivariant graph convolutional layer."""

import numpy as np
import dgl.function as dgl_fn
import torch
from torch import nn

from tfold.modules.common import SiLU


class EGCL(nn.Module):  # pylint: disable=too-many-instance-attributes
    """E(n)-equivariant graph convolutional layer."""

    def __init__(
            self,
            n_dims_node_src,  # number of dimensions in source node features
            n_dims_node_dst,  # number of dimensions in destination node embeddings
            n_dims_edge,      # number of dimensions in edge features
            n_dims_emsg=16,   # number of dimensions in edge-wise messages
            enbl_denc=False,  # whether to enable distance encodings
            denc_base=2.0,    # exp. base for distance encodings
            n_dims_denc=11,   # number of dimensions in distance encodings
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        # initialization
        super().__init__()
        self.n_dims_node_src = n_dims_node_src
        self.n_dims_node_dst = n_dims_node_dst
        self.n_dims_edge = n_dims_edge
        self.n_dims_emsg = n_dims_emsg
        self.enbl_denc = enbl_denc
        self.denc_base = denc_base
        self.n_dims_denc = n_dims_denc
        self.n_dims_radi = n_dims_denc if self.enbl_denc else 1  # radial distance's dimension
        self.nets = nn.ModuleDict()

        # generate thresholds for distance encodings
        dist_vals_np = np.power(self.denc_base, np.arange(self.n_dims_denc))[None, :]
        self.dist_vals = torch.tensor(dist_vals_np, dtype=torch.float32)

        # build a sub-network to compute <m_ij> from {h_i, h_j, g_ij, r_ij}
        self.nets['m'] = nn.Sequential(
            nn.Linear(n_dims_node_src * 2 + n_dims_edge + self.n_dims_radi, n_dims_emsg),
            SiLU(),
            nn.Linear(n_dims_emsg, n_dims_emsg),
            SiLU(),
        )

        # build a sub-network to compute <e_ij> from {m_ij}
        self.nets['e'] = nn.Sequential(
            nn.Linear(n_dims_emsg, 1),
            nn.Sigmoid(),
        )

        # build a sub-network to compute <c_ij> from {m_ij}
        self.nets['c'] = nn.Sequential(
            nn.Linear(n_dims_emsg, n_dims_emsg),
            SiLU(),
            nn.Linear(n_dims_emsg, 1),
        )

        # build a sub-network to compute <h'_ij> from {m_ij, e_ij}
        if n_dims_node_src != n_dims_node_dst:
            self.nets['f'] = nn.Sequential(
                nn.Linear(n_dims_node_src + n_dims_emsg, n_dims_emsg),
                SiLU(),
                nn.Linear(n_dims_emsg, n_dims_node_dst),
            )
        else:
            self.nets['f'] = nn.Sequential(
                nn.Linear(n_dims_emsg, n_dims_emsg),
                SiLU(),
                nn.Linear(n_dims_emsg, n_dims_node_dst),
            )


    def forward(self, graph, node_feats, node_cords, edge_feats, node_masks=None):  # pylint: disable=too-many-arguments
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * node_feats: node features of size Nv x Dv
        * node_cords: node coordinates of size Nv x Dc (Dc = 3 for 3D coordinates)
        * edge_feats: edge features of size Ne x De
        * (optional) node_masks: node coordinates' validness masks of size Nv

        Returns:
        * node_feats_out: updated node features of size Nv x Dv
        * node_cords_out: updated node coordinates of size Nv x Dc
        """

        # initialization
        device = node_feats.device

        # perform the forward pass
        self.dist_vals = self.dist_vals.to(device)
        with graph.local_scope():
            # initialization
            graph.ndata['f'] = node_feats
            graph.ndata['x'] = node_cords.view(-1, 3)
            graph.edata['g'] = edge_feats
            graph.ndata['mk'] = node_masks.view(-1) if node_masks is not None else \
                torch.ones((node_feats.shape[0]), dtype=torch.float32, device=device)

            # calculate relative coordinates
            graph.apply_edges(dgl_fn.u_sub_v('x', 'x', 'dx'))

            # calculate the radial distance for all the edges
            graph.apply_edges(self.__calc_erad)

            # (h_i, h_j, a_ij, r_ij) => m_ij
            graph.apply_edges(self.__calc_emsg)

            # m_ij => e_ij & m_ij => c_ij
            graph.edata['e'] = self.nets['e'](graph.edata['m'])
            graph.edata['c'] = self.nets['c'](graph.edata['m'])

            # m_ij => c_ij & (x_i, x_j, c_ij) => x'_i
            graph.edata['cdx'] = graph.edata['c'] * graph.edata['dx']
            #graph.update_all(dgl_fn.copy_e('cdx', 'cdx'), dgl_fn.sum('cdx', 'dx'))
            graph.update_all(dgl_fn.u_mul_e('mk', 'cdx', 'mkcdx'), dgl_fn.sum('mkcdx', 'dx'))
            graph.ndata['xo'] = graph.ndata['x'] + graph.ndata['dx']

            # (f_i, m_ij, e_ij) => f'_i
            graph.edata['em'] = graph.edata['e'] * graph.edata['m']
            graph.update_all(dgl_fn.copy_edge('em', 'em'), dgl_fn.sum('em', 'em'))
            if self.n_dims_node_src != self.n_dims_node_dst:
                graph.ndata['fo'] = self.nets['f'](
                    torch.cat([graph.ndata['f'], graph.ndata['em']], dim=1))
            else:
                graph.ndata['fo'] = graph.ndata['f'] + self.nets['f'](graph.ndata['em'])

            # fetch the updated node features & delta term of node coordinates
            node_feats_out = graph.ndata['fo']
            node_cords_out = graph.ndata['xo']

        return node_feats_out, node_cords_out


    def __calc_erad(self, edges):
        """Calculate the edge-wise radial distance, with optional encodings."""

        mask_tns = edges.src['mk'].unsqueeze(dim=1)
        dist_tns = torch.norm(edges.data['dx'], dim=1, keepdim=True)
        if not self.enbl_denc:
            outputs = {'r': mask_tns * dist_tns}
        else:
            outputs = {'r': mask_tns * torch.sigmoid(dist_tns / self.dist_vals - 1.0)}

        return outputs


    def __calc_emsg(self, edges):
        """Calculate the edge-wise message."""

        inputs = torch.cat(
            [edges.src['f'], edges.dst['f'], edges.data['g'], edges.data['r']], dim=1)
        outputs = self.nets['m'](inputs)

        return {'m': outputs}
