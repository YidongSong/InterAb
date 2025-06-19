"""Multi-head attention-based E(n)-equivariant graph convolutional layer."""

import numpy as np
import dgl.function as dgl_fn
import torch
from torch import nn

from tfold.utils import split_by_head
from tfold.modules.common import SiLU
from tfold.modules.common import MhaLinear


class MhaEGCL(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Multi-head attention-based E(n)-equivariant graph convolutional layer.

    This module supports each graph node being associated with multiple groups of 3D coordinates.
    """

    def __init__(
            self,
            n_dims_node_src,  # number of dimensions in source node features
            n_dims_node_dst,  # number of dimensions in destination node embeddings
            n_dims_edge,      # number of dimensions in edge features
            n_dims_emsg=16,   # number of dimensions in edge-wise messages
            enbl_denc=False,  # whether to enable distance encodings
            denc_base=2.0,    # exp. base for distance encodings
            n_dims_denc=11,   # number of dimensions in distance encodings
            n_grps_cord=1,    # number of groups of 3D coordinates per node
            n_heads=1,        # number of heads in multi-head attentions
            n_dims_attn=16,   # number of dimensions in attention embeddings (query & key)
            merge_fn='avg',   # merge function
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
        self.n_grps_cord = n_grps_cord
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn
        self.merge_fn = merge_fn

        self.eps = 1e-6
        self.idx_atom_ca = 0 if self.n_grps_cord == 1 else 1
        self.nets = nn.ModuleDict()

        # generate thresholds for distance encodings
        dist_vals_np = np.power(self.denc_base, np.arange(self.n_dims_denc))[None, :]
        self.dist_vals = torch.tensor(dist_vals_np, dtype=torch.float32)

        # fall back to 'avg-merge' if <cat-merge> is not available
        if (self.merge_fn == 'cat') and (self.n_dims_node_src % self.n_heads != 0):
            self.merge_fn = 'avg'

        # determine the dimension of each <AttnEGCL>'s output node features
        if self.merge_fn == 'avg':
            n_dims_node_hid = n_dims_node_src
        elif self.merge_fn == 'cat':
            n_dims_node_hid = n_dims_node_src // n_heads
        else:
            raise ValueError('unrecognized merge function for multi-head attention: ' + merge_fn)

        # build a sub-network to compute <m_ij^k> from {h_i, h_j, g_ij, r_ij}
        self.nets['m'] = nn.Sequential(
            nn.Linear(n_dims_node_src * 2 + n_dims_edge + self.n_dims_radi, n_heads * n_dims_emsg),
            SiLU(),
            MhaLinear(n_heads, n_dims_emsg, n_dims_emsg),
            SiLU(),
        )  # per-head message

        # build a sub-network to compute <q_i^k> from <h_i>
        self.nets['q'] = nn.Sequential(
            nn.Linear(n_dims_node_src, n_heads * n_dims_attn),
            nn.ELU(),
            MhaLinear(n_heads, n_dims_attn, n_dims_attn),
        )  # per-head query vectors

        # build a sub-network to compute <k_ij^k> from {h_j, m_ij^k}
        self.nets['k'] = nn.Sequential(
            MhaLinear(n_heads, n_dims_node_src + n_dims_emsg, n_dims_attn),
            nn.ELU(),
            MhaLinear(n_heads, n_dims_attn, n_dims_attn),
        )  # per-head key vectors

        # build a sub-network to compute <e_ij^k> from {q_i^k, k_ij^k}
        self.nets['e'] = nn.Sequential(
            MhaLinear(n_heads, 2 * n_dims_attn, 1),
            nn.LeakyReLU(),
        )  # per-head unnormalized attention coefficients

        # build a sub-network to compute <c_ij^k> from {m_ij^k}
        self.nets['c'] = nn.Sequential(
            MhaLinear(n_heads, n_dims_emsg, n_dims_emsg),
            SiLU(),
            MhaLinear(n_heads, n_dims_emsg, n_grps_cord),
        )  # per-head weighting coefficients for updating coordinates

        # build a sub-network to compute <s_i^k> from {m_ij^k, a_ij^k}
        self.nets['s'] = nn.Sequential(
            MhaLinear(n_heads, n_dims_emsg, n_dims_node_hid),
            SiLU(),
            MhaLinear(n_heads, n_dims_node_hid, n_dims_node_hid),
        )  # per-head aggregated messages

        # final output layer (in case of <n_dims_node_src> and <n_dims_node_dst> do not match)
        if n_dims_node_src != n_dims_node_dst:
            self.nets['o'] = nn.Linear(n_dims_node_src, n_dims_node_dst)
        else:
            self.nets['o'] = nn.Identity()  # use shortcut connection when possible


    def forward(self, graph, node_feats, node_cords, edge_feats, node_masks=None):  # pylint: disable=too-many-arguments
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * node_feats: node features of size Nv x Dv
        * node_cords: node coordinates of size Nv x G x Dc (Dc = 3 for 3D coordinates)
        * edge_feats: edge features of size Ne x De
        * (optional) node_masks: node coordinates' validness masks of size Nv

        Returns:
        * node_feats_out: updated node features of size Nv x Dv
        * node_cords_out: updated node coordinates of size Nv x G x Dc
        """

        # initialization
        device = node_feats.device

        # perform the forward pass
        self.dist_vals = self.dist_vals.to(device)
        with graph.local_scope():
            # initialization
            n_nodes = node_feats.shape[0]
            n_edges = edge_feats.shape[0]
            graph.ndata['f'] = node_feats
            graph.ndata['x'] = node_cords.view(-1, self.n_grps_cord, 3)
            graph.ndata['z'] = graph.ndata['x'][:, self.idx_atom_ca]  # atom CA
            graph.edata['g'] = edge_feats
            graph.ndata['mk'] = node_masks.view(-1) if node_masks is not None else \
                torch.ones((n_nodes), dtype=torch.float32, device=device)

            # calculate relative coordinates
            graph.apply_edges(self.__calc_ecrd)

            # calculate the radial distance for all the edges
            graph.apply_edges(self.__calc_erad)

            # (f_i, f_j, g_ij, r_ij) => m_ij
            graph.apply_edges(self.__calc_emsg)

            # f_i => q_i
            graph.ndata['q'] = self.nets['q'](graph.ndata['f'])

            # (f_j, m_ij) => k_ij
            graph.apply_edges(self.__calc_ekey)

            # (q_i, k_ij) => e_ij
            graph.apply_edges(self.__calc_eatt_raw)

            # e_ij => (e_max_i, a_nrm_i)
            graph.update_all(dgl_fn.copy_e('e', 'e'), dgl_fn.max('e', 'e_max'))
            graph.update_all(self.__calc_attn_norm_message_fn, dgl_fn.sum('e_exp', 'a_nrm'))

            # (e_ij, e_max_i, a_nrm_i) => a_ij
            graph.apply_edges(self.__calc_eatt_fnl)

            # m_ij => c_ij & (x_i, x_j, c_ij, a_ij) => x'_i
            graph.edata['cm'] = split_by_head(self.nets['c'](graph.edata['m']), self.n_heads)
            graph.edata['ca'] = torch.sum(
                graph.edata['cm'] * torch.unsqueeze(graph.edata['a'], dim=-1), dim=1)
            graph.edata['cadx'] = torch.unsqueeze(graph.edata['ca'], dim=-1) * graph.edata['dx']
            #graph.update_all(dgl_fn.copy_e('cadx', 'cadx'), dgl_fn.sum('cadx', 'dx'))
            graph.update_all(dgl_fn.u_mul_e('mk', 'cadx', 'mkcadx'), dgl_fn.sum('mkcadx', 'dx'))
            graph.ndata['xo'] = graph.ndata['x'] + graph.ndata['dx']

            # (m_ij, a_ij) => s_i
            graph.edata['am'] = torch.reshape(
                graph.edata['a'].unsqueeze(dim=2) * split_by_head(graph.edata['m'], self.n_heads),
                [n_edges, -1],
            )
            graph.update_all(dgl_fn.copy_edge('am', 'am'), dgl_fn.sum('am', 'am'))
            if self.merge_fn == 'avg':
                graph.ndata['s'] = torch.mean(
                    split_by_head(self.nets['s'](graph.ndata['am']), self.n_heads), dim=1)
            elif self.merge_fn == 'cat':
                graph.ndata['s'] = self.nets['s'](graph.ndata['am'])
            else:
                raise ValueError('unrecognized merge function: ' + self.merge_fn)

            # (f_i, s_i) => f'_i
            graph.ndata['fo'] = self.nets['o'](graph.ndata['f'] + graph.ndata['s'])

            # fetch the updated node features & delta term of node coordinates
            node_feats_out = graph.ndata['fo']
            node_cords_out = graph.ndata['xo']

        return node_feats_out, node_cords_out


    @classmethod
    def __calc_ecrd(cls, edges):
        """Calculate the edge-wise delta coordinates."""

        outputs = {
            'dx': edges.dst['x'] - edges.src['x'],  # N_e x G x 3
            'dz': edges.dst['z'] - edges.src['z'],  # N_e x 3
        }

        return outputs


    def __calc_erad(self, edges):
        """Calculate the edge-wise radial distance, with optional encodings."""

        mask_tns = edges.src['mk'].unsqueeze(dim=1)
        dist_tns = torch.norm(edges.data['dz'], dim=1, keepdim=True)  # N_e x 1
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


    def __calc_ekey(self, edges):
        """Calculate the edge-wise key vector."""

        inputs = torch.reshape(torch.cat([
            torch.repeat_interleave(edges.src['f'].unsqueeze(dim=1), self.n_heads, dim=1),
            split_by_head(edges.data['m'], self.n_heads),
        ], dim=2), [-1, self.n_heads * (self.n_dims_node_src + self.n_dims_emsg)])
        outputs = self.nets['k'](inputs)

        return {'k': outputs}


    def __calc_eatt_raw(self, edges):
        """Calculate the edge-wise attention coefficients (raw - unnormalized)."""

        inputs = torch.cat([
            split_by_head(edges.dst['q'], self.n_heads),
            split_by_head(edges.data['k'], self.n_heads),
        ], dim=2).view(-1, self.n_heads * (2 * self.n_dims_attn))
        outputs = self.nets['e'](inputs)  # BS x H

        return {'e': outputs}


    @classmethod
    def __calc_attn_norm_message_fn(cls, edges):
        """Message function for calculating attention coefficients' normalization factors."""

        return {'e_exp': torch.exp(edges.data['e'] - edges.dst['e_max'])}


    @classmethod
    def __calc_eatt_fnl(cls, edges):
        """Calculate the edge-wise attention coefficients (final - normalized)."""

        return {'a': torch.exp(edges.data['e'] - edges.dst['e_max']) / edges.dst['a_nrm']}
