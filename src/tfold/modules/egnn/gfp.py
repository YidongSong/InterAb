"""Geometric frame perceptron (GFP).

Requirements:
* At least 3 atoms must be associated with each node (for constructing the reference frame).

For comparison, two types of edge-wise messages are constructed:
1. (h_i, h_j, g_ij, E_dist(d_ij)) => m_ij
2. (h_i, h_j, g_ij, E_frcd(R_i, t_i, X_i, R_j, t_j, X_j)) => m_ij
where E_dist() is the distance encoder, and E_frcd() is the frame & coordinate encoder.

Notations:
* h_i: node features associated with node (i)
* g_ij: edge features associated with edge (i, j)
* d_ij: radial distance (possibly encoded) between node (i) and (j)
* R_i & t_i: reference frame associated with node (i)
* X_i: 3D coordinates associated with node (i)
"""

import torch
from torch import nn
import dgl
import dgl.function as dgl_fn

from tfold.utils import split_by_head
from tfold.utils import calc_rot_n_tsl_batch
from tfold.modules.common import SiLU
from tfold.modules.common import MhaLinear


class GFP(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Geometric frame perceptron (GFP)."""

    def __init__(
            self,
            n_dims_nfea,  # number of dimensions in node features
            n_dims_efea,  # number of dimensions in edge features
            n_grps_cord,  # number of groups of 3D coordinates (per node)
            encd_type='frcd',  # encoding type (choices: 'dist-s' / 'dist-m' / 'frcd')
            n_dims_emsg=32,  # number of dimensions in edge-wise messages
            dist_encoder=None,  # distance encoder
            frcd_encoder=None,  # frame & coordinate encoder
            n_heads=8,  # number of attention heads
            n_dims_attn=32,  # number of dimensions in attention embeddings
            updt_cord=False,  # whether to update 3D coordinates
            merge_fn='avg',   # merge function for hidden node embeddings
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        super().__init__()

        # initialization
        self.n_dims_nfea = n_dims_nfea
        self.n_dims_efea = n_dims_efea
        self.n_grps_cord = n_grps_cord
        self.encd_type = encd_type
        self.n_dims_emsg = n_dims_emsg
        self.dist_encoder = dist_encoder
        self.frcd_encoder = frcd_encoder
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn
        self.updt_cord = updt_cord
        self.merge_fn = merge_fn

        # determine the number of dimensions in encoding vectors
        if self.encd_type == 'dist-s':
            assert self.dist_encoder is not None
            self.n_dims_encd = self.dist_encoder.n_dims
        elif self.encd_type == 'dist-m':
            assert self.dist_encoder is not None
            self.n_dims_encd = self.n_grps_cord * self.dist_encoder.n_dims
        elif self.encd_type == 'frcd':
            assert self.frcd_encoder is not None
            self.n_dims_encd = self.frcd_encoder.n_dims
        else:
            raise ValueError(f'unrecognized encoding type: {self.encd_type}')

        # additional configurations
        self.eps = 1e-6
        self.idx_atom_pvt = 1  # index to the pivot atom (CA)
        self.n_dims_nhid = self.n_dims_nfea \
            if self.merge_fn == 'avg' else self.n_dims_nfea // self.n_heads
        self.n_dims_ehid = 2 * self.n_dims_nfea + self.n_dims_efea + self.n_dims_encd

        # sub-network: edge-wise messages
        self.net = nn.ModuleDict()
        self.net['m'] = nn.Sequential(
            nn.Linear(self.n_dims_ehid, self.n_heads * self.n_dims_emsg),
            SiLU(),
            MhaLinear(self.n_heads, self.n_dims_emsg, self.n_dims_emsg),
            SiLU(),
        )

        # sub-network: query & key embeddings
        self.net['q'] = nn.Sequential(
            nn.Linear(self.n_dims_nfea, self.n_heads * self.n_dims_attn),
            nn.ELU(),
            MhaLinear(self.n_heads, self.n_dims_attn, self.n_dims_attn),
        )
        self.net['k'] = nn.Sequential(
            MhaLinear(self.n_heads, self.n_dims_nfea + self.n_dims_emsg, self.n_dims_attn),
            nn.ELU(),
            MhaLinear(self.n_heads, self.n_dims_attn, self.n_dims_attn),
        )

        # sub-network: attention weights
        self.net['e'] = nn.Sequential(
            MhaLinear(self.n_heads, 2 * self.n_dims_attn, 1),
            nn.LeakyReLU(),
        )

        # sub-network: update node features
        self.net['s'] = nn.Sequential(
            MhaLinear(self.n_heads, self.n_dims_emsg, self.n_dims_nhid),
            SiLU(),
            MhaLinear(self.n_heads, self.n_dims_nhid, self.n_dims_nhid),
        )

        # sub-network: update node coordinates
        if self.updt_cord:
            self.net['c'] = nn.Sequential(
                MhaLinear(self.n_heads, self.n_dims_emsg, self.n_dims_emsg),
                SiLU(),
                MhaLinear(self.n_heads, self.n_dims_emsg, self.n_grps_cord),
            )


    def forward(self, graph, node_feats, node_cords, edge_feats, node_masks=None):  # pylint: disable=too-many-arguments
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * node_feats: node features of size Nv x Dv
        * node_cords: node coordinates of size Nv x G x 3
        * edge_feats: edge features of size Ne x De
        * (optional) node_masks: node coordinates' validness masks of size Nv

        Returns:
        * node_feats_out: updated node features of size Nv x Dv
        * node_cords_out: updated node coordinates of size Nv x G x 3
        """

        # initialization
        device = node_feats.device
        n_nodes = node_feats.shape[0]
        n_edges = edge_feats.shape[0]

        # perform the forward pass
        with graph.local_scope():
            # initialization
            graph.ndata['f'] = node_feats
            graph.ndata['x'] = node_cords
            graph.edata['g'] = edge_feats
            graph.ndata['v'] = node_masks.view(-1) if node_masks is not None else \
                torch.ones((n_nodes), dtype=torch.float32, device=device)  # validness masks

            # build edge-wise messages
            graph.apply_edges(self.__calc_emsg)

            # build query & key embeddings
            graph.ndata['q'] = self.net['q'](graph.ndata['f'])
            graph.apply_edges(self.__calc_ekey)

            # build multi-head attention coefficients
            graph.apply_edges(self.__calc_eatt)
            graph.edata['a'] = dgl.nn.functional.edge_softmax(graph, graph.edata['e'])

            # update node features
            graph.edata['am'] = torch.reshape(
                graph.edata['a'] * split_by_head(graph.edata['m'], self.n_heads), [n_edges, -1])
            graph.update_all(dgl_fn.copy_edge('am', 'am'), dgl_fn.sum('am', 'am'))
            graph.apply_nodes(self.__calc_nhid)
            graph.ndata['fo'] = graph.ndata['f'] + graph.ndata['df']

            # (optional) update node coordinates
            if self.updt_cord:
                graph.apply_edges(dgl_fn.u_sub_v('x', 'x', 'dx'))
                graph.edata['cm'] = split_by_head(self.net['c'](graph.edata['m']), self.n_heads)
                graph.edata['cmdx'] = graph.edata['dx'] * \
                    torch.sum(graph.edata['cm'] * graph.edata['a'], dim=1).unsqueeze(dim=2)
                graph.update_all(dgl_fn.u_mul_e('v', 'cmdx', 'vcmdx'), dgl_fn.sum('vcmdx', 'dx'))
                graph.ndata['xo'] = graph.ndata['x'] + graph.ndata['dx']

            # obtain updated node features & coordinates
            node_feats_out = graph.ndata['fo']
            node_cords_out = graph.ndata['xo'] if self.updt_cord else node_cords

        return node_feats_out, node_cords_out


    def __calc_emsg(self, edges):
        """Calculate edge-wise messages."""

        # initialization
        n_edges = edges.batch_size()

        # calculating encoding vectors for all the edges
        emsk_vec = edges.src['v'] * edges.dst['v']
        if self.encd_type == 'dist-s':
            cord_mat_src = edges.src['x'][:, self.idx_atom_pvt]
            cord_mat_dst = edges.dst['x'][:, self.idx_atom_pvt]
            dist_vec = torch.norm(cord_mat_dst - cord_mat_src, dim=1)
            encd_mat = emsk_vec.unsqueeze(dim=1) * self.dist_encoder.run(dist_vec)
        elif self.encd_type == 'dist-m':
            cord_mat_src = edges.src['x'].view(-1, 3)  # (Nv x G) x 3
            cord_mat_dst = edges.dst['x'].view(-1, 3)
            dist_vec = torch.norm(cord_mat_dst - cord_mat_src, dim=1)  # (Nv x G)
            encd_mat = emsk_vec.unsqueeze(dim=1) * self.dist_encoder.run(dist_vec).view(n_edges, -1)
        elif self.encd_type == 'frcd':
            rota_tns_src, trsl_mat_src = calc_rot_n_tsl_batch(edges.src['x'][:, :3])
            rota_tns_dst, trsl_mat_dst = calc_rot_n_tsl_batch(edges.dst['x'][:, :3])
            encd_mat = emsk_vec.unsqueeze(dim=1) * self.frcd_encoder.run(
                rota_tns_src, trsl_mat_src, edges.src['x'],
                rota_tns_dst, trsl_mat_dst, edges.dst['x'],
            )
        else:
            raise ValueError(f'unrecognized encoding type: {self.encd_type}')

        # concatenate node features, edge features, and encoding vectors together
        ehid_mat = torch.cat(
            [edges.src['f'], edges.dst['f'], edges.data['g'], encd_mat], dim=1)

        return {'m': self.net['m'](ehid_mat)}


    def __calc_ekey(self, edges):
        """Calculate edge-wise key embeddings."""

        n_edges = edges.batch_size()
        ehid_mat = torch.reshape(torch.cat([
            edges.src['f'].unsqueeze(dim=1).repeat(1, self.n_heads, 1),
            split_by_head(edges.data['m'], self.n_heads),
        ], dim=2), [n_edges, self.n_heads * (self.n_dims_nfea + self.n_dims_emsg)])

        return {'k': self.net['k'](ehid_mat)}


    def __calc_eatt(self, edges):
        """Calculate edge-wise attention coefficients (unnormalized)."""

        n_edges = edges.batch_size()
        ehid_mat = torch.cat([
            split_by_head(edges.dst['q'], self.n_heads),
            split_by_head(edges.data['k'], self.n_heads),
        ], dim=2).view(n_edges, self.n_heads * (2 * self.n_dims_attn))

        return {'e': self.net['e'](ehid_mat).unsqueeze(dim=2)}


    def __calc_nhid(self, nodes):
        """Calculate hidden node embeddings (before the residual connection)."""

        if self.merge_fn == 'avg':
            nhid_mat = torch.mean(
                split_by_head(self.net['s'](nodes.data['am']), self.n_heads), dim=1)
        elif self.merge_fn == 'cat':
            nhid_mat = self.net['s'](nodes.data['am'])
        else:
            raise ValueError('unrecognized merge function: ' + self.merge_fn)

        return {'df': nhid_mat}
