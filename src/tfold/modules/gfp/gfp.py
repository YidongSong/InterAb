"""Geometric frame perceptron (GFP).

Requirements:
* At least 3 atoms must be associated with each node (for constructing the reference frame).

Notations:
* h_i: node features associated with node (i)
* g_ij: edge features associated with edge (i, j)
* X_i: 3D coordinates associated with node (i)
* R_i & t_i: reference frame associated with node (i)
"""

import torch
from torch import nn
import dgl
import dgl.function as dgl_fn

from tfold.utils import split_by_head
from tfold.utils import calc_rot_n_tsl_batch
from tfold.tools import DistEncoder
from tfold.tools import FrcdEncoder
from tfold.modules.common import SiLU
from tfold.modules.common import MhaLinear
from tfold.modules.gfp.cp_net import CPNetV1
from tfold.modules.gfp.cp_net import CPNetV2
from tfold.modules.gfp.cp_net import CPNetV3


class GFP(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Geometric frame perceptron (GFP)."""

    def __init__(
            self,
            n_dims_nfea,  # number of dimensions in node features
            n_dims_efea,  # number of dimensions in edge features
            n_grps_cord,  # number of groups of 3D coordinates (per node)
            nfcd_type='frcd-s',  # encoding type for node features & absolute coordinates
            efcd_type='frcd-s',  # encoding type for edge features & relative coordinates
            n_dims_emsg=32,  # number of dimensions in edge-wise messages
            n_heads=8,  # number of attention heads
            n_dims_attn=32,  # number of dimensions in attention embeddings
            updt_cord=True,  # whether to update 3D coordinates
            merge_fn='avg',  # merge function for hidden node embeddings
            cp_net_ver='v1',  # <CPNetVx> version
        ):  # pylint: disable=too-many-arguments,too-many-branches,too-many-statements
        """Constructor function."""

        super().__init__()

        # initialization
        self.n_dims_nfea = n_dims_nfea
        self.n_dims_efea = n_dims_efea
        self.n_grps_cord = n_grps_cord
        self.nfcd_type = nfcd_type
        self.efcd_type = efcd_type
        self.n_dims_emsg = n_dims_emsg
        self.n_heads = n_heads
        self.n_dims_attn = n_dims_attn
        self.updt_cord = updt_cord
        self.merge_fn = merge_fn
        self.cp_net_ver = cp_net_ver

        # additional configurations
        self.idx_atom_pvt = 1  # index to the pivot atom (CA)
        self.n_grps_cord_fram = 4  # reference frames
        self.n_grps_cord_proj = 2 * self.n_grps_cord  # projected point coordinates
        self.cp_net_dict = {'v1': CPNetV1, 'v2': CPNetV2, 'v3': CPNetV3}
        self.cp_net = self.cp_net_dict[self.cp_net_ver]

        # encoder for node features & absolute coordinates
        self.dist_encoder = DistEncoder()  # needed by all the encoding methods
        if self.nfcd_type == 'none':
            self.n_dims_nfcd = 0
        elif self.nfcd_type == 'frcd-o':
            self.frcd_encoder_nfcd = FrcdEncoder(self.dist_encoder, self.n_grps_cord)
            self.n_dims_nfcd = self.frcd_encoder_nfcd.n_dims
        elif self.nfcd_type in ['frcd-s', 'frcd-d']:
            self.frcd_encoder_nfcd = FrcdEncoder(self.dist_encoder, self.n_grps_cord_proj)
            self.n_dims_nfcd = self.frcd_encoder_nfcd.n_dims
        else:
            raise ValueError(f'unrecognized node feature encoding type: {self.nfcd_type}')

        # encoder for edge features & relative coordinates
        if self.efcd_type == 'none':
            self.n_dims_efcd = 0
        elif self.efcd_type == 'dist-s':
            self.n_dims_efcd = self.dist_encoder.n_dims
        elif self.efcd_type == 'dist-m':
            self.n_dims_efcd = self.n_grps_cord * self.dist_encoder.n_dims
        elif self.efcd_type == 'frcd-o':
            self.frcd_encoder_efcd = FrcdEncoder(self.dist_encoder, self.n_grps_cord)
            self.n_dims_efcd = self.frcd_encoder_efcd.n_dims
        elif self.efcd_type in ['frcd-s', 'frcd-d']:
            self.frcd_encoder_efcd = FrcdEncoder(self.dist_encoder, self.n_grps_cord_proj)
            self.n_dims_efcd = self.frcd_encoder_efcd.n_dims
        else:
            raise ValueError(f'unrecognized edge feature encoding type: {self.efcd_type}')

        # additional configurations
        self.n_dims_ehid = \
            2 * self.n_dims_nfea + self.n_dims_efea + 2 * self.n_dims_nfcd + self.n_dims_efcd
        self.n_dims_nhid = \
            self.n_dims_nfea if self.merge_fn == 'avg' else (self.n_dims_nfea // self.n_heads)

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

        # sub-network: project 3D coordinates
        if (self.nfcd_type == 'frcd-d') or (self.efcd_type == 'frcd-d'):
            self.net['f'] = self.cp_net(self.n_grps_cord, self.n_grps_cord_fram)
        if (self.nfcd_type in ['frcd-s', 'frcd-d']) or (self.efcd_type in ['frcd-s', 'frcd-d']):
            self.net['p'] = self.cp_net(self.n_grps_cord, self.n_grps_cord_proj)


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
            graph.ndata['v'] = node_masks.to(torch.float32) if node_masks is not None \
                else torch.ones((n_nodes), dtype=torch.float32, device=device)

            # build encoding vectors from node & edge features
            graph.apply_nodes(self.__calc_nfcd)
            graph.apply_edges(self.__calc_efcd)

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


    def __calc_nfcd(self, nodes):  # pylint: disable=too-many-locals
        """Calculate encoding vectors from node features & absolute coordinates."""

        # initialization
        n_nodes = nodes.batch_size()
        dtype = nodes.data['f'].dtype
        device = nodes.data['f'].device

        # calculating encoding vectors for all the nodes
        nmsk_vec = nodes.data['v']
        if self.nfcd_type == 'none':
            encd_mat = torch.zeros((n_nodes, 0), dtype=dtype, device=device)
        elif self.nfcd_type in ['frcd-o', 'frcd-s', 'frcd-d']:
            # fix all-zero 3D coordinates (to avoid invalid rotation matrices)
            cord_tns_base = self.__fix_allzero_cords(nodes.data['x'], nodes.data['v'])

            # obtain 3D coordinates for reference frames
            if self.nfcd_type in ['frcd-o', 'frcd-s']:
                cord_tns_fram = cord_tns_base
            elif self.nfcd_type == 'frcd-d':
                cord_tns_fram = self.net['f'](cord_tns_base)
            else:
                raise ValueError(f'unrecognized node feature encoding type: {self.nfcd_type}')

            # obtain 3D coordinates for projected point coordinates
            if self.nfcd_type == 'frcd-o':
                cord_tns_proj = cord_tns_base
            elif self.nfcd_type in ['frcd-s', 'frcd-d']:
                cord_tns_proj = self.net['p'](cord_tns_base)
            else:
                raise ValueError(f'unrecognized node feature encoding type: {self.nfcd_type}')

            # build reference frames from 3D coordinates
            rota_tns_src, trsl_mat_src = calc_rot_n_tsl_batch(cord_tns_fram[:, 0:3])
            rota_tns_dst, trsl_mat_dst = calc_rot_n_tsl_batch(cord_tns_fram[:, 1:4])

            # generate encodings for reference frames & projected point coordinates
            cord_tns_dst_proj = cord_tns_proj - torch.mean(cord_tns_proj, dim=1, keepdim=True)
            cord_tns_src_proj = torch.zeros_like(cord_tns_dst_proj)
            encd_mat = nmsk_vec.unsqueeze(dim=1) * self.frcd_encoder_nfcd.run(
                rota_tns_src, trsl_mat_src, cord_tns_src_proj,
                rota_tns_dst, trsl_mat_dst, cord_tns_dst_proj,
            )
        else:
            raise ValueError(f'unrecognized edge feature encoding type: {self.efcd_type}')

        return {'fe': encd_mat}


    def __calc_efcd(self, edges):  # pylint: disable=too-many-locals
        """Calculate encoding vectors from edge features & relative coordinates."""

        # initialization
        n_edges = edges.batch_size()
        dtype = edges.data['g'].dtype
        device = edges.data['g'].device

        # calculating encoding vectors for all the edges
        emsk_vec = edges.src['v'] * edges.dst['v']
        if self.efcd_type == 'none':
            encd_mat = torch.zeros((n_edges, 0), dtype=dtype, device=device)
        elif self.efcd_type == 'dist-s':
            cord_mat_src = edges.src['x'][:, self.idx_atom_pvt]
            cord_mat_dst = edges.dst['x'][:, self.idx_atom_pvt]
            dist_vec = torch.norm(cord_mat_dst - cord_mat_src, dim=1)
            encd_mat = emsk_vec.unsqueeze(dim=1) * self.dist_encoder.run(dist_vec)
        elif self.efcd_type == 'dist-m':
            cord_mat_src = edges.src['x'].view(-1, 3)  # (Nv x G) x 3
            cord_mat_dst = edges.dst['x'].view(-1, 3)
            dist_vec = torch.norm(cord_mat_dst - cord_mat_src, dim=1)  # (Nv x G)
            encd_mat = emsk_vec.unsqueeze(dim=1) * self.dist_encoder.run(dist_vec).view(n_edges, -1)
        elif self.efcd_type in ['frcd-o', 'frcd-s', 'frcd-d']:
            # fix all-zero 3D coordinates (to avoid invalid rotation matrices)
            cord_tns_src_base = self.__fix_allzero_cords(edges.src['x'], edges.src['v'])
            cord_tns_dst_base = self.__fix_allzero_cords(edges.dst['x'], edges.dst['v'])

            # obtain 3D coordinates for reference frames
            if self.efcd_type in ['frcd-o', 'frcd-s']:
                cord_tns_src_fram = cord_tns_src_base
                cord_tns_dst_fram = cord_tns_dst_base
            elif self.efcd_type == 'frcd-d':
                cord_tns_src_fram = self.net['f'](cord_tns_src_base)
                cord_tns_dst_fram = self.net['f'](cord_tns_dst_base)
            else:
                raise ValueError(f'unrecognized edge feature encoding type: {self.efcd_type}')

            # obtain 3D coordinates for projected point coordinates
            if self.efcd_type == 'frcd-o':
                cord_tns_src_proj = cord_tns_src_base
                cord_tns_dst_proj = cord_tns_dst_base
            elif self.efcd_type in ['frcd-s', 'frcd-d']:
                cord_tns_src_proj = self.net['p'](cord_tns_src_base)
                cord_tns_dst_proj = self.net['p'](cord_tns_dst_base)
            else:
                raise ValueError(f'unrecognized edge feature encoding type: {self.efcd_type}')

            # build reference frames from 3D coordinates
            rota_tns_src, trsl_mat_src = calc_rot_n_tsl_batch(cord_tns_src_fram[:, :3])
            rota_tns_dst, trsl_mat_dst = calc_rot_n_tsl_batch(cord_tns_dst_fram[:, :3])

            # generate encodings for reference frames & projected point coordinates
            encd_mat = emsk_vec.unsqueeze(dim=1) * self.frcd_encoder_efcd.run(
                rota_tns_src, trsl_mat_src, cord_tns_src_proj,
                rota_tns_dst, trsl_mat_dst, cord_tns_dst_proj,
            )
        else:
            raise ValueError(f'unrecognized edge feature encoding type: {self.efcd_type}')

        return {'ge': encd_mat}


    def __calc_emsg(self, edges):
        """Calculate edge-wise messages."""

        # concatenate node features, edge features, and encoding vectors together
        ehid_mat = torch.cat([
            edges.src['f'], edges.dst['f'], edges.data['g'],
            edges.src['fe'], edges.dst['fe'], edges.data['ge'],
        ], dim=1)

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


    def __fix_allzero_cords(self, cord_tns, mask_vec):
        """Fix all-zero 3D coordinates (to avoid invalid rotation matrices)."""

        return cord_tns + (1 - mask_vec).view(-1, 1, 1) * torch.randn_like(cord_tns)
