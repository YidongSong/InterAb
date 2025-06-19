"""The residue-atom message-passing neural network (MPNN).

Notes:
* <RAMpnn> supports 3D coordinate updates in two possible ways:
  > v1: update node coordinates at each layer. This leads to different relative distance encodings
    to be used at each layer.
  > v2: do not update node coordinates at each layer. This ensures that relative distance encodings
    remain constant throughout the network. Node velocities are accumulated at each layer to update
    node coordinates at the end of forward pass.
"""

import torch
from torch import nn
from torch_geometric.data import Batch

from tfold.tools import ProtStruct
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.modules.ra_mpnn.ra_mpnn_layer import MpnnLayer
from tfold.modules.ra_mpnn.ra_mpnn_layer import RAMpnnLayer


class Mpnn(nn.Module):
    """The message-passing neural network (MPNN) - single node type."""

    def __init__(
            self,
            n_lyrs,  # number of <RAMpnnLayer> layers
            n_dims_nfea,  # number of dimensions in node features
            n_dims_efea=0,  # number of dimensions in edge features
            cord_scale=1.0,  # scaling factor of 3D coordinates (in Angstrom)
            updt_nvlc=True,  # whether to update node velocities (and coordinates)
            updt_nfrc=True,  # whether to update node forces
            version='v1',  # EGCL version (choices: 'v1' / 'v2')
        ):
        """Constructor function."""

        super().__init__()

        # setup configuraations
        self.n_lyrs = n_lyrs
        self.n_dims_nfea = n_dims_nfea
        self.n_dims_efea = n_dims_efea
        self.cord_scale = cord_scale
        self.updt_nvlc = updt_nvlc
        self.updt_nfrc = updt_nfrc
        self.version = version

        # additional configurations
        self.n_dims_emsg = 64  # number of dimensions in edge-wise messages
        self.norm_cord = True  # normalize relative coordinates between nodes (more stable)
        self.rezero_init = True  # use re-zero initialization (helpful for deep networks)
        self.mp_aggr_op = 'add'  # aggregation operation (choices: 'add' / 'mean' / 'max')

        # build the network
        self.net = nn.ModuleList()
        for _ in range(self.n_lyrs):
            self.net.append(MpnnLayer(
                self.n_dims_nfea,
                n_dims_emsg=self.n_dims_emsg,
                n_dims_efea=self.n_dims_efea,
                cord_scale=self.cord_scale,
                updt_nvlc=self.updt_nvlc,
                updt_nfrc=self.updt_nfrc,
                norm_cord=self.norm_cord,
                rezero_init=self.rezero_init,
                mp_aggr_op=self.mp_aggr_op,
                version=self.version,
            ))


    def forward(self, graph, use_checkpoint=False):
        """Perform the forward pass.

        Args:
        * graph: homogeneous graph (torch_geometric.data.Data)
        * use_checkpoint: whether to enable the checkpointing mechanism

        Returns:
        * graph: updated homogeneous graph
        """

        # record initial node coordinates
        ncrd_mat_init = graph.pos.detach().clone()

        # perform the forward pass
        for layer in self.net:
            graph = layer(graph, use_checkpoint=use_checkpoint)

        # replace node coordinates or velocities w/ correct values
        if self.version == 'v1':  # node coordinates => node velocities
            graph.vlc = graph.pos - ncrd_mat_init
        elif self.version == 'v2':  # node velocities => node coordinates
            ncum_vec = torch.ones_like(ncrd_mat_init[:, 0]) \
                if not hasattr(graph, 'cum') else graph.cum
            graph.pos = ncrd_mat_init + ncum_vec.unsqueeze(dim=1) * graph.vlc
        else:
            raise ValueError(f'unrecognized EGCL version: {self.version}')

        return graph


class RAMpnn(nn.Module):
    """The residue-atom message-passing neural network (MPNN)."""

    def __init__(
            self,
            n_lyrs,  # number of <RAMpnnLayer> layers
            n_dims_resd,  # number of dimensions in per-residue embeddings
            n_dims_atom,  # number of dimensions in per-atom embeddings
            n_dims_efea_r2r=0,  # number of dimensions in R2R edge features
            n_dims_efea_r2a=0,  # number of dimensions in R2A edge features
            n_dims_efea_a2a=0,  # number of dimensions in A2A edge features
            n_dims_efea_a2r=0,  # number of dimensions in A2R edge features
            cord_scale=1.0,  # scaling factor of 3D coordinates (in Angstrom)
            updt_nvlc=True,  # whether to update node velocities (and coordinates)
            updt_nfrc=True,  # whether to update node forces
            version='v1',  # EGCL version (choices: 'v1' / 'v2')
        ):
        """Constructor function."""

        super().__init__()

        # setup configuraations
        self.n_lyrs = n_lyrs
        self.n_dims_resd = n_dims_resd
        self.n_dims_atom = n_dims_atom
        self.n_dims_efea_r2r = n_dims_efea_r2r
        self.n_dims_efea_r2a = n_dims_efea_r2a
        self.n_dims_efea_a2a = n_dims_efea_a2a
        self.n_dims_efea_a2r = n_dims_efea_a2r
        self.cord_scale = cord_scale
        self.updt_nvlc = updt_nvlc
        self.updt_nfrc = updt_nfrc
        self.version = version

        # additional configurations
        self.n_dims_emsg = 64  # number of dimensions in edge-wise messages
        self.norm_cord = True  # normalize relative coordinates between nodes (more stable)
        self.rezero_init = True  # use re-zero initialization (helpful for deep networks)
        self.mp_aggr_op = 'add'  # aggregation operation (choices: 'add' / 'mean' / 'max')

        # build the network
        self.net = nn.ModuleList()
        for _ in range(self.n_lyrs):
            self.net.append(RAMpnnLayer(
                self.n_dims_resd, self.n_dims_atom,
                n_dims_emsg=self.n_dims_emsg,
                n_dims_efea_r2r=self.n_dims_efea_r2r,
                n_dims_efea_r2a=self.n_dims_efea_r2a,
                n_dims_efea_a2a=self.n_dims_efea_a2a,
                n_dims_efea_a2r=self.n_dims_efea_a2r,
                cord_scale=self.cord_scale,
                updt_nvlc=self.updt_nvlc,
                updt_nfrc=self.updt_nfrc,
                norm_cord=self.norm_cord,
                rezero_init=self.rezero_init,
                mp_aggr_op=self.mp_aggr_op,
                version=self.version,
            ))


    def forward(self, graph, use_checkpoint=False):
        """Perform the forward pass.

        Args:
        * graph: heterogeneous graph (torch_geometric.data.HeteroData)
        * use_checkpoint: whether to enable the checkpointing mechanism

        Returns:
        * graph: updated heterogeneous graph
        """

        # record initial node coordinates
        rcrd_mat_init = graph['resd'].pos.detach().clone()
        acrd_mat_init = graph['atom'].pos.detach().clone()

        # perform the forward pass
        for layer in self.net:
            graph = layer(graph, use_checkpoint=use_checkpoint)

        # replace node coordinates or velocities w/ correct values
        if self.version == 'v1':  # node coordinates => node velocities
            graph['resd'].vlc = graph['resd'].pos - rcrd_mat_init
            graph['atom'].vlc = graph['atom'].pos - acrd_mat_init
        elif self.version == 'v2':  # node velocities => node coordinates
            rcum_vec = torch.ones_like(rcrd_mat_init[:, 0]) \
                if not hasattr(graph['resd'], 'cum') else graph['resd'].cum
            acum_vec = torch.ones_like(acrd_mat_init[:, 0]) \
                if not hasattr(graph['atom'], 'cum') else graph['atom'].cum
            graph['resd'].pos = rcrd_mat_init + rcum_vec.unsqueeze(dim=1) * graph['resd'].vlc
            graph['atom'].pos = acrd_mat_init + acum_vec.unsqueeze(dim=1) * graph['atom'].vlc
        else:
            raise ValueError(f'unrecognized EGCL version: {self.version}')

        return graph


class RAMpnnMha(nn.Module):
    """The residue-atom message-passing neural network (MPNN) w/ multi-head attention.

    Notes:
    * The main difference between <RAMpnn> and <RAMpnnMha> is that the latter one adopts SE(3)
        invariant updates among all the per-residue node embeddings. This allows fully-connected
        message passing in the graph, even if edge connections correspond to disjoint sub-graphs.
    """

    def __init__(
            self,
            n_lyrs,  # number of <RAMpnnLayer> layers
            n_dims_resd,  # number of dimensions in per-residue embeddings
            n_dims_atom,  # number of dimensions in per-atom embeddings
            n_dims_efea_r2r=0,  # number of dimensions in R2R edge features
            n_dims_efea_r2a=0,  # number of dimensions in R2A edge features
            n_dims_efea_a2a=0,  # number of dimensions in A2A edge features
            n_dims_efea_a2r=0,  # number of dimensions in A2R edge features
            cord_scale=1.0,  # scaling factor of 3D coordinates (in Angstrom)
            updt_nvlc=True,  # whether to update node velocities (and coordinates)
            updt_nfrc=True,  # whether to update node forces
            version='v1',  # EGCL version (choices: 'v1' / 'v2')
        ):
        """Constructor function."""

        super().__init__()

        # setup configuraations
        self.n_lyrs = n_lyrs
        self.n_dims_resd = n_dims_resd
        self.n_dims_atom = n_dims_atom
        self.n_dims_efea_r2r = n_dims_efea_r2r
        self.n_dims_efea_r2a = n_dims_efea_r2a
        self.n_dims_efea_a2a = n_dims_efea_a2a
        self.n_dims_efea_a2r = n_dims_efea_a2r
        self.cord_scale = cord_scale
        self.updt_nvlc = updt_nvlc
        self.updt_nfrc = updt_nfrc
        self.version = version

        # additional configurations
        self.n_dims_emsg = 64  # number of dimensions in edge-wise messages
        self.norm_cord = True  # normalize relative coordinates between nodes (more stable)
        self.rezero_init = True  # use re-zero initialization (helpful for deep networks)
        self.mp_aggr_op = 'add'  # aggregation operation (choices: 'add' / 'mean' / 'max')
        self.n_attn_heads = 8  # number of attention heads

        # build the network
        self.net = nn.ModuleDict()
        for idx_lyr in range(self.n_lyrs):
            self.net[f'ramp-{idx_lyr}'] = RAMpnnLayer(
                self.n_dims_resd, self.n_dims_atom,
                n_dims_emsg=self.n_dims_emsg,
                n_dims_efea_r2r=self.n_dims_efea_r2r,
                n_dims_efea_r2a=self.n_dims_efea_r2a,
                n_dims_efea_a2a=self.n_dims_efea_a2a,
                n_dims_efea_a2r=self.n_dims_efea_a2r,
                cord_scale=self.cord_scale,
                updt_nvlc=self.updt_nvlc,
                updt_nfrc=self.updt_nfrc,
                norm_cord=self.norm_cord,
                rezero_init=self.rezero_init,
                mp_aggr_op=self.mp_aggr_op,
                version=self.version,
            )
            self.net[f'mha-{idx_lyr}'] = nn.MultiheadAttention(
                self.n_dims_resd,
                self.n_attn_heads,
                dropout=0.1,
                batch_first=True,
            )


    def forward(self, graph, use_checkpoint=False):
        """Perform the forward pass.

        Args:
        * graph: heterogeneous graph (torch_geometric.data.HeteroData)
        * use_checkpoint: whether to enable the checkpointing mechanism

        Returns:
        * graph: updated heterogeneous graph

        Notes:
        * If the input graph is batched, then the number of nodes in each sample must be the same.
        """

        # initialization
        n_nodes = graph['resd'].num_nodes

        # record initial node coordinates
        rcrd_mat_init = graph['resd'].pos.detach().clone()
        acrd_mat_init = graph['atom'].pos.detach().clone()

        # determine whether the input graph is batched (no MHA between samples in the batch)
        if isinstance(graph, Batch):
            is_batched = True
            batch_size = graph.num_graphs
        else:
            is_batched = False
            batch_size = 1

        # perform the forward pass
        for idx_lyr in range(self.n_lyrs):
            graph = self.net[f'ramp-{idx_lyr}'](graph, use_checkpoint=use_checkpoint)
            rfea_tns = graph['resd'].x.view(batch_size, -1, self.n_dims_resd)
            rfea_tns_out, _ = self.net[f'mha-{idx_lyr}'](
                rfea_tns, rfea_tns, rfea_tns, need_weights=False)
            graph['resd'].x += rfea_tns_out.reshape(n_nodes, self.n_dims_resd)  # residual updates

        # replace node coordinates or velocities w/ correct values
        if self.version == 'v1':  # node coordinates => node velocities
            graph['resd'].vlc = graph['resd'].pos - rcrd_mat_init
            graph['atom'].vlc = graph['atom'].pos - acrd_mat_init
        elif self.version == 'v2':  # node velocities => node coordinates
            rcum_vec = torch.ones_like(rcrd_mat_init[:, 0]) \
                if not hasattr(graph['resd'], 'cum') else graph['resd'].cum
            acum_vec = torch.ones_like(acrd_mat_init[:, 0]) \
                if not hasattr(graph['atom'], 'cum') else graph['atom'].cum
            graph['resd'].pos = rcrd_mat_init + rcum_vec.unsqueeze(dim=1) * graph['resd'].vlc
            graph['atom'].pos = acrd_mat_init + acum_vec.unsqueeze(dim=1) * graph['atom'].vlc
        else:
            raise ValueError(f'unrecognized EGCL version: {self.version}')

        return graph
