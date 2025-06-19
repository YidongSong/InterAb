"""The residue-atom message-passing neural network (MPNN) layer."""

from torch import nn
from torch.utils.checkpoint import checkpoint

from tfold.modules.ra_mpnn.eq_graph_conv import EqGraphConv
from tfold.modules.ra_mpnn.eq_graph_conv_bg import EqGraphConvBG


class MpnnLayer(nn.Module):
    """The message-passing neural network (MPNN) layer - single node type."""

    def __init__(
            self,
            n_dims_nfea,  # number of dimensions in node features
            n_dims_emsg=64,  # number of dimensions in edge-wise messages
            n_dims_efea=0,  # number of dimensions in edge features
            cord_scale=1.0,  # scaling factor of 3D coordinates (in Angstrom)
            updt_nvlc=True,  # whether to update node velocities
            updt_nfrc=True,  # whether to update node forces
            norm_cord=True,  # whether to normalize relative coordinates between nodes
            rezero_init=True,  # whether to use re-zero initialization
            mp_aggr_op='add',  # aggregation operation (choices: 'add' / 'mean' / 'max')
            version='v1',  # EGCL version (choices: 'v1' / 'v2')
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_nfea = n_dims_nfea
        self.n_dims_emsg = n_dims_emsg
        self.n_dims_efea = n_dims_efea
        self.cord_scale = cord_scale
        self.updt_nvlc = updt_nvlc
        self.updt_nfrc = updt_nfrc
        self.norm_cord = norm_cord
        self.rezero_init = rezero_init
        self.mp_aggr_op = mp_aggr_op
        self.version = version

        # choose the forward function based on the specified EGCL version
        self.forward_fn = self.__forward_v1 if self.version == 'v1' else self.__forward_v2

        # build the network
        self.net = nn.ModuleDict()
        self.net['conv'] = EqGraphConv(
            self.n_dims_nfea,
            n_dims_emsg=n_dims_emsg, n_dims_efea=n_dims_efea,
            cord_scale=self.cord_scale, updt_nvlc=self.updt_nvlc, updt_nfrc=self.updt_nfrc,
            norm_cord=self.norm_cord, rezero_init=self.rezero_init, mp_aggr_op=self.mp_aggr_op,
        )


    def forward(self, graph, use_checkpoint=False):
        """Perform the forward pass.

        Args:
        * graph: homogeneous graph (torch_geometric.data.Data)
        * use_checkpoint: whether to enable the checkpointing mechanism

        Returns:
        * graph: updated homogeneous graph
        """

        # extract essential input tensors
        nfea_mat = graph.x
        ncrd_mat = graph.pos
        eidx_mat = graph.edge_index

        # extract optional input tensors
        nvlc_mat = None if not hasattr(graph, 'vlc') else graph.vlc
        nfrc_mat = None if not hasattr(graph, 'frc') else graph.frc
        ncum_vec = None if not hasattr(graph, 'cum') else graph.cum
        efea_mat = None if self.n_dims_efea == 0 else graph.edge_attr

        # perform the forward pass
        if not (self.training and use_checkpoint):
            nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat = self.forward_fn(
                nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat, ncum_vec, eidx_mat, efea_mat)
        else:
            nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat = checkpoint(
                self.forward_fn,
                nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat, ncum_vec, eidx_mat, efea_mat)

        # pack output tensors into the homogeneous graph
        graph.x = nfea_mat
        graph.pos = ncrd_mat
        graph.vlc = nvlc_mat
        graph.frc = nfrc_mat

        return graph


    def __forward_v1(self, nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat, ncum_vec, eidx_mat, efea_mat):
        """Perform the forward pass - v1.

        Notes:
        * DO NOT use <nvlc_mat> for subsequent loss evaluation.
        """

        nfea_mat, ncrd_mat, _, nfrc_mat = self.net['conv'](
            nfea_mat, ncrd_mat, eidx_mat,
            efea_mat=efea_mat, nfrc_mat=nfrc_mat, ncum_vec=ncum_vec,
        )

        return nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat


    def __forward_v2(self, nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat, ncum_vec, eidx_mat, efea_mat):
        """Perform the forward pass - v2.

        Notes:
        * DO NOT use <ncrd_mat> for subsequent loss evaluation.
        """

        nfea_mat, _, nvlc_mat, nfrc_mat = self.net['conv'](
            nfea_mat, ncrd_mat, eidx_mat,
            efea_mat=efea_mat, nvlc_mat=nvlc_mat, nfrc_mat=nfrc_mat, ncum_vec=ncum_vec,
        )

        return nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat


class RAMpnnLayer(nn.Module):
    """The residue-atom message-passing neural network (MPNN) layer."""

    def __init__(
            self,
            n_dims_resd,  # number of dimensions in per-residue embeddings
            n_dims_atom,  # number of dimensions in per-atom embeddings
            n_dims_emsg=64,  # number of dimensions in edge-wise messages
            n_dims_efea_r2r=0,  # number of dimensions in R2R edge features
            n_dims_efea_r2a=0,  # number of dimensions in R2A edge features
            n_dims_efea_a2a=0,  # number of dimensions in A2A edge features
            n_dims_efea_a2r=0,  # number of dimensions in A2R edge features
            cord_scale=1.0,  # scaling factor of 3D coordinates (in Angstrom)
            updt_nvlc=True,  # whether to update node velocities
            updt_nfrc=True,  # whether to update node forces
            norm_cord=True,  # whether to normalize relative coordinates between nodes
            rezero_init=True,  # whether to use re-zero initialization
            mp_aggr_op='add',  # aggregation operation (choices: 'add' / 'mean' / 'max')
            version='v1',  # EGCL version (choices: 'v1' / 'v2')
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_resd = n_dims_resd
        self.n_dims_atom = n_dims_atom
        self.n_dims_emsg = n_dims_emsg
        self.n_dims_efea_r2r = n_dims_efea_r2r
        self.n_dims_efea_r2a = n_dims_efea_r2a
        self.n_dims_efea_a2a = n_dims_efea_a2a
        self.n_dims_efea_a2r = n_dims_efea_a2r
        self.cord_scale = cord_scale
        self.updt_nvlc = updt_nvlc
        self.updt_nfrc = updt_nfrc
        self.norm_cord = norm_cord
        self.rezero_init = rezero_init
        self.mp_aggr_op = mp_aggr_op
        self.version = version

        # choose the forward function based on the specified EGCL version
        self.forward_fn = self.__forward_v1 if self.version == 'v1' else self.__forward_v2

        # build the network
        self.net = nn.ModuleDict()
        self.net['r2r'] = EqGraphConv(
            self.n_dims_resd,
            n_dims_emsg=n_dims_emsg, n_dims_efea=n_dims_efea_r2r,
            cord_scale=self.cord_scale, updt_nvlc=self.updt_nvlc, updt_nfrc=self.updt_nfrc,
            norm_cord=self.norm_cord, rezero_init=self.rezero_init, mp_aggr_op=self.mp_aggr_op,
        )
        self.net['r2a'] = EqGraphConvBG(
            self.n_dims_resd, self.n_dims_atom,
            n_dims_emsg=n_dims_emsg, n_dims_efea=n_dims_efea_r2a,
            cord_scale=self.cord_scale, updt_nvlc=self.updt_nvlc, updt_nfrc=self.updt_nfrc,
            norm_cord=self.norm_cord, rezero_init=self.rezero_init, mp_aggr_op=self.mp_aggr_op,
        )
        self.net['a2a'] = EqGraphConv(
            self.n_dims_atom,
            n_dims_emsg=n_dims_emsg, n_dims_efea=n_dims_efea_a2a,
            cord_scale=self.cord_scale, updt_nvlc=self.updt_nvlc, updt_nfrc=self.updt_nfrc,
            norm_cord=self.norm_cord, rezero_init=self.rezero_init, mp_aggr_op=self.mp_aggr_op,
        )
        self.net['a2r'] = EqGraphConvBG(
            self.n_dims_atom, self.n_dims_resd,
            n_dims_emsg=n_dims_emsg, n_dims_efea=n_dims_efea_a2r,
            cord_scale=self.cord_scale, updt_nvlc=self.updt_nvlc, updt_nfrc=self.updt_nfrc,
            norm_cord=self.norm_cord, rezero_init=self.rezero_init, mp_aggr_op=self.mp_aggr_op,
        )


    def forward(self, graph, use_checkpoint=False):
        """Perform the forward pass.

        Args:
        * graph: heterogeneous graph (torch_geometric.data.HeteroData)
        * use_checkpoint: whether to enable the checkpointing mechanism

        Returns:
        * graph: updated heterogeneous graph
        """

        # extract essential input tensors
        rfea_mat = graph['resd'].x
        rcrd_mat = graph['resd'].pos
        afea_mat = graph['atom'].x
        acrd_mat = graph['atom'].pos
        eidx_mat_r2r = graph['resd', 'link', 'resd'].edge_index
        eidx_mat_r2a = graph['resd', 'link', 'atom'].edge_index
        eidx_mat_a2a = graph['atom', 'link', 'atom'].edge_index
        eidx_mat_a2r = graph['atom', 'link', 'resd'].edge_index

        # extract optional input tensors
        rvlc_mat = None if not hasattr(graph['resd'], 'vlc') else graph['resd'].vlc
        rfrc_mat = None if not hasattr(graph['resd'], 'frc') else graph['resd'].frc
        rcum_vec = None if not hasattr(graph['resd'], 'cum') else graph['resd'].cum
        avlc_mat = None if not hasattr(graph['atom'], 'vlc') else graph['atom'].vlc
        afrc_mat = None if not hasattr(graph['atom'], 'frc') else graph['atom'].frc
        acum_vec = None if not hasattr(graph['atom'], 'cum') else graph['atom'].cum
        efea_mat_r2r = None if self.n_dims_efea_r2r == 0 else graph['resd', 'link', 'resd'].edge_attr
        efea_mat_r2a = None if self.n_dims_efea_r2a == 0 else graph['resd', 'link', 'atom'].edge_attr
        efea_mat_a2a = None if self.n_dims_efea_a2a == 0 else graph['atom', 'link', 'atom'].edge_attr
        efea_mat_a2r = None if self.n_dims_efea_a2r == 0 else graph['atom', 'link', 'resd'].edge_attr

        # perform the forward pass
        if not (self.training and use_checkpoint):
            rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, afea_mat, acrd_mat, avlc_mat, afrc_mat = \
                self.forward_fn(
                    rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, rcum_vec,
                    afea_mat, acrd_mat, avlc_mat, afrc_mat, acum_vec,
                    eidx_mat_r2r, efea_mat_r2r, eidx_mat_r2a, efea_mat_r2a,
                    eidx_mat_a2a, efea_mat_a2a, eidx_mat_a2r, efea_mat_a2r,
                )
        else:
            rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, afea_mat, acrd_mat, avlc_mat, afrc_mat = \
                checkpoint(
                    self.forward_fn,
                    rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, rcum_vec,
                    afea_mat, acrd_mat, avlc_mat, afrc_mat, acum_vec,
                    eidx_mat_r2r, efea_mat_r2r, eidx_mat_r2a, efea_mat_r2a,
                    eidx_mat_a2a, efea_mat_a2a, eidx_mat_a2r, efea_mat_a2r,
                )

        # pack output tensors into the heterogeneous graph
        graph['resd'].x = rfea_mat
        graph['resd'].pos = rcrd_mat
        graph['resd'].vlc = rvlc_mat
        graph['resd'].frc = rfrc_mat
        graph['atom'].x = afea_mat
        graph['atom'].pos = acrd_mat
        graph['atom'].vlc = avlc_mat
        graph['atom'].frc = afrc_mat

        return graph


    def __forward_v1(
            self,
            rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, rcum_vec,
            afea_mat, acrd_mat, avlc_mat, afrc_mat, acum_vec,
            eidx_mat_r2r, efea_mat_r2r, eidx_mat_r2a, efea_mat_r2a,
            eidx_mat_a2a, efea_mat_a2a, eidx_mat_a2r, efea_mat_a2r,
        ):
        """Perform the forward pass - v1.

        Notes:
        * DO NOT use <rvlc_mat> & <avlc_mat> for subsequent loss evaluation.
        """

        # residue => residue
        rfea_mat, rcrd_mat, _, rfrc_mat = self.net['r2r'](
            rfea_mat, rcrd_mat, eidx_mat_r2r,
            efea_mat=efea_mat_r2r, nfrc_mat=rfrc_mat, ncum_vec=rcum_vec,
        )

        # residue => atom
        afea_mat, acrd_mat, _, afrc_mat = self.net['r2a'](
            rfea_mat, afea_mat, rcrd_mat, acrd_mat, eidx_mat_r2a,
            efea_mat=efea_mat_r2a, nfrc_mat_sec=afrc_mat, ncum_vec_sec=acum_vec,
        )

        # atom => atom
        afea_mat, acrd_mat, _, afrc_mat = self.net['a2a'](
            afea_mat, acrd_mat, eidx_mat_a2a,
            efea_mat=efea_mat_a2a, nfrc_mat=afrc_mat, ncum_vec=acum_vec,
        )

        # atom => residue
        rfea_mat, rcrd_mat, _, rfrc_mat = self.net['a2r'](
            afea_mat, rfea_mat, acrd_mat, rcrd_mat, eidx_mat_a2r,
            efea_mat=efea_mat_a2r, nfrc_mat_sec=rfrc_mat, ncum_vec_sec=rcum_vec,
        )

        return rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, afea_mat, acrd_mat, avlc_mat, afrc_mat


    def __forward_v2(
            self,
            rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, rcum_vec,
            afea_mat, acrd_mat, avlc_mat, afrc_mat, acum_vec,
            eidx_mat_r2r, efea_mat_r2r, eidx_mat_r2a, efea_mat_r2a,
            eidx_mat_a2a, efea_mat_a2a, eidx_mat_a2r, efea_mat_a2r,
        ):
        """Perform the forward pass - v2.

        Notes:
        * DO NOT use <rcrd_mat> & <acrd_mat> for subsequent loss evaluation.
        """

        # residue => residue
        rfea_mat, _, rvlc_mat, rfrc_mat = self.net['r2r'](
            rfea_mat, rcrd_mat, eidx_mat_r2r,
            efea_mat=efea_mat_r2r, nvlc_mat=rvlc_mat, nfrc_mat=rfrc_mat, ncum_vec=rcum_vec,
        )

        # residue => atom
        afea_mat, _, avlc_mat, afrc_mat = self.net['r2a'](
            rfea_mat, afea_mat, rcrd_mat, acrd_mat, eidx_mat_r2a,
            efea_mat=efea_mat_r2a, nvlc_mat_sec=avlc_mat,
            nfrc_mat_sec=afrc_mat, ncum_vec_sec=acum_vec,
        )

        # atom => atom
        afea_mat, _, avlc_mat, afrc_mat = self.net['a2a'](
            afea_mat, acrd_mat, eidx_mat_a2a,
            efea_mat=efea_mat_a2a, nvlc_mat=avlc_mat, nfrc_mat=afrc_mat, ncum_vec=acum_vec,
        )

        # atom => residue
        rfea_mat, _, rvlc_mat, rfrc_mat = self.net['a2r'](
            afea_mat, rfea_mat, acrd_mat, rcrd_mat, eidx_mat_a2r,
            efea_mat=efea_mat_a2r, nvlc_mat_sec=rvlc_mat,
            nfrc_mat_sec=rfrc_mat, ncum_vec_sec=rcum_vec,
        )

        return rfea_mat, rcrd_mat, rvlc_mat, rfrc_mat, afea_mat, acrd_mat, avlc_mat, afrc_mat
