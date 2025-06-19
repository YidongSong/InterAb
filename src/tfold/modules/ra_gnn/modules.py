"""Basic modules for iterative updates between residues and atoms."""

import torch
from torch import nn

from tfold.tools import DistEncoder
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.modules.common.ffn import FFN
from tfold.modules.ra_gnn.utils import ds2sp
from tfold.modules.ra_gnn.egcl import EGCL


class Resd2Atom(nn.Module):
    """The residue-to-atom network."""

    def __init__(
            self,
            n_dims_afea,  # number of dimensions in per-atom features
            n_dims_rfea,  # number of dimensions in per-residue features
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_afea = n_dims_afea
        self.n_dims_rfea = n_dims_rfea

        # build the network
        self.net = nn.ModuleDict()
        self.net['linear'] = nn.Linear(self.n_dims_afea + self.n_dims_rfea, self.n_dims_afea)
        self.net['ffn'] = FFN(self.n_dims_afea)


    def forward(self, afea_mat, rfea_mat, ridx_vec):
        """Perform the forward pass.

        Args:
        * afea_mat: per-atom features of size Na x Da
        * rfea_mat: per-residue features of size Nr x Dr
        * idxs_vec: per-atom residue indices of size Na

        Returns:
        * afea_mat: per-atom features of size Na x Da
        """

        afea_mat_hid = torch.cat([afea_mat, torch.index_select(rfea_mat, 0, ridx_vec)], dim=1)
        afea_mat_hid = self.net['linear'](afea_mat_hid)
        afea_mat = afea_mat + self.net['ffn'](afea_mat_hid)

        return afea_mat


class Atom2Atom(nn.Module):
    """The atom-to-atom network."""

    def __init__(
            self,
            n_dims_afea,  # number of dimensions in per-atom features
            n_dims_apfe,  # number of dimensions in per-atom-pair features
            n_dims_emsg=32,  # number of dimensions in hidden edge messages
            dist_encoder=None,  # distance encoder
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_afea = n_dims_afea
        self.n_dims_apfe = n_dims_apfe
        self.n_dims_emsg = n_dims_emsg
        self.dist_encoder = dist_encoder if dist_encoder is not None else DistEncoder()

        # build the network
        self.net = nn.ModuleDict()
        self.net['egcl'] = EGCL(
            self.n_dims_afea,
            self.n_dims_apfe,
            n_dims_emsg=self.n_dims_emsg,
            dist_encoder=self.dist_encoder,
        )
        self.net['ffn'] = FFN(self.n_dims_afea)


    def forward(self, graph, afea_mat, apfe_mat, acrd_mat, aumk_vec):  # pylint: disable=too-many-arguments
        """Perform the forward pass.

        Args:
        * graph: DGL graph
        * afea_mat: per-atom features of size Na x Da
        * apfe_mat: per-atom-pair features of size Nap x Dap
        * acrd_mat: per-atom 3D coordinates of size Na x 3
        * aumk_vec: per-atom 3D coordinates' update-or-not masks of size Na

        Returns:
        * afea_mat: updated per-atom features of size Na x Da
        * acrd_mat: updated per-atom 3D coordinates of size Na x 3
        """

        # initialization
        avmk_vec = torch.ones_like(aumk_vec)  # all the 3D coordinates are valid

        # perform the forward pass
        afea_mat_hid, acrd_mat = self.net['egcl'](
            graph, afea_mat, apfe_mat, acrd_mat, avmk_vec, aumk_vec)
        afea_mat = afea_mat + self.net['ffn'](afea_mat_hid)

        return afea_mat, acrd_mat


class Atom2Resd(nn.Module):
    """The atom-to-residue network."""

    def __init__(
            self,
            n_dims_afea,  # number of dimensions in per-atom features
            n_dims_rfea,  # number of dimensions in per-residue features
            dist_encoder=None,  # distance encoder
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_afea = n_dims_afea
        self.n_dims_rfea = n_dims_rfea
        self.dist_encoder = dist_encoder

        # additional configurations
        self.n_dims_encd = self.dist_encoder.n_dims

        # build the network
        self.net = nn.ModuleDict()
        self.net['c'] = nn.Sequential(
            nn.Linear(self.n_dims_encd, self.n_dims_encd),
            nn.ReLU(),
            nn.Linear(self.n_dims_encd, 1),
        )
        self.net['r'] = nn.Sequential(
            nn.Linear(self.n_dims_afea, self.n_dims_rfea),
            nn.ReLU(),
            nn.Linear(self.n_dims_rfea, self.n_dims_rfea),
        )
        self.net['ffn'] = FFN(self.n_dims_rfea)


    def forward(self, aa_seq, cmsk_mat, afea_mat, acrd_mat, rfea_mat, ridx_vec):  # pylint: disable=too-many-arguments,too-many-locals
        """Perform the forward pass.

        Args:
        * aa_seq: amino-acid sequence
        * cmsk_mat: per-atom 3D coordinates' valid-or-not masks of size L x M
        * afea_mat: per-atom features of size Na x Da
        * acrd_mat: per-atom 3D coordinates of size Na x 3
        * rfea_mat: per-residue features of size Nr x Dr
        * ridx_vec: indexing vector (from valid residues to all residues) of size Nr

        Returns:
        * rfea_mat: updated per-residue features of size Nr x Dr
        * rcrd_mat: per-residue 3D coordinates of size Nr x 3
        """

        # initialization
        n_resds = rfea_mat.shape[0]

        # convert per-atom tensors back into the sparse format (for faster computation)
        afea_tns = torch.index_select(ds2sp(aa_seq, afea_mat, cmsk_mat), 0, ridx_vec)  # Nr x M x Da
        acrd_tns = torch.index_select(ds2sp(aa_seq, acrd_mat, cmsk_mat), 0, ridx_vec)  # Nr x M x 3
        cmsk_tns = torch.index_select(cmsk_mat, 0, ridx_vec).unsqueeze(dim=2)  # Nr x M x 1
        div_fctr = torch.sum(cmsk_tns, dim=1)  # Nr x 1

        # calculate per-residue 3D coordinates
        rcrd_mat = torch.sum(cmsk_tns * acrd_tns, dim=1) / div_fctr

        # calculate per-residue features
        dist_mat = torch.norm(rcrd_mat.unsqueeze(dim=1) - acrd_tns, dim=2)
        encd_tns = self.dist_encoder.run(dist_mat.flatten()).view(n_resds, N_ATOMS_PER_RESD, -1)
        coef_tns = self.net['c'](encd_tns)
        rfea_tns_hid = self.net['r'](coef_tns * afea_tns)
        rfea_mat_hid = torch.sum(cmsk_tns * rfea_tns_hid, dim=1) / div_fctr
        rfea_mat = rfea_mat + self.net['ffn'](rfea_mat_hid)

        return rfea_mat, rcrd_mat


class Resd2Resd(nn.Module):
    """The residue-to-residue network."""

    def __init__(
            self,
            n_dims_rfea,  # number of dimensions in per-residue features
            n_dims_rpfe,  # number of dimensions in per-residue-pair features
            n_dims_emsg=32,  # number of dimensions in hidden edge messages
            dist_encoder=None,  # distance encoder
        ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_rfea = n_dims_rfea
        self.n_dims_rpfe = n_dims_rpfe
        self.n_dims_emsg = n_dims_emsg
        self.dist_encoder = dist_encoder if dist_encoder is not None else DistEncoder()

        # build the network
        self.net = nn.ModuleDict()
        self.net['egcl'] = EGCL(
            self.n_dims_rfea,
            self.n_dims_rpfe,
            n_dims_emsg=self.n_dims_emsg,
            dist_encoder=self.dist_encoder,
        )
        self.net['ffn'] = FFN(self.n_dims_rfea)


    def forward(self, graph, rfea_mat, rpfe_mat, rcrd_mat):
        """Perform the forward pass.

        Args:
        * rfea_mat: per-residue features of size Nr x Dr
        * rpfe_mat: per-residue-pair features of size Nrp x Drp
        * rcrd_mat: per-residue 3D coordinates of size Nr x 3

        Returns:
        * rfea_mat: updated per-residue features of size Nr x Dr
        * rcrd_mat: updated per-residue 3D coordinates of size Nr x 3
        """

        # initialization
        device = rfea_mat.device
        n_resds = rfea_mat.shape[0]
        rvmk_vec = torch.ones((n_resds), dtype=torch.int8, device=device)  # all residues are valid
        rumk_vec = torch.ones((n_resds), dtype=torch.int8, device=device)  # update all residues

        # perform the forward pass
        rfea_mat_hid, rcrd_mat = self.net['egcl'](
            graph, rfea_mat, rpfe_mat, rcrd_mat, rvmk_vec, rumk_vec)
        rfea_mat = rfea_mat + self.net['ffn'](rfea_mat_hid)

        return rfea_mat, rcrd_mat
