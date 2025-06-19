"""The builder class for inter-residue DA ground-truth labels."""

import torch
import numpy as np

from tfold.utils import cdist
from tfold.utils import calc_dihd_angl_batch
from tfold.utils import calc_plnr_angl_batch
from tfold.tools.prot_struct import ProtStruct


class DaLablBuilder():  # pylint: disable=too-few-public-methods
    """The builder class for inter-residue DA ground-truth labels."""

    def __init__(self):
        """Constructor function."""

        # configurations
        self.nctc_pos = 'first'  # 'first / 'last'
        self.n_bins_dist = 37
        self.dist_min = 2.0
        self.dist_max = 20.0
        self.n_bins_angl = 25
        self.angl_min = -np.pi
        self.angl_max = np.pi


    def run(self, aa_seq, cord_tns, mask_mat):  # pylint: disable=too-many-locals
        """Build ground-truth labels for inter-residue DA predictions."""

        # initialization
        n_resds = len(aa_seq)
        labl_dict = {}

        # obtain 3D coordinates for N/CA/CB atoms
        atom_names = ['N', 'CA', 'CB']
        cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)
        mask_mat_sel = ProtStruct.get_atoms(aa_seq, mask_mat, atom_names)
        x_n, x_ca, x_cb = [torch.squeeze(x, dim=1) for x in torch.split(cord_tns_sel, 1, dim=1)]
        m_n, m_ca, m_cb = [torch.squeeze(x, dim=1) for x in torch.split(mask_mat_sel, 1, dim=1)]

        # use GLY's CA atom as the replacement for its missing CB atom
        is_gly = torch.tensor([1 if x == 'G' else 0 for x in aa_seq], dtype=torch.int8)
        x_cab = is_gly[:, None] * x_ca + (1 - is_gly[:, None]) * x_cb
        m_cab = is_gly * m_ca + (1 - is_gly) * m_cb

        # calculate the CB-CB distance matrix (CA for Glycine)
        dist_mat = cdist(x_cab)
        labl_dict['cb-idx'], nctc_mat = self.__dist2idx(dist_mat)
        labl_dict['cb-msk'] = torch.outer(m_cab, m_cab).to(torch.int8)

        # calculate the CA-CB-CB'-CA' dihedral angle matrix
        cord_tns = torch.stack([
            x_ca.repeat_interleave(n_resds, dim=0),
            x_cb.repeat_interleave(n_resds, dim=0),
            x_cb.repeat(n_resds, 1),
            x_ca.repeat(n_resds, 1),
        ], dim=1)
        angl_mat = calc_dihd_angl_batch(cord_tns).view(n_resds, n_resds)
        labl_dict['om-idx'] = self.__angl2idx(angl_mat, nctc_mat)
        labl_dict['om-msk'] = torch.outer(m_ca * m_cb, m_cb * m_ca).to(torch.int8)

        # calculate the N-CA-CB-CB' dihedral angle matrix
        cord_tns = torch.stack([
            x_n.repeat_interleave(n_resds, dim=0),
            x_ca.repeat_interleave(n_resds, dim=0),
            x_cb.repeat_interleave(n_resds, dim=0),
            x_cb.repeat(n_resds, 1),
        ], dim=1)
        angl_mat = calc_dihd_angl_batch(cord_tns).view(n_resds, n_resds)
        labl_dict['th-idx'] = self.__angl2idx(angl_mat, nctc_mat)
        labl_dict['th-msk'] = torch.outer(m_n * m_ca * m_cb, m_cb).to(torch.int8)

        # calculate the CA-CB-CB' planar angle matrix
        cord_tns = torch.stack([
            x_ca.repeat_interleave(n_resds, dim=0),
            x_cb.repeat_interleave(n_resds, dim=0),
            x_cb.repeat(n_resds, 1),
        ], dim=1)
        angl_mat = calc_plnr_angl_batch(cord_tns).view(n_resds, n_resds)
        labl_dict['ph-idx'] = self.__angl2idx(angl_mat, nctc_mat)
        labl_dict['ph-msk'] = torch.outer(m_ca * m_cb, m_cb).to(torch.int8)

        return labl_dict


    def __dist2idx(self, dist_mat):
        """Convert distance values into classification indices."""

        bin_wid = (self.dist_max - self.dist_min) / (self.n_bins_dist - 1)
        idxs_mat = torch.floor((dist_mat - self.dist_min) / bin_wid).to(torch.int64)
        idxs_mat = torch.clip(idxs_mat, 0, self.n_bins_dist - 1)
        nctc_mat = torch.eq(idxs_mat, self.n_bins_dist - 1).to(torch.bool)

        if self.nctc_pos == 'first':
            idxs_mat = torch.remainder(idxs_mat + 1, self.n_bins_dist)

        return idxs_mat, nctc_mat


    def __angl2idx(self, angl_mat, nctc_mat):
        """Convert angle values into classification indices."""

        bin_wid = (self.angl_max - self.angl_min) / (self.n_bins_angl - 1)
        idxs_mat = torch.floor((angl_mat - self.angl_min) / bin_wid).to(torch.int64)
        idxs_mat = torch.clip(idxs_mat, 0, self.n_bins_angl - 2)
        idxs_mat = torch.where(nctc_mat, self.n_bins_angl - 1, idxs_mat)

        if self.nctc_pos == 'first':
            idxs_mat = torch.remainder(idxs_mat + 1, self.n_bins_angl)

        return idxs_mat
