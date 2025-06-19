"""Steric clash checker - v3.

Notes:
* Unlike <ClashCheckerV2>, <ClashCheckerV3> supports evaluating structure violations of bond length
    beside peptide bonds. In other words, intra-residue bond length is validated, which is useful
    for structure prediction methods that do not build upon the local-frame parameterization.
* List of loss items:
  > Loss-PTBA: improper bond angle in peptide bonds
  > Loss-PTBL: improper bond length in peptide bonds
  > Loss-IRBA: improper bond angle in intra-residue bonds
  > Loss-IRBL: improper bond length in intra-residue bonds
  > Loss-NB: improper inter-atom distance between non-bonded atoms
"""

import os
import logging
import itertools
from collections import defaultdict

import torch
from torch import nn
import numpy as np

from tfold.utils import cdist
from tfold.utils import calc_plnr_angl_batch
from tfold.tools.prot_struct import ProtStruct
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import RESD_MAP_3TO1
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


# Ideal bond length & angle for peptide bonds (1st: general case / 2nd: proline)
PTBL_STATS_LIST = [(1.329, 0.014), (1.341, 0.016)]  # (mean, stdev)
PTBA_STATS_DICT = {
    'CA-C-N': (-0.4473, 0.0311),
    'C-N-CA': (-0.5203, 0.0353),  # (mean, stdev)
}  # in radian

# Van der Waals radius
VDW_RADIUS = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}


class ClashCheckerV3():  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Steric clash checker - v3."""

    def __init__(
            self,
            norm_by_cpairs=True,  # whether to normalize <loss_nb> by # of clashing atom pairs
            loss_bd_max=3.0,  # maximal loss value of improper bond length & angles
            loss_nb_max=3.0,  # maximal loss value of improper distance between non-bonded atoms
            check_irbd=False,  # whether to check intra-residue bond length & angles
            debug=False,  # whether to enable the debug mode
        ):
        """Constructor function."""

        # setup configurations
        self.norm_by_cpairs = norm_by_cpairs
        self.loss_bd_max = loss_bd_max  # set to <None> for unlimited loss values
        self.loss_nb_max = loss_nb_max  # set to <None> for unlimited loss values
        self.check_irbd = check_irbd
        self.debug = debug

        # additional configurations
        self.eps = 1e-6
        self.mult_bl = 12.0  # in order to pass the stereochemical quality check of lDDT
        self.mult_ba = 12.0  # we choose 12.0, instead of 15.0, to follow AF2's setup
        self.dist_tol = 1.5
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.txt_fpath = os.path.join(curr_dir, 'data/stereo_chemical_props.txt')

        # initialize the Van der Waals radius vector for each residue type
        self.radi_vec_dict = self.__init_radi_vec()

        # get statistics of intra-residue bond length & angles
        self.irbl_list_dict, self.irba_list_dict = self.__get_ir_bond_stats()

        # initialize masks of non-bonded atom pairs for each residue type
        self.mask_mat_dict = self.__init_mask_mat()


    def run(self, aa_seq, cord_tns, cmsk_mat, asym_id=None):
        """Check for steric clashes in per-atom 3D coordinates.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * asym_id: (optional) asymmetric unit IDs of size L

        Returns:
        * loss: steric clash loss
        * metrics: dict of evaluation metrics
        """

        # initialization
        device = cord_tns.device
        n_resds = cord_tns.shape[0]
        if next(iter(self.radi_vec_dict.values())).device != device:
            self.radi_vec_dict = {k: v.to(device) for k, v in self.radi_vec_dict.items()}
        if next(iter(self.mask_mat_dict.values())).device != device:
            self.mask_mat_dict = {k: v.to(device) for k, v in self.mask_mat_dict.items()}

        # detect chain delimiters from asymmetric unit IDs
        if asym_id is not None:
            pmsk_vec = torch.eq(asym_id[:-1], asym_id[1:]).to(torch.int8)  # 1: peptide bond exists
        else:
            pmsk_vec = torch.ones(n_resds - 1, dtype=torch.int8, device=device)

        # check for improper bond length & angle in peptide bonds
        loss_ptbl, loss_ptba = self.__check_pt_bonds(aa_seq, cord_tns, cmsk_mat, pmsk_vec)

        # check for imporper bond length & angle in intra-residue bonded atoms
        loss_irbl, loss_irba = self.__check_ir_bonds(aa_seq, cord_tns, cmsk_mat)

        # check for improper inter-atom distance between non-bonded atoms
        loss_nb = self.__check_nb_dist(aa_seq, cord_tns, cmsk_mat, pmsk_vec)

        # apply gradient-preserving loss scaling
        if self.loss_bd_max is not None:
            if loss_ptbl > self.loss_bd_max:
                loss_ptbl = self.loss_bd_max * loss_ptbl / loss_ptbl.detach()
            if loss_ptba > self.loss_bd_max:
                loss_ptba = self.loss_bd_max * loss_ptba / loss_ptba.detach()
            if loss_irbl > self.loss_bd_max:
                loss_irbl = self.loss_bd_max * loss_irbl / loss_irbl.detach()
            if loss_irba > self.loss_bd_max:
                loss_irba = self.loss_bd_max * loss_irba / loss_irba.detach()
        if self.loss_nb_max is not None:
            if loss_nb > self.loss_nb_max:
                loss_nb = self.loss_nb_max * loss_nb / loss_nb.detach()

        # calculate the overall loss function
        loss = loss_ptbl + loss_ptba + loss_irbl + loss_irba + loss_nb
        metrics = {
            'Loss-PTBL': loss_ptbl.item(),
            'Loss-PTBA': loss_ptba.item(),
            'Loss-IRBL': loss_irbl.item(),
            'Loss-IRBA': loss_irba.item(),
            'Loss-NB': loss_nb.item(),
        }

        return loss, metrics


    def __check_pt_bonds(self, aa_seq, cord_tns, cmsk_mat, pmsk_vec):
        """Check for improper bond length & angles in peptide bonds."""

        def _idx2name(idx_resd):
            resd_name_ext = f'{aa_seq[idx_resd]}{idx_resd + 1}'
            return resd_name_ext

        # initialization
        device = cord_tns.device

        # get 3D coordinates for N / CA / C atoms
        atom_names = ['N', 'CA', 'C']
        cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)
        cmsk_mat_sel = ProtStruct.get_atoms(aa_seq, cmsk_mat, atom_names)
        cord_mat_n, cord_mat_ca, cord_mat_c = \
            [x.squeeze(dim=1) for x in torch.split(cord_tns_sel, 1, dim=1)]
        cmsk_vec_n, cmsk_vec_ca, cmsk_vec_c = \
            [x.squeeze(dim=1) for x in torch.split(cmsk_mat_sel, 1, dim=1)]

        # calculate the steric clash loss for bond length (C-N)
        dist_vec = torch.norm(cord_mat_c[:-1] - cord_mat_n[1:], dim=1)
        dmsk_vec = pmsk_vec * cmsk_vec_c[:-1] * cmsk_vec_n[1:]
        idx_list = [0 if x != 'P' else 1 for x in aa_seq[1:]]  # general: 0 / proline: 1
        mean_vec = torch.tensor(
            [PTBL_STATS_LIST[idx][0] for idx in idx_list], dtype=torch.float32, device=device)
        stdv_vec = torch.tensor(
            [PTBL_STATS_LIST[idx][1] for idx in idx_list], dtype=torch.float32, device=device)
        # derr_vec = torch.clip(torch.abs(dist_vec - mean_vec) - self.mult_bl * stdv_vec, min=0.0)
        derr_vec = torch.clip(torch.abs((dist_vec - mean_vec) / stdv_vec) - self.mult_bl, min=0.0)
        loss_bl = torch.sum(dmsk_vec * derr_vec) / (torch.sum(dmsk_vec) + self.eps)

        # [DEBUG-only] report improper bond length
        if self.debug:
            idxs_nnz = torch.nonzero(dmsk_vec * derr_vec).view(-1).tolist()
            if len(idxs_nnz) != 0:
                logging.debug('list of residue pairs w/ improper bond length:')
                for idx in idxs_nnz:
                    resd_name_ext_pri = _idx2name(idx)
                    resd_name_ext_sec = _idx2name(idx + 1)
                    dist_val = dist_vec[idx].item()
                    logging.debug('> %s - %s: %.4f', resd_name_ext_pri, resd_name_ext_sec, dist_val)

        # calculate the steric clash loss for bond angles (CA-C-N)
        cord_tns_tmp = torch.stack([cord_mat_ca[:-1], cord_mat_c[:-1], cord_mat_n[1:]], dim=1)
        angl_vec = torch.cos(calc_plnr_angl_batch(cord_tns_tmp))  # compare in the cosine space
        amsk_vec = pmsk_vec * cmsk_vec_ca[:-1] * cmsk_vec_c[:-1] * cmsk_vec_n[1:]
        mean_vec = PTBA_STATS_DICT['CA-C-N'][0] * torch.ones_like(angl_vec)
        stdv_vec = PTBA_STATS_DICT['CA-C-N'][1] * torch.ones_like(angl_vec)
        # aerr_vec = torch.clip(torch.abs(angl_vec - mean_vec) - self.mult_ba * stdv_vec, min=0.0)
        aerr_vec = torch.clip(torch.abs((angl_vec - mean_vec) / stdv_vec) - self.mult_ba, min=0.0)
        loss_ba_pri = torch.sum(amsk_vec * aerr_vec) / (torch.sum(amsk_vec) + self.eps)

        # [DEBUG-only] report improper bond angle (CA-C-N)
        if self.debug:
            idxs_nnz = torch.nonzero(amsk_vec * aerr_vec).view(-1).tolist()
            if len(idxs_nnz) != 0:
                logging.debug('list of residue pairs w/ improper bond angle (CA-C-N):')
                for idx in idxs_nnz:
                    resd_name_ext_pri = _idx2name(idx)
                    resd_name_ext_sec = _idx2name(idx + 1)
                    angl_val = angl_vec[idx].item()
                    logging.debug('> %s - %s: %.4f', resd_name_ext_pri, resd_name_ext_sec, angl_val)

        # calculate the steric clash loss for bond angles (C-N-CA)
        cord_tns_tmp = torch.stack([cord_mat_c[:-1], cord_mat_n[1:], cord_mat_ca[1:]], dim=1)
        angl_vec = torch.cos(calc_plnr_angl_batch(cord_tns_tmp))  # compare in the cosine space
        amsk_vec = pmsk_vec * cmsk_vec_c[:-1] * cmsk_vec_n[1:] * cmsk_vec_ca[1:]
        mean_vec = PTBA_STATS_DICT['C-N-CA'][0] * torch.ones_like(angl_vec)
        stdv_vec = PTBA_STATS_DICT['C-N-CA'][1] * torch.ones_like(angl_vec)
        # aerr_vec = torch.clip(torch.abs(angl_vec - mean_vec) - self.mult_ba * stdv_vec, min=0.0)
        aerr_vec = torch.clip(torch.abs((angl_vec - mean_vec) / stdv_vec) - self.mult_ba, min=0.0)
        loss_ba_sec = torch.sum(amsk_vec * aerr_vec) / (torch.sum(amsk_vec) + self.eps)

        # [DEBUG-only] report improper bond angle (C-N-CA)
        if self.debug:
            idxs_nnz = torch.nonzero(amsk_vec * aerr_vec).view(-1).tolist()
            if len(idxs_nnz) != 0:
                logging.debug('list of residue pairs w/ improper bond angle (C-N-CA):')
                for idx in idxs_nnz:
                    resd_name_ext_pri = _idx2name(idx)
                    resd_name_ext_sec = _idx2name(idx + 1)
                    angl_val = angl_vec[idx].item()
                    logging.debug('> %s - %s: %.4f', resd_name_ext_pri, resd_name_ext_sec, angl_val)

        # calculate the overall steric clash loss for bond angles
        loss_ba = (loss_ba_pri + loss_ba_sec) / 2.0

        return loss_bl, loss_ba


    def __check_ir_bonds(self, aa_seq, cord_tns, cmsk_mat):
        """Check for improper bond length & angles in intra-residue bonds."""

        n_resds = len(aa_seq)
        loss_bl_list = []
        loss_ba_list = []
        for resd_name in RESD_NAMES_1C:
            idxs_resd = [idx for idx, name in enumerate(aa_seq) if name == resd_name]
            if len(idxs_resd) == 0:
                continue
            ridx_vec = torch.tensor(idxs_resd, dtype=torch.int32, device=cord_tns.device)
            cord_tns_sel = torch.index_select(cord_tns, 0, ridx_vec)
            cmsk_mat_sel = torch.index_select(cmsk_mat, 0, ridx_vec)
            loss_bl, loss_ba = self.__check_ir_bonds_impl(resd_name, cord_tns_sel, cmsk_mat_sel)
            loss_bl_list.append(loss_bl)
            loss_ba_list.append(loss_ba)
        loss_bl = torch.sum(torch.stack(loss_bl_list)) / n_resds
        loss_ba = torch.sum(torch.stack(loss_ba_list)) / n_resds

        return loss_bl, loss_ba


    def __check_ir_bonds_impl(self, resd_name, cord_tns, cmsk_mat):
        """Check for improper bond length & angles in intra-residue bonds - core implementation."""

        # initialization
        dtype = cord_tns.dtype
        device = cord_tns.device
        n_resds = cord_tns.shape[0]
        irbl_list = self.irbl_list_dict[resd_name]
        irba_list = self.irba_list_dict[resd_name]
        atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]

        # check for improper bond length
        n_bonds = len(irbl_list)
        cord_tns_bl = torch.zeros((n_resds, n_bonds, 2, 3), dtype=dtype, device=device)
        bmsk_mat = torch.zeros((n_resds, n_bonds), dtype=torch.int8, device=device)
        for idx_bond, ((idx_atom_0, idx_atom_1), _, _) in enumerate(irbl_list):
            cord_tns_bl[:, idx_bond, 0] = cord_tns[:, idx_atom_0]
            cord_tns_bl[:, idx_bond, 1] = cord_tns[:, idx_atom_1]
            bmsk_mat[:, idx_bond] = cmsk_mat[:, idx_atom_0] * cmsk_mat[:, idx_atom_1]
        dist_mat = torch.norm(cord_tns_bl[:, :, 0] - cord_tns_bl[:, :, 1], dim=2)
        mean_mat = torch.tensor([[x[1] for x in irbl_list]], dtype=dtype, device=device)
        stdv_mat = torch.tensor([[x[2] for x in irbl_list]], dtype=dtype, device=device)
        # derr_mat = torch.clip(torch.abs(dist_mat - mean_mat) - self.mult_bl * stdv_mat, min=0.0)
        derr_mat = torch.clip(torch.abs((dist_mat - mean_mat) / stdv_mat) - self.mult_bl, min=0.0)
        loss_bl = torch.sum(
            torch.sum(bmsk_mat * derr_mat, dim=1) / (torch.sum(bmsk_mat, dim=1) + self.eps))

        # check for improper bond angles
        n_bonds = len(irba_list)
        cord_tns_ba = torch.zeros((n_resds, n_bonds, 3, 3), dtype=dtype, device=device)
        bmsk_mat = torch.zeros((n_resds, n_bonds), dtype=torch.int8, device=device)
        for idx_bond, ((idx_atom_0, idx_atom_1, idx_atom_2), _, _) in enumerate(irba_list):
            cord_tns_ba[:, idx_bond, 0] = cord_tns[:, idx_atom_0]
            cord_tns_ba[:, idx_bond, 1] = cord_tns[:, idx_atom_1]
            cord_tns_ba[:, idx_bond, 2] = cord_tns[:, idx_atom_2]
            bmsk_mat[:, idx_bond] = \
                cmsk_mat[:, idx_atom_0] * cmsk_mat[:, idx_atom_1] * cmsk_mat[:, idx_atom_2]
        angl_mat = calc_plnr_angl_batch(cord_tns_ba.view(-1, 3, 3)).view(n_resds, n_bonds)
        mean_mat = torch.tensor([[x[1] for x in irba_list]], dtype=dtype, device=device)
        stdv_mat = torch.tensor([[x[2] for x in irba_list]], dtype=dtype, device=device)
        # aerr_mat = torch.clip(torch.abs(angl_mat - mean_mat) - self.mult_ba * stdv_mat, min=0.0)
        aerr_mat = torch.clip(torch.abs((angl_mat - mean_mat) / stdv_mat) - self.mult_ba, min=0.0)
        loss_ba = torch.sum(
            torch.sum(bmsk_mat * aerr_mat, dim=1) / (torch.sum(bmsk_mat, dim=1) + self.eps))

        return loss_bl, loss_ba


    def __check_nb_dist(self, aa_seq, cord_tns, cmsk_mat, pmsk_vec):
        """Check for improper inter-atom distance between non-bonded atoms."""

        def _idx2name(idx_atom_ext):
            idx_resd = idx_atom_ext // n_atoms
            idx_atom = idx_atom_ext % n_atoms
            resd_name = aa_seq[idx_resd]
            atom_name = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]][idx_atom]
            atom_name_ext = f'{resd_name}{idx_resd + 1}({atom_name})'
            return atom_name_ext

        # initialization
        n_resds, n_atoms = cmsk_mat.shape

        # calculate inter-atom distance
        dist_mat = cdist(cord_tns.view(-1, 3))
        dmsk_mat = cmsk_mat.view(-1, 1) * cmsk_mat.view(1, -1)

        # find all the CYS residues (to skip steric clash checks on disulfide bonds)
        idxs_resd_cys = [idx for idx, name in enumerate(aa_seq) if name == 'C']
        idx_atom_sg = ATOM_NAMES_PER_RESD['CYS'].index('SG')

        # determine the lower bound of inter-atom distance between non-bonded atoms
        radi_vec = torch.cat([self.radi_vec_dict[x] for x in aa_seq], dim=0)
        lbnd_mat = radi_vec.view(-1, 1) + radi_vec.view(1, -1) - self.dist_tol
        lmsk_tns = dmsk_mat.view(n_resds, n_atoms, n_resds, n_atoms).permute(0, 2, 1, 3)
        for idx_resd, resd_name in enumerate(aa_seq):
            lmsk_tns[idx_resd, :idx_resd] = 0  # mask-out lower triangular entries
            lmsk_tns[idx_resd, idx_resd] *= self.mask_mat_dict[resd_name]
        for idx_resd_pri in idxs_resd_cys:
            for idx_resd_sec in idxs_resd_cys:
                lmsk_tns[idx_resd_pri, idx_resd_sec, idx_atom_sg, idx_atom_sg] = 0
        for idx_resd_prev in range(n_resds - 1):
            if pmsk_vec[idx_resd_prev] == 0:
                continue  # skip chain delimiters
            idx_resd_next = idx_resd_prev + 1
            atom_names_prev = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa_seq[idx_resd_prev]]]
            atom_names_next = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa_seq[idx_resd_next]]]
            idx_atom_prev = atom_names_prev.index('C')
            idx_atom_next = atom_names_next.index('N')
            lmsk_tns[idx_resd_prev, idx_resd_next, idx_atom_prev, idx_atom_next] = 0
        lmsk_mat = lmsk_tns.permute(0, 2, 1, 3).view(n_resds * n_atoms, n_resds * n_atoms)

        # calculate the steric clash loss for non-bonded atoms
        derr_mat = torch.clip(lbnd_mat - dist_mat, min=0.0)
        if not self.norm_by_cpairs:
            loss = torch.sum(lmsk_mat * derr_mat)
        else:
            n_pairs_clsh = torch.sum(lmsk_mat * (derr_mat > self.eps))
            loss = torch.sum(lmsk_mat * derr_mat) / (n_pairs_clsh + self.eps)

        # [DEBUG-only] report improper inter-atom distance between non-bonded atoms
        if self.debug:
            idxs_nnz = torch.nonzero(lmsk_mat * derr_mat)
            if idxs_nnz.shape[0] != 0:
                logging.debug('list of atom pairs w/ improper inter-atom distance (non-bonded)')
                for idx_pair in range(idxs_nnz.shape[0]):
                    idx_atom_ext_pri = idxs_nnz[idx_pair][0].item()
                    idx_atom_ext_sec = idxs_nnz[idx_pair][1].item()
                    atom_name_ext_pri = _idx2name(idx_atom_ext_pri)
                    atom_name_ext_sec = _idx2name(idx_atom_ext_sec)
                    dist_val = dist_mat[idx_atom_ext_pri, idx_atom_ext_sec].item()
                    lbnd_val = lbnd_mat[idx_atom_ext_pri, idx_atom_ext_sec].item()
                    logging.debug('> %s - %s: %.4f (%.4f)',
                                  atom_name_ext_pri, atom_name_ext_sec, dist_val, lbnd_val)

        return loss


    def __init_radi_vec(self):
        """Initialize the Van der Waals radius vector for each residue type."""

        # initialize the Van der Waals radius vector for each residue type
        radi_vec_dict = {}
        for resd_name in RESD_NAMES_1C:
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            radi_vec = torch.tensor([VDW_RADIUS[x[0]] for x in atom_names], dtype=torch.float32)
            pad_size = N_ATOMS_PER_RESD - len(atom_names)
            radi_vec_dict[resd_name] = nn.functional.pad(radi_vec, [0, pad_size])

        # [DEBUG-only] display the Van der Waals radius vector for each residue type
        if self.debug:
            for resd_name, radi_vec in radi_vec_dict.items():
                radi_vec_str = ' '.join([f'{x:.2f}' for x in radi_vec.tolist()])
                logging.debug('%s: %s', resd_name, radi_vec_str)

        return radi_vec_dict


    def __get_ir_bond_stats(self):
        """Get statistics of intra-residue bond length & angles, indexed by residue names."""

        # get statistics of intra-residue bond length & angles
        parse_irbl = False
        parse_irba = False
        irbl_list_dict = defaultdict(list)
        irba_list_dict = defaultdict(list)
        with open(self.txt_fpath, 'r', encoding='UTF-8') as i_file:
            for i_line in i_file:
                # determine the parsing mode
                sub_strs = i_line.split()
                if i_line.startswith('Bond'):
                    parse_irbl = True
                    continue
                if i_line.startswith('Angle'):
                    parse_irba = True
                    continue
                if i_line.startswith('-'):
                    if parse_irbl:
                        parse_irbl = False
                    else:
                        break

                # parse statistics for intra-residue bond length OR angle
                sub_strs = i_line.split()
                if parse_irbl:
                    resd_name = RESD_MAP_3TO1[sub_strs[1]]
                    atom_names = ATOM_NAMES_PER_RESD[sub_strs[1]]
                    idxs_atom = [atom_names.index(x) for x in sub_strs[0].split('-')]
                    dist_avg = float(sub_strs[2])
                    dist_std = float(sub_strs[3])
                    irbl_list_dict[resd_name].append((idxs_atom, dist_avg, dist_std))
                if parse_irba:
                    resd_name = RESD_MAP_3TO1[sub_strs[1]]
                    atom_names = ATOM_NAMES_PER_RESD[sub_strs[1]]
                    idxs_atom = [atom_names.index(x) for x in sub_strs[0].split('-')]
                    angl_avg = np.pi * float(sub_strs[2]) / 180.0
                    angl_std = np.pi * float(sub_strs[3]) / 180.0
                    irba_list_dict[resd_name].append((idxs_atom, angl_avg, angl_std))

        return irbl_list_dict, irba_list_dict


    def __init_mask_mat(self):  # pylint: disable=too-many-locals
        """Initialize masks of non-bonded atom pairs for each residue type."""

        # initialize masks of non-bonded atom pairs for each residue type
        mask_mat_dict = {}
        for resd_name, irbl_list in self.irbl_list_dict.items():
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            n_atoms = len(atom_names)
            mask_mat = 1 - torch.eye(n_atoms, dtype=torch.int8)  # 1: non-bonded
            for (idx_atom_0, idx_atom_1), _, _ in irbl_list:
                mask_mat[idx_atom_0, idx_atom_1] = 0
                mask_mat[idx_atom_1, idx_atom_0] = 0
            for idx_atom in range(n_atoms):
                mask_mat[idx_atom, :idx_atom] = 0  # mask-out lower triangular entries
            pad_size = N_ATOMS_PER_RESD - n_atoms
            mask_mat_dict[resd_name] = nn.functional.pad(mask_mat, [0, pad_size, 0, pad_size])

        # [DEBUG-only] display masks of non-bonded atom pairs for each residue type
        if self.debug:
            for resd_name, mask_mat in mask_mat_dict.items():
                atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
                for idx in range(N_ATOMS_PER_RESD):
                    atom_name = f'{atom_names[idx]:3s}' if idx < len(atom_names) else 'PAD'
                    mask_mat_str = ''.join(['*' if x == 1 else '.' for x in mask_mat[idx].tolist()])
                    logging.debug('%s - %s: %s', resd_name, atom_name, mask_mat_str)

        return mask_mat_dict
