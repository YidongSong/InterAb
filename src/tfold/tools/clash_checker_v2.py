"""Steric clash checker - v2."""

import os
import logging
from collections import defaultdict

import torch
from torch import nn

from tfold.utils import cdist
from tfold.tools.prot_struct import ProtStruct
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import RESD_MAP_3TO1
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


# Ideal bond length & angle for peptide bonds (1st: general case / 2nd: proline)
PPTD_BND_LEN_LIST = [(1.329, 0.014), (1.341, 0.016)]  # (mean, stdev)
PPTD_BND_AGL_DICT = {
    'CA-C-N': (-0.4473, 0.0311),
    'C-N-CA': (-0.5203, 0.0353),  # (mean, stdev)
}  # in radian

# Van der Waals radius
VDW_RADIUS = {'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80}


class ClashCheckerV2():  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """Steric clash checker - v2."""

    def __init__(self, norm_by_cpairs=True, loss_nb_max=None, debug=False):
        """Constructor function."""

        # setup configurations
        self.norm_by_cpairs = norm_by_cpairs
        self.loss_nb_max = loss_nb_max
        self.debug = debug

        # additional configurations
        self.eps = 1e-6
        self.mult_bl = 12.0  # in order to pass the stereochemical quality check of lDDT
        self.mult_ba = 12.0
        self.dist_tol = 1.5

        # initialize the Van der Waals radius vector for each residue type
        self.radi_vec_dict = self.__init_radi_vec()

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
        if next(iter(self.radi_vec_dict.values())).device != device:
            self.radi_vec_dict = {k: v.to(device) for k, v in self.radi_vec_dict.items()}
        if next(iter(self.mask_mat_dict.values())).device != device:
            self.mask_mat_dict = {k: v.to(device) for k, v in self.mask_mat_dict.items()}

        # detect chain delimiters from asymmetric unit IDs
        idxs_resd_dlim = []
        if asym_id is not None:
            n_resds = asym_id.shape[0]
            for idx_resd in range(n_resds - 1):
                if asym_id[idx_resd] != asym_id[idx_resd + 1]:
                    idxs_resd_dlim.append(idx_resd)

        # check for improper bond length & angle in peptide bonds
        loss_bl, loss_ba = self.__check_pptd_bnds(aa_seq, cord_tns, cmsk_mat, idxs_resd_dlim)

        # check for improper inter-atom distance between non-bonded atoms
        loss_nb = self.__check_nbnd_dist(aa_seq, cord_tns, cmsk_mat, idxs_resd_dlim)
        if (self.loss_nb_max is not None) and (loss_nb > self.loss_nb_max):
            loss_nb = self.loss_nb_max * loss_nb / loss_nb.detach()

        # calculate the overall loss function
        loss = loss_bl + loss_ba + loss_nb
        metrics = {'Loss-BL': loss_bl.item(), 'Loss-BA': loss_ba.item(), 'Loss-NB': loss_nb.item()}

        return loss, metrics


    def __check_pptd_bnds(self, aa_seq, cord_tns, cmsk_mat, idxs_resd_dlim):  # pylint: disable=too-many-locals,too-many-statements
        """Check for improper bond length & angle in peptide bonds."""

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

        # calculate the steric clash loss for bond length
        dist_vec = torch.norm(cord_mat_c[:-1] - cord_mat_n[1:], dim=1)
        dmsk_vec = cmsk_vec_c[:-1] * cmsk_vec_n[1:]
        for idx_resd in idxs_resd_dlim:
            dmsk_vec[idx_resd] = 0  # skip chain delimiters
        idx_list = [0 if x != 'P' else 1 for x in aa_seq[1:]]  # general: 0 / proline: 1
        mean_vec = torch.tensor(
            [PPTD_BND_LEN_LIST[idx][0] for idx in idx_list], dtype=torch.float32, device=device)
        stdv_vec = torch.tensor(
            [PPTD_BND_LEN_LIST[idx][1] for idx in idx_list], dtype=torch.float32, device=device)
        derr_vec = torch.clip(torch.abs(dist_vec - mean_vec) - self.mult_bl * stdv_vec, min=0.0)
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

        # calculate unit vectors between adjacent atoms
        dcrd_mat_ca_c = cord_mat_c[:-1] - cord_mat_ca[:-1]
        dcrd_mat_c_n = cord_mat_n[1:] - cord_mat_c[:-1]
        dcrd_mat_n_ca = cord_mat_ca[1:] - cord_mat_n[1:]
        unit_vecs_ca_c = dcrd_mat_ca_c / torch.norm(dcrd_mat_ca_c, dim=1, keepdim=True)
        unit_vecs_c_n = dcrd_mat_c_n / torch.norm(dcrd_mat_c_n, dim=1, keepdim=True)
        unit_vecs_n_ca = dcrd_mat_n_ca / torch.norm(dcrd_mat_n_ca, dim=1, keepdim=True)

        # calculate the steric clash loss for bond angle (CA-C-N)
        angl_vec = torch.sum((-unit_vecs_ca_c) * unit_vecs_c_n, dim=1)
        amsk_vec = cmsk_vec_ca[:-1] * cmsk_vec_c[:-1] * cmsk_vec_n[1:]
        for idx_resd in idxs_resd_dlim:
            amsk_vec[idx_resd] = 0  # skip chain delimiters
        mean_vec = PPTD_BND_AGL_DICT['CA-C-N'][0] * torch.ones_like(angl_vec)
        stdv_vec = PPTD_BND_AGL_DICT['CA-C-N'][1] * torch.ones_like(angl_vec)
        aerr_vec = torch.clip(torch.abs(angl_vec - mean_vec) - self.mult_ba * stdv_vec, min=0.0)
        loss_ba_ca_c_n = torch.sum(amsk_vec * aerr_vec) / (torch.sum(amsk_vec) + self.eps)

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

        # calculate the steric clash loss for bond angle (C-N-CA)
        angl_vec = torch.sum((-unit_vecs_c_n) * unit_vecs_n_ca, dim=1)
        amsk_vec = cmsk_vec_c[:-1] * cmsk_vec_n[1:] * cmsk_vec_ca[1:]
        for idx_resd in idxs_resd_dlim:
            amsk_vec[idx_resd] = 0  # skip chain delimiters
        mean_vec = PPTD_BND_AGL_DICT['C-N-CA'][0] * torch.ones_like(angl_vec)
        stdv_vec = PPTD_BND_AGL_DICT['C-N-CA'][1] * torch.ones_like(angl_vec)
        aerr_vec = torch.clip(torch.abs(angl_vec - mean_vec) - self.mult_ba * stdv_vec, min=0.0)
        loss_ba_c_n_ca = torch.sum(amsk_vec * aerr_vec) / (torch.sum(amsk_vec) + self.eps)

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
        loss_ba = (loss_ba_ca_c_n + loss_ba_c_n_ca) / 2.0

        return loss_bl, loss_ba


    def __check_nbnd_dist(self, aa_seq, cord_tns, cmsk_mat, idxs_resd_dlim):  # pylint: disable=too-many-locals
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
            if idx_resd_prev in idxs_resd_dlim:
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


    def __init_mask_mat(self):  # pylint: disable=too-many-locals
        """Initialize masks of non-bonded atom pairs for each residue type."""

        # obtain the bond list from ./data/stereo_chemical_props.txt
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        txt_fpath = os.path.join(curr_dir, 'data', 'stereo_chemical_props.txt')
        self.bond_list_dict = defaultdict(list)  # indexed by residue types
        with open(txt_fpath, 'r', encoding='UTF-8') as i_file:
            for i_line in i_file:
                sub_strs = i_line.split()
                if sub_strs[0] == 'Bond':
                    continue
                if sub_strs[0] == '-':
                    break
                atom_name_pri, atom_name_sec = sub_strs[0].split('-')
                resd_name = RESD_MAP_3TO1[sub_strs[1]]
                self.bond_list_dict[resd_name].append((atom_name_pri, atom_name_sec))

        # initialize masks of non-bonded atom pairs for each residue type
        mask_mat_dict = {}
        for resd_name in RESD_NAMES_1C:
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            n_atoms = len(atom_names)
            mask_mat = 1 - torch.eye(n_atoms, dtype=torch.int8)
            for atom_name_pri, atom_name_sec in self.bond_list_dict[resd_name]:
                idx_atom_pri = atom_names.index(atom_name_pri)
                idx_atom_sec = atom_names.index(atom_name_sec)
                mask_mat[idx_atom_pri, idx_atom_sec] = 0
                mask_mat[idx_atom_sec, idx_atom_pri] = 0
            for idx_atom in range(1, n_atoms):
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
