"""Steric clash checker for distance between non-bonded atoms."""

import os
import logging

import torch

from tfold.utils import cdist
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


class ClashChecker():
    """Steric clash checker for distance between non-bonded atoms."""

    def __init__(self, device=None):
        """Constructor function."""

        # initialization
        self.device = device if device is not None else torch.device('cpu')

        # additional configurations
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        prm_fpath = os.path.join(curr_dir, 'data/stereo_chemical_props.txt')
        self.params = self.__load_params(prm_fpath)

        # find all the possible element & element-pair types
        elem_type_unk = 'X'
        self.elem_types = sorted(list({elem_type_unk} | {x[0] for x in self.params}))
        self.n_elems = len(self.elem_types)
        self.idx_elem_unk = self.elem_types.index(elem_type_unk)


    def run(self, aa_seq, cord_tns, cmsk_mat, rtn_atpr_list=False):  # pylint: disable=too-many-locals
        """Check for steric clashes in the PDB file.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * rtn_atpr_list: (optional) whether to return a list of problematic atom pairs

        Returns:
        * loss: steric clash loss
        * atpr_list: (optional) list of problematic atom pairs
        """

        # send 3D coordinates and validness masks to the specified device
        device = cord_tns.device
        n_resds, n_atoms, _ = cord_tns.shape
        cord_tns = cord_tns.to(self.device)
        cmsk_mat = cmsk_mat.to(self.device)

        # build the distance threshold for all the non-bonded atom pairs
        dthr_tns, dmsk_tns = self.__build_dthr_n_dmsk(aa_seq, cmsk_mat)

        # calculate the steric clash loss term for each atom
        dist_tns = cdist(cord_tns.view(-1, 3)).view(n_resds, n_atoms, n_resds, n_atoms)
        loss_tns = dmsk_tns * torch.clamp(dthr_tns - dist_tns, min=0.0)
        loss_mat = torch.max(loss_tns.view(n_resds, n_atoms, -1), dim=-1)[0]
        loss = torch.mean(loss_mat)

        # check for residue pairs w/ non-zero steric clash losses
        if rtn_atpr_list:
            atpr_list = []
            for idx_resd, resd_name in enumerate(aa_seq):
                atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
                for idx_atom, atom_name in enumerate(atom_names):
                    if loss_mat[idx_resd, idx_atom] > 0.0:
                        idx_clsh = torch.argmax(loss_tns[idx_resd, idx_atom].view(-1)).item()
                        idx_resd_clsh = idx_clsh // n_atoms
                        if idx_resd > idx_resd_clsh:
                            continue  # should have been added before
                        idx_atom_clsh = idx_clsh % n_atoms
                        resd_name_clsh = aa_seq[idx_resd_clsh]
                        atom_name_clsh = \
                            ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name_clsh]][idx_atom_clsh]
                        dist_val = dist_tns[idx_resd, idx_atom, idx_resd_clsh, idx_atom_clsh]
                        dthr_val = dthr_tns[idx_resd, idx_atom, idx_resd_clsh, idx_atom_clsh]
                        atpr_list.append({
                            'ir1': idx_resd,
                            'rn1': RESD_MAP_1TO3[resd_name],
                            'an1': atom_name,
                            'ir2': idx_resd_clsh,
                            'rn2': RESD_MAP_1TO3[resd_name_clsh],
                            'an2': atom_name_clsh,
                            'dist': dist_val,
                            'dthr': dthr_val,
                        })

        # send <loss> back to the correct device
        loss = loss.to(device)

        return loss if not rtn_atpr_list else (loss, atpr_list)


    @classmethod
    def disp_atpr_list(cls, atpr_list):
        """Display the list of problematic atom pairs.

        Args:
        * atpr_list: (optional) list of problematic atom pairs

        Returns: n/a
        """

        # early exit on the empty list
        if len(atpr_list) == 0:
            logging.info('no problematic atom pairs detected; aborting ...')
            return

        # show the list of problematic atom pairs
        logging.info('=== problematic atom pairs (below) ===')
        for atpr in atpr_list:
            idx_resd_1, resd_name_1, atom_name_1 = atpr['ir1'], atpr['rn1'], atpr['an1']
            idx_resd_2, resd_name_2, atom_name_2 = atpr['ir2'], atpr['rn2'], atpr['an2']
            atom_1 = f'{resd_name_1}({idx_resd_1})-{atom_name_1}'
            atom_2 = f'{resd_name_2}({idx_resd_2})-{atom_name_2}'
            logging.info('%s <=> %s: %.4f (%.4f)', atom_1, atom_2, atpr['dist'], atpr['dthr'])
        logging.info('=== problematic atom pairs (above) ===')


    @classmethod
    def __load_params(cls, path):
        """Load steric clash check parameters for distance between non-bonded atoms."""

        params = {}
        enbl_read = False
        with open(path, 'r', encoding='UTF-8') as i_file:
            for i_line in i_file:
                if i_line.startswith('Non-bonded distance'):
                    enbl_read = True
                elif i_line.startswith('-'):
                    enbl_read = False
                elif enbl_read:
                    atom_pair, dist_min, dist_tlr = i_line.split()
                    params[atom_pair] = float(dist_min) - float(dist_tlr)

        return params


    @torch.no_grad()
    def __build_dthr_n_dmsk(self, aa_seq, cmsk_mat):  # pylint: disable=too-many-locals
        """Build the distance threshold tensor and its validness masks.

        Note:
        * Based on the design of AlphaFold2's structure module, intra-residue atoms should not have
            any violations of distance between non-bonded atoms.
        * Thus, we only need to check for inter-residue atoms, with N-C atoms from adjacent residues
            omitted, which form peptide bonds and therefore are not non-bonded.

        FIXME:
        * Actually, intra-residue steric clashes are possible between different rigid groups!
        """

        # initialization
        n_resds, n_atoms = cmsk_mat.shape

        # convert the (pair-of-elements, dist-thres) dict into a vector
        wei_mat = torch.zeros((self.n_elems, self.n_elems), dtype=torch.float32, device=self.device)
        for key, val in self.params.items():
            elem_type_pri, elem_type_sec = key.split('-')
            idx_elem_pri = self.elem_types.index(elem_type_pri)
            idx_elem_sec = self.elem_types.index(elem_type_sec)
            wei_mat[idx_elem_pri, idx_elem_sec] = val
            wei_mat[idx_elem_sec, idx_elem_pri] = val
        wei_vec = wei_mat.view(-1)

        # generate one-hot encodings of atoms based on element types
        idxs_elem_mat = self.idx_elem_unk * \
            torch.ones((n_resds, n_atoms), dtype=torch.int64, device=self.device)
        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            idxs_elem_mat[idx_resd, :len(atom_names)] = \
                torch.tensor([self.elem_types.index(x[0]) for x in atom_names], device=self.device)

        # generate one-hot encodings of atom pairs based on element type pairs
        idxs_elmp_mat = self.n_elems * idxs_elem_mat.view(-1, 1) + idxs_elem_mat.view(1, -1)

        # build the distance threshold tensor
        dthr_tns = torch.gather(
            wei_vec, 0, idxs_elmp_mat.view(-1)).view(n_resds, n_atoms, n_resds, n_atoms)

        # build the distance threshold tensor' validness masks
        rmsk_mat = 1 - torch.eye(n_resds, dtype=torch.int8, device=self.device)
        dmsk_tns = rmsk_mat.view(n_resds, 1, n_resds, 1) * \
            cmsk_mat.view(n_resds, n_atoms, 1, 1) * cmsk_mat.view(1, 1, n_resds, n_atoms)
        for idx_resd_curr in range(1, n_resds):
            idx_resd_prev = idx_resd_curr - 1
            atom_names_prev = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa_seq[idx_resd_prev]]]
            atom_names_curr = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[aa_seq[idx_resd_curr]]]
            idx_atom_prev = atom_names_prev.index('C')
            idx_atom_curr = atom_names_curr.index('N')
            dmsk_tns[idx_resd_prev, idx_atom_prev, idx_resd_curr, idx_atom_curr] = 0
            dmsk_tns[idx_resd_curr, idx_atom_curr, idx_resd_prev, idx_atom_prev] = 0

        return dthr_tns, dmsk_tns
