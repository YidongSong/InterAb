"""Antibody structure assessor.

Notes:
* For TCR inputs, the beta chain ID must be 'B' and the alpha chain ID must be 'A'.
"""

import math
from collections import defaultdict

import torch
from anarci import anarci

from tfold.utils import kabsch
from tfold.utils import parse_fas_file_mult
from tfold.tools.pdb_parser import PdbParser
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


class TCRAssessor():  # pylint: disable=too-few-public-methods
    """TCR structure assessor."""

    def __init__(self, atom_set='bb', align_set='fr', scheme='imgt'):
        """Constructor function."""

        # setup configurations
        self.atom_set = atom_set
        self.align_set = align_set
        self.scheme = scheme
        assert self.atom_set in ['ca', 'bb', 'fa'], f'unrecognized atom set: {self.atom_set}'
        assert self.align_set in ['all', 'fr'], f'unrecognized alignment set: {self.align_set}'
        assert self.scheme in ['imgt'], f'unrecognized renumber scheme: {self.scheme}'

        # additional configurations
        self.eps = 1e-6
        self.regions = ['FR', 'CDR1', 'CDR2', 'CDR3']
        if self.scheme == 'imgt':
            self.cdr_bnds_dict = {
                'B': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
                'A': {'CDR1': (27, 38), 'CDR2': (56, 65), 'CDR3': (105, 117)},
            }
        else:
            raise ValueError(f'unrecognized renumber scheme: {self.scheme}')

    @torch.no_grad()
    def run(self, fas_fpath, pdb_fpath_natv, pdb_fpath_decy):  # pylint: disable=too-many-locals
        """Run the antibody structure assessor."""

        # parse the FASTA file
        aa_seq_dict = parse_fas_file_mult(fas_fpath)
        chain_ids = [x[-1] for x in aa_seq_dict]
        assert len(chain_ids) == 2, 'Alpha and Beta chain should be paired'

        # parse native & decoy PDB files
        assert all([chain_id in ['B', 'A'] for chain_id in chain_ids])
        metrics = {}
        for chain_id in chain_ids:
            metrics_raw = self.__calc_metrics(
                aa_seq_dict[chain_id], pdb_fpath_natv, pdb_fpath_decy, chain_id)
            metrics.update({f'{k}-TCR-{chain_id}': v for k, v in metrics_raw.items()})

        return metrics

    def __calc_metrics(self, aa_seq, pdb_fpath_natv, pdb_fpath_decy, chain_id):
        """Calculate evaluation metrics for a single chain (TCR-B, TCR-A)."""

        # initialization
        metrics = {}

        # get region annotations (framework or CDRs)
        resd_list = self.__get_regions(aa_seq)
        fmsk_vec = torch.tensor([1 if x[1] == 'FR' else 0 for x in resd_list], dtype=torch.int8)

        # parse native & decoy PDB files
        _, cord_tns_natv, cmsk_mat_natv, _, error_msg = \
            PdbParser.load(pdb_fpath_natv, chain_id=chain_id, aa_seq=aa_seq)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_natv} ({error_msg})'
        _, cord_tns_decy, cmsk_mat_decy, _, error_msg = \
            PdbParser.load(pdb_fpath_decy, chain_id=chain_id, aa_seq=aa_seq)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_decy} ({error_msg})'

        # get per-atom validness masks
        cmsk_mat_natv = self.__filter_atoms(aa_seq, cmsk_mat_natv)
        cmsk_mat_decy = self.__filter_atoms(aa_seq, cmsk_mat_decy)
        cmsk_mat = cmsk_mat_natv * cmsk_mat_decy
        assert torch.sum(cmsk_mat).item() == torch.sum(cmsk_mat_natv).item(), \
            'one or more atoms are missing in the decoy structure'

        # centralize native & decoy structures
        n_atoms = torch.sum(cmsk_mat).item()
        assert n_atoms >= 3, 'at least 3 atoms are needed for the structure alignment'
        cord_vec_natv = torch.sum(cmsk_mat.unsqueeze(dim=2) * cord_tns_natv, dim=(0, 1)) / n_atoms
        cord_vec_decy = torch.sum(cmsk_mat.unsqueeze(dim=2) * cord_tns_decy, dim=(0, 1)) / n_atoms
        cord_tns_natv = cord_tns_natv - cord_vec_natv.view(1, 1, 3)
        cord_tns_decy = cord_tns_decy - cord_vec_decy.view(1, 1, 3)

        # calculate per-atom errors under the optimal transformation (from decoy to native)
        rota_mat = self.__calc_rota_mat(cord_tns_natv, cord_tns_decy, cmsk_mat, fmsk_vec)
        cord_tns_decy = torch.sum(rota_mat.view(1, 1, 3, 3) * cord_tns_decy.unsqueeze(dim=2), dim=3)
        cerr_mat = torch.sum(torch.square(cord_tns_decy - cord_tns_natv), dim=2)  # squared L2-norm

        # calculate the overall RMSD value
        metrics['RMSD-All'] = torch.sqrt(torch.sum(cmsk_mat * cerr_mat) / n_atoms).item()

        # calculate per-region RMSD values
        rmsd_num_dict = defaultdict(float)  # numerators for RMSD calcuation
        rmsd_den_dict = defaultdict(float)  # denominators for RMSD calculation
        for idx_resd, (_, region) in enumerate(resd_list):
            if region is None:
                continue
            rmsd_num_dict[region] += torch.sum(cmsk_mat[idx_resd] * cerr_mat[idx_resd]).item()
            rmsd_den_dict[region] += torch.sum(cmsk_mat[idx_resd]).item()
        for region in self.regions:
            metrics[f'RMSD-{region}'] = \
                math.sqrt(rmsd_num_dict[region] / (rmsd_den_dict[region] + self.eps))

        return metrics

    def __get_regions(self, aa_seq_all):
        """Get region annotations (framework or CDRs)."""

        # run ANARCI
        numbering, details, _ = anarci([('null', aa_seq_all)], scheme=self.scheme, output=False)
        aa_seq_sel = ''.join([x[1] for x in numbering[0][0][0] if x[1] != '-'])
        chn_type = details[0][0]['chain_type']
        cdr_bnds = self.cdr_bnds_dict[chn_type]
        assert aa_seq_sel in aa_seq_all
        n_resds_lp = aa_seq_all.index(aa_seq_sel)  # padding on the left side
        n_resds_rp = len(aa_seq_all) - len(aa_seq_sel) - n_resds_lp  # padding on the right side

        # parse ANARCI's outputs
        resd_list = []
        if n_resds_lp != 0:
            resd_list.extend([(x, None) for x in aa_seq_all[:n_resds_lp]])
        for (idx_resd, _), resd_name in numbering[0][0][0]:
            if resd_name == '-':
                continue
            rgn_name = 'FR'  # default region name
            for cdr_name, (idx_resd_beg, idx_resd_end) in cdr_bnds.items():
                if idx_resd_beg <= idx_resd <= idx_resd_end:
                    rgn_name = cdr_name
                    break
            resd_list.append((resd_name, rgn_name))
        if n_resds_rp != 0:
            resd_list.extend([(x, None) for x in aa_seq_all[-n_resds_rp:]])
        assert ''.join([x[0] for x in resd_list]) == aa_seq_all

        return resd_list

    def __filter_atoms(self, aa_seq, cmsk_mat):
        """Filter atoms based the specified atom set."""

        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            for idx_atom, atom_name in enumerate(atom_names):
                if (self.atom_set == 'ca') and (atom_name != 'CA'):
                    cmsk_mat[idx_resd, idx_atom] = 0
                elif (self.atom_set == 'bb') and (atom_name not in ['N', 'CA', 'C']):
                    cmsk_mat[idx_resd, idx_atom] = 0

        return cmsk_mat

    def __calc_rota_mat(self, cord_tns_natv, cord_tns_decy, cmsk_mat, fmsk_vec):
        """Calculate the optimal rotation for aligning the decoy structure with the native one."""

        # find atom indices for structure alignment
        if self.align_set == 'fr':
            idxs_vld = torch.nonzero((fmsk_vec.unsqueeze(dim=1) * cmsk_mat).view(-1))[:, 0]
        elif self.align_set == 'all':
            idxs_vld = torch.nonzero(cmsk_mat.view(-1))[:, 0]
        else:
            raise ValueError(f'unrecognized alignment set: {self.align_set}')

        # calculate the optimal rotation
        cord_mat_natv = cord_tns_natv.view(-1, 3)[idxs_vld]
        cord_mat_decy = cord_tns_decy.view(-1, 3)[idxs_vld]
        rota_mat = kabsch(cord_mat_decy, cord_mat_natv)

        return rota_mat

    def get_regions(self, aa_seq):
        """Get region annotations (framework or CDRs)."""

        resd_list = self.__get_regions(aa_seq)
        regions = {'fr': [], 'cdr1': [], 'cdr2': [], 'cdr3': []}

        for idx, (_, region) in enumerate(resd_list):
            if region is None:
                continue
            regions[region.lower()].append(idx)
            if region.lower().startswith('cdr'):
                regions['cdr'].append(idx)

        return regions
