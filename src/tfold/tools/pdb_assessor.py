"""Assessor for decoy PDB files."""

import os
import shutil
import logging
import subprocess

import torch

from tfold.utils import get_rand_str
from tfold.utils import get_tmp_dpath
from tfold.utils import kabsch
from tfold.tools.pdb_parser import PdbParser
from tfold.tools.prot_struct import ProtStruct


class PdbAssessor():
    """Assessor for decoy PDB files.

    Available built-in evaluation metrics:
    > Adj-CA-Dev (deviation of ideal CA-CA distance in adjacent residues)
    > RMSD-CA (root of mean squared deviation for CA atoms' 3D coordinates)
    > RMSD-FA (root of mean squared deviation for full atoms' 3D coordinates)

    Available external evaluation metrics:
    > GDT-TS
    > TM-Scr
    > lDDT-CA-C (CA-atom w/ steric clash checks)
    > lDDT-CA-F (CA-atom w/o steric clash checks)
    > lDDT-FA-C (full-atom w/ steric clash checks)
    > lDDT-FA-F (full-atom w/o steric clash checks)
    """

    def __init__(self):
        """Constructor function."""

        # path to the steric clash check's parameters
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        prm_fpath = os.path.join(curr_dir, 'data/stereo_chemical_props.txt')

        # setup configurations
        self.eps = 1e-6
        self.dist_adj_ca = 3.8  # ideal CA-CA distance in adjacent residues
        self.args_chck = f' -a 15 -b 15 -r 15 -f -p {prm_fpath}'  # for lDDT evaluation
        self.tmp_dpath = get_tmp_dpath()
        logging.info('temporary directory: %s', self.tmp_dpath)


    @property
    def metric_names(self):
        """Get a list of available evaluation metric names."""

        return self.metric_names_builtin + self.metric_names_external


    @property
    def metric_names_builtin(self):
        """Get a list of built-in evaluation metric names."""

        return ['Adj-CA-Dev', 'RMSD-CA', 'RMSD-FA']


    @property
    def metric_names_external(self):
        """Get a list of external evaluation metric names."""

        return ['GDT-TS', 'TM-Scr', 'lDDT-CA-C', 'lDDT-CA-F', 'lDDT-FA-C', 'lDDT-FA-F']


    def run(
            self, fas_fpath, pdb_fpath_natv, pdb_fpath_decy,
            metric_name, metrics_dict=None, skip_prep=False,
        ):  # pylint: disable=too-many-arguments
        """Evaluate the PDB file w/ the specified metric.

        Args:
        * fas_fpath: path to the FASTA file
        * pdb_fpath_natv: path to the native PDB file
        * pdb_fpath_decy: path to the decoy PDB file
        * metric_name: evaluation metric name
        * metrics_dict: (optional) dict of evaluation metrics (key: (path, metric)-tuple)
        * skip_prep: (optional) whether PDB file preparation can be safely skipped

        Returns:
        * val: evaluation metric's value
        """

        # initialization
        key = (pdb_fpath_decy, metric_name)  # this will be used by <metrics_dict>
        assert metric_name in self.metric_names, f'unrecognized evaluation metric: {metric_name}'

        # prepare PDB files for upcoming evaluation
        if not skip_prep:
            pdb_fpath_natv, pdb_fpath_decy = \
                self.__prep_pdb_files(fas_fpath, pdb_fpath_natv, pdb_fpath_decy)

        # call <DeepScore> to compute GDT-TS & TM-Score
        if metric_name in ['GDT-TS', 'TM-Scr']:
            cmd_str = f'DeepScore {pdb_fpath_decy} {pdb_fpath_natv} -P 0 -n -2'
            cmd_out = subprocess.check_output(cmd_str, shell=True)
            val = self.__parse_dpsc_outputs(cmd_out, metric_name)

        # call <lDDT> to compute lDDT-CA/FA-C/F scores
        if metric_name in ['lDDT-CA-C', 'lDDT-CA-F', 'lDDT-FA-C', 'lDDT-FA-F']:
            args = '-c' if metric_name.split('-')[1] == 'CA' else ''  # atom set
            if metric_name.endswith('C'):  # steric clash check enabled
                args += self.args_chck
            cmd_str = f'lddt {args} {pdb_fpath_decy} {pdb_fpath_natv}'
            cmd_out = subprocess.check_output(cmd_str, shell=True)
            val = self.__parse_lddt_outputs(cmd_out)

        # RMSD for CA/full atoms
        if metric_name in ['RMSD-CA', 'RMSD-FA']:
            atom_set = 'ca' if metric_name.split('-')[1] == 'CA' else 'fa'
            val = self.__calc_rmsd(fas_fpath, pdb_fpath_natv, pdb_fpath_decy, atom_set)

        # deviation of ideal CA-CA distance in adjacent residues
        if metric_name == 'Adj-CA-Dev':
            val = self.__calc_adj_ca_dev(fas_fpath, pdb_fpath_decy)

        # record the evaluation metric's value
        if metrics_dict is not None:
            metrics_dict[key] = val

        return val


    def run_batch(self, fas_fpath, pdb_fpath_natv, pdb_fpath_decy, metric_names):
        """Evaluate the PDB file w/ specified metrics in the batch mode.

        Args:
        * fas_fpath: path to the FASTA file
        * pdb_fpath_natv: path to the native PDB file
        * pdb_fpath_decy: path to the decoy PDB file
        * metric_names: list of evaluation metric names

        Returns:
        * metrics_dict: dict of evaluation metric values (key: metric name)
        """

        # prepare PDB files for upcoming evaluation
        pdb_fpath_natv, pdb_fpath_decy = \
            self.__prep_pdb_files(fas_fpath, pdb_fpath_natv, pdb_fpath_decy)

        # evaluate the PDB file w/ specified metrics in the batch mode
        metrics_dict = {}
        for metric_name in metric_names:
            val = self.run(fas_fpath, pdb_fpath_natv, pdb_fpath_decy, metric_name, skip_prep=True)
            metrics_dict[metric_name] = val

        return metrics_dict


    def clear(self):
        """Clear-up temporary files.

        Note:
        * This should be called after all the evaluation processes are finished.
        """

        shutil.rmtree(self.tmp_dpath)


    def __prep_pdb_files(self, fas_fpath, pdb_fpath_natv, pdb_fpath_decy):
        """Prepare PDB files for upcoming evaluation."""

        # initialization
        rand_str = get_rand_str()

        # re-number residues in native & decoy PDB files
        os.makedirs(self.tmp_dpath, exist_ok=True)
        pdb_fpath_naln = os.path.join(self.tmp_dpath, f'{rand_str}_naln.pdb')
        pdb_fpath_daln = os.path.join(self.tmp_dpath, f'{rand_str}_daln.pdb')
        self.__renumber_residues(fas_fpath, pdb_fpath_natv, pdb_fpath_naln)
        self.__renumber_residues(fas_fpath, pdb_fpath_decy, pdb_fpath_daln)

        return pdb_fpath_naln, pdb_fpath_daln


    @classmethod
    def __renumber_residues(cls, fas_fpath, pdb_fpath_src, pdb_fpath_dst):
        """Re-number residues in the same order as in the FASTA file."""

        aa_seq, cord_tns, cmsk_mat, _, error_msg = \
            PdbParser.load(pdb_fpath_src, fas_fpath=fas_fpath)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_src} ({error_msg})'
        PdbParser.save(aa_seq, cord_tns, cmsk_mat, pdb_fpath_dst)


    @classmethod
    def __parse_dpsc_outputs(cls, cmd_out, metric_name):
        """Parse <DeepScore> output string to obtain the corresponding score."""

        try:
            line_str = cmd_out.decode('utf-8')
            if metric_name == 'GDT-TS':
                val = float(line_str.split()[14])
            else:  # then <metric_name> must be 'TM-Scr'
                val = float(line_str.split()[11])
        except UnicodeDecodeError:
            val = 0.0

        return val


    @classmethod
    def __parse_lddt_outputs(cls, cmd_out):
        """Parse <lddt> output strings to obtain the global lDDT score."""

        try:
            line_strs = cmd_out.decode('utf-8').split('\n')
            for line_str in line_strs:
                if line_str.startswith('Global LDDT score'):
                    val = float(line_str.split()[-1])
                    break
        except UnicodeDecodeError:
            val = 0.0

        return val


    @classmethod
    def __calc_rmsd(cls, fas_fpath, pdb_fpath_natv, pdb_fpath_decy, atom_set):  # pylint: disable=too-many-locals
        """Calculate the RMSD score on CA/full-atom."""

        # parse PDB files
        aa_seq, cord_tns_natv, cmsk_mat_natv, _, error_msg = \
            PdbParser.load(pdb_fpath_natv, fas_fpath=fas_fpath)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_natv} ({error_msg})'
        _, cord_tns_decy, cmsk_mat_decy, _, error_msg = \
            PdbParser.load(pdb_fpath_decy, fas_fpath=fas_fpath)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_decy} ({error_msg})'

        # choose atoms' (CA/full) 3D coordinates & validness masks
        if atom_set == 'ca':
            cord_mat_natv = ProtStruct.get_atoms(aa_seq, cord_tns_natv, ['CA'])
            cord_mat_decy = ProtStruct.get_atoms(aa_seq, cord_tns_decy, ['CA'])
            cmsk_vec_natv = ProtStruct.get_atoms(aa_seq, cmsk_mat_natv, ['CA'])
            cmsk_vec_decy = ProtStruct.get_atoms(aa_seq, cmsk_mat_decy, ['CA'])
            cmsk_vec = cmsk_vec_natv * cmsk_vec_decy
            idxs_atom = torch.nonzero(cmsk_vec, as_tuple=True)[0]
            cord_mat_natv = cord_mat_natv[idxs_atom]
            cord_mat_decy = cord_mat_decy[idxs_atom]
        elif atom_set == 'fa':
            cmsk_vec = cmsk_mat_natv.view(-1) * cmsk_mat_decy.view(-1)
            idxs_atom = torch.nonzero(cmsk_vec, as_tuple=True)[0]
            cord_mat_natv = cord_tns_natv.view(-1, 3)[idxs_atom]
            cord_mat_decy = cord_tns_decy.view(-1, 3)[idxs_atom]
        else:
            raise ValueError(f'unrecognized atom set: {atom_set}')

        # calculate the optimal rotation & translation
        dcrd_mat_natv = cord_mat_natv - torch.mean(cord_mat_natv, dim=0, keepdim=True)
        dcrd_mat_decy = cord_mat_decy - torch.mean(cord_mat_decy, dim=0, keepdim=True)
        rota_mat = kabsch(dcrd_mat_decy, dcrd_mat_natv)
        dcrd_mat_daln = torch.sum(rota_mat.unsqueeze(dim=0) * dcrd_mat_decy.unsqueeze(dim=1), dim=2)
        rmsd = torch.mean(torch.norm(dcrd_mat_daln - dcrd_mat_natv, dim=1)).item()

        return rmsd


    def __calc_adj_ca_dev(self, fas_fpath, pdb_fpath):
        """Calculate the deviation of ideal CA-CA distance in adjacent residues."""

        # parse the PDB file
        aa_seq, cord_tns, cmsk_mat, _, error_msg = \
            PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath} ({error_msg})'

        # extract 3D coordinates of CA atoms
        atom_names = ['CA']
        cord_mat = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)
        cmsk_vec = ProtStruct.get_atoms(aa_seq, cmsk_mat, atom_names)

        # calculate CA-CA distance in adjacent residues
        derr_vec = torch.abs(torch.norm(cord_mat[:-1] - cord_mat[1:], dim=1) - self.dist_adj_ca)
        dmsk_vec = cmsk_vec[:-1] * cmsk_vec[1:]

        # calculate the deviation of ideal CA-CA distance in adjacent residues
        score = torch.sum(dmsk_vec * derr_vec) / (torch.sum(dmsk_vec) + self.eps)

        return score.item()
