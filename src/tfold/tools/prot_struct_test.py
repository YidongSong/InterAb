"""Unit-tests for the <ProtStruct> class."""

import os
import logging
from collections import defaultdict

import torch
import numpy as np

from tfold.utils import tfold_init
from tfold.utils import cdist
from tfold.utils import quat2rot
from tfold.utils import rot2quat
from tfold.tools import PdbParser
from tfold.tools import ProtStruct
from tfold.tools import ProtConverter
from tfold.tools.prot_constants import N_ANGLS_PER_RESD
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


def init_params(aa_seq):
    """Randomly initialize QTA parameters."""

    # initialization
    n_resds = len(aa_seq)
    n_dims_quat = 4  # full quaternion vectors
    n_dims_trsl = 3  # 3D coordiantes
    n_dims_angl = 2  # cosine & sine values

    # randomly initialize QTA parameters
    params = {
        'quat': torch.randn((n_resds, n_dims_quat), dtype=torch.float32),
        'trsl': torch.randn((n_resds, n_dims_trsl), dtype=torch.float32),
        'angl': torch.randn((n_resds, N_ANGLS_PER_RESD, n_dims_angl), dtype=torch.float32),
    }
    params['rota'] = quat2rot(params['quat'])

    return params


def calc_drmsd(cord_tns_ref, cord_tns_qry, cmsk_mat):
    """Calculate dRMSD between reference and query structures."""

    dist_mat_ref = cdist(cord_tns_ref.view(-1, 3))
    dist_mat_qry = cdist(cord_tns_qry.view(-1, 3))
    dmsk_mat = cmsk_mat.view(-1, 1) * cmsk_mat.view(1, -1)
    drmsd = torch.sum(dmsk_mat * torch.abs(dist_mat_ref - dist_mat_qry)) / torch.sum(dmsk_mat)

    return drmsd


def calc_reco_err(pid_fpath, fas_dpath, pdb_dpath):  # pylint: disable=too-many-locals
    """"Calculate the reconstruction error for each atom type."""

    # initialization
    prot_struct_orig = ProtStruct()
    prot_struct_reco = ProtStruct()
    prot_converter = ProtConverter()

    # get protein IDs
    with open(pid_fpath, 'r', encoding='UTF-8') as i_file:
        prot_ids = [i_line.strip() for i_line in i_file]

    # calculate the reconstruction error for each protein ID
    cerr_dict = defaultdict(list)
    for prot_id in prot_ids:
        # build original & reconstructed protein structures
        fas_fpath = os.path.join(fas_dpath, f'{prot_id}.fasta')
        pdb_fpath = os.path.join(pdb_dpath, f'{prot_id}.pdb')
        prot_struct_orig.init_from_file(fas_fpath, pdb_fpath)
        prot_struct_orig.build_fram_n_angl(prot_converter, build_sc=True)
        params = {
            'quat': rot2quat(prot_struct_orig.fram_tns_bb[:, 0, :3], quat_type='full'),
            'trsl': prot_struct_orig.fram_tns_bb[:, 0, 3],
            'angl': prot_struct_orig.angl_tns,
        }
        params['rota'] = quat2rot(params['quat'])
        prot_struct_reco.init_from_param(prot_struct_orig.aa_seq, params, prot_converter)

        # calculate the reconstruction error for each atom type
        cerr_mat = torch.norm(prot_struct_reco.cord_tns - prot_struct_orig.cord_tns, dim=2)
        cmsk_mat = prot_struct_orig.cmsk_mat * prot_struct_reco.cmsk_mat
        for idx_resd, resd_name_1c in enumerate(prot_struct_orig.aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            for idx_atom, atom_name in enumerate(ATOM_NAMES_PER_RESD[resd_name_3c]):
                if cmsk_mat[idx_resd, idx_atom] == 1:
                    key = f'{resd_name_3c}-{atom_name}'
                    cerr_dict[key].append(cerr_mat[idx_resd, idx_atom].item())

    # summarize results
    cerr_list = [(k, np.mean(v).item()) for k, v in cerr_dict.items()]
    cerr_list.sort(key=lambda x: x[1])
    for atom_name, cerr_val in cerr_list:
        logging.info('%s: %.4f', atom_name, cerr_val)


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    prot_id = 'T1024-D1'
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath = os.path.join(data_dir, f'{prot_id}_native.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')
    prot_struct = ProtStruct()
    prot_converter = ProtConverter()

    # initialize the protein structure from FASTA & PDB files
    prot_struct.init_from_file(fas_fpath, pdb_fpath)
    prot_struct.summarize()

    # initialize the protein structure from 3D coordinates
    aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
    prot_struct.init_from_cord(aa_seq, cord_tns, cmsk_mat)
    prot_struct.summarize()

    # initialize the protein structure from QTA parameters
    params = init_params(aa_seq)
    prot_struct.init_from_param(aa_seq, params, prot_converter)
    prot_struct.summarize()

    # build backbone and/or side-chain local frames and torison angles from 3D coordinates
    prot_struct.init_from_file(fas_fpath, pdb_fpath)
    prot_struct.build_fram_n_angl(prot_converter, build_sc=True)
    prot_struct.summarize()

    # build the alternative pose by flipping all the symmetric torsion angles
    cord_tns_alt, _, _ = prot_struct.build_alt_pose(prot_converter)
    drmsd = calc_drmsd(prot_struct.cord_tns, cord_tns_alt, prot_struct.cmsk_mat)
    logging.info('dRMSD: %.4f', drmsd.item())

    # validate the reconstruction error for each atom
    data_dir = '/data/jonathanwu/datasets/CASP14'
    pid_fpath = os.path.join(data_dir, 'prot_ids.txt')
    fas_dpath = os.path.join(data_dir, 'fasta.files')
    pdb_dpath = os.path.join(data_dir, 'pdb.files.native')
    calc_reco_err(pid_fpath, fas_dpath, pdb_dpath)


if __name__ == '__main__':
    main()
