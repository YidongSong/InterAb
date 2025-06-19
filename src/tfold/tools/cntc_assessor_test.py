"""Unit-tests for the <CntcAssessor> class."""

import os
import logging

import numpy as np
import torch

from tfold.utils import tfold_init
from tfold.tools import CntcAssessor
from tfold.tools import PdbParser
from tfold.tools import ProtStruct


def build_cord_from_pdb(fas_fpath, pdb_fpath):
    """Build 3D coordinates for CB (CA for Glycine) atoms."""

    aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, 'failed to parse the PDB file: ' + pdb_fpath
    cord_mat_ca = ProtStruct.get_atoms(aa_seq, cord_tns, ['CA'])
    cmsk_vec_ca = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])
    cord_mat_cb = ProtStruct.get_atoms(aa_seq, cord_tns, ['CB'])
    cmsk_vec_cb = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CB'])
    is_gly = torch.tensor([1 if x == 'G' else 0 for x in aa_seq], dtype=torch.int8)
    cord_mat = is_gly.unsqueeze(dim=1) * cord_mat_ca + (1 - is_gly).unsqueeze(dim=1) * cord_mat_cb
    cmsk_vec = is_gly * cmsk_vec_ca + (1 - is_gly) * cmsk_vec_cb

    return cord_mat, cmsk_vec


def build_labl_from_npz(path):
    """Build categorical labels for inter-residue contacts."""

    # configurations
    n_bins = 37
    dist_min = 2.0
    dist_max = 20.0
    bin_wid = (dist_max - dist_min) / (n_bins - 1)

    # build categorical labels for inter-residue contacts
    with np.load(path) as npz_data:
        dist_mat = npz_data['cb-val']
        mask_mat = npz_data['cb-msk']
        labl_mat = np.clip(
            np.floor((dist_mat - dist_min) / bin_wid).astype(np.int64), 0, n_bins - 1)
        labl_mat = (labl_mat + 1) % n_bins  # move the non-contact bin to the first one

    return labl_mat, mask_mat


def main():
    """Main entry."""

    # configurations
    prot_id = 'T1024-D1'
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath = os.path.join(data_dir, f'{prot_id}_native.pdb')
    npz_fpath_labl = os.path.join(data_dir, f'{prot_id}_da_labl.npz')
    npz_fpath_pred = os.path.join(data_dir, f'{prot_id}_da_pred.npz')

    # initialization
    tfold_init(verb_levl='DEBUG')
    assessor = CntcAssessor()

    # restore predicted inter-residue contact probabilities
    with np.load(npz_fpath_pred) as npz_data:
        logging.info('pairwise sum: %.4f', np.mean(np.sum(npz_data['dist'], axis=2)))
        prob_mat = np.sum(npz_data['dist'][:, :, 1:13], axis=-1)

    # calculate the top-L precision for long-range contact predictions from 3D coordinates
    cord_mat, cmsk_vec = build_cord_from_pdb(fas_fpath, pdb_fpath)
    prec = assessor.calc_prec_w_cord(cord_mat, cmsk_vec, prob_mat)
    logging.info('Top-L precision w/ 3D coordinates: %.4f', prec)

    # calculate the top-L precision for long-range contact predictions from categorical labels
    labl_mat, lmsk_mat = build_labl_from_npz(npz_fpath_labl)
    prec = assessor.calc_prec_w_labl(labl_mat, lmsk_mat, prob_mat)
    logging.info('Top-L precision w/ categorical labels: %.4f', prec)


if __name__ == '__main__':
    main()
