"""Unit-tests for the <ProtConverter> class."""

import os
import logging
from timeit import default_timer as timer

import torch
import numpy as np

from tfold.utils import tfold_init
from tfold.tools import PdbParser
from tfold.tools import ProtConverter


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    eps = 1e-6
    prot_id = 'T1024-D1'
    device = torch.device('cpu')
    #device = torch.device('cuda:0')
    n_repts = 16  # number of repeated runs for measuring the time consumption
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath = os.path.join(data_dir, f'{prot_id}_native.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse the PDB file
    aa_seq, cord_tns_base, cmsk_mat_base, _, error_msg = \
        PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
    cord_tns_base = cord_tns_base.to(device)
    cmsk_mat_base = cmsk_mat_base.to(device)

    # test w/ <ProtConverter>
    converter = ProtConverter()
    fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat = \
        converter.cord2fa(aa_seq, cord_tns_base, cmsk_mat_base)
    cord_tns_reco, cmsk_mat_reco = \
        converter.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat)

    # calculate the overall coordinate RMSD
    dist_mat = torch.norm(cord_tns_reco - cord_tns_base, dim=-1)
    cmsk_mat = cmsk_mat_base * cmsk_mat_reco
    rmsd = torch.sum(cmsk_mat * dist_mat) / (torch.sum(cmsk_mat) + eps)
    logging.info('coordinate RMSD: %.4f', rmsd.item())

    # compare backbone & side-chain local frames
    fram_tns_base, fmsk_mat_base = converter.cord2fram(aa_seq, cord_tns_base, cmsk_mat_base)
    fram_tns_reco, fmsk_mat_reco = converter.cord2fram(aa_seq, cord_tns_reco, cmsk_mat_reco)
    logging.info('cmsk_mat: %.4f', torch.norm((cmsk_mat_reco - cmsk_mat_base).float()).item())
    logging.info('fmsk_mat: %.4f', torch.norm((fmsk_mat_reco - fmsk_mat_base).float()).item())
    logging.info('fram_tns: %.4f', torch.mean(
        fmsk_mat_base * torch.norm(fram_tns_reco - fram_tns_base, dim=(2, 3))).item())

    # measure the time consumption of <cord2fa>
    time_vec = np.zeros((n_repts), dtype=np.float32)
    for idx_rept in range(n_repts):
        time_beg = timer()
        converter.cord2fa(aa_seq, cord_tns_base, cmsk_mat_base)
        time_vec[idx_rept] = 1000.0 * (timer() - time_beg)
    logging.info('cord2fa: %.2f +/- %.2f (ms)', np.mean(time_vec), np.std(time_vec))

    # measure the time consumption of <fa2cord>
    time_vec = np.zeros((n_repts), dtype=np.float32)
    for idx_rept in range(n_repts):
        time_beg = timer()
        converter.fa2cord(aa_seq, fram_tns_bb, fmsk_mat_bb, angl_tns, amsk_mat)
        time_vec[idx_rept] = 1000.0 * (timer() - time_beg)
    logging.info('fa2cord: %.2f +/- %.2f (ms)', np.mean(time_vec), np.std(time_vec))

    # measure the time consumption of <cord2fram>
    time_vec = np.zeros((n_repts), dtype=np.float32)
    for idx_rept in range(n_repts):
        time_beg = timer()
        converter.cord2fram(aa_seq, cord_tns_base, cmsk_mat_base)
        time_vec[idx_rept] = 1000.0 * (timer() - time_beg)
    logging.info('cord2fram: %.2f +/- %.2f (ms)', np.mean(time_vec), np.std(time_vec))


if __name__ == '__main__':
    main()
