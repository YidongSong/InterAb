"""Unit-tests for <AtomMapper>."""

import os
import random
import logging
from timeit import default_timer as timer

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.tools import AtomMapper
from tfold.tools import PdbParser


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    fas_fpath = os.path.join(curr_dir, 'examples/101m.fasta')
    pdb_fpath = os.path.join(curr_dir, 'examples/101m.pdb')
    chain_id = 'A'

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse atom coordinates
    aa_seq, cord_tns, cmsk_mat, meta_data, error_msg = \
        PdbParser.load(pdb_fpath, fas_fpath=fas_fpath, chain_id=chain_id)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_src}'
    logging.info('sequence: %s', aa_seq)
    logging.info('cord_tns: %s / %s', cord_tns.shape, cord_tns.dtype)
    logging.info('cmsk_mat: %s / %s', cmsk_mat.shape, cmsk_mat.dtype)

    # test w/ <AtomMapper>
    atom_mapper = AtomMapper()
    cord_tns_n3 = atom_mapper.run(aa_seq, cord_tns, frmt_src='n14-tf', frmt_dst='n3')
    cmsk_mat_n3 = atom_mapper.run(aa_seq, cmsk_mat, frmt_src='n14-tf', frmt_dst='n3')
    logging.info('cord_tns_n3: %s / %s', cord_tns_n3.shape, cord_tns_n3.dtype)
    logging.info('cmsk_mat_n3: %s / %s', cmsk_mat_n3.shape, cmsk_mat_n3.dtype)
    logging.info('cord_tns_n3[:2]:\n%s', cord_tns_n3[:2])
    cord_tns_n37 = atom_mapper.run(aa_seq, cord_tns, frmt_src='n14-tf', frmt_dst='n37')
    cmsk_mat_n37 = atom_mapper.run(aa_seq, cmsk_mat, frmt_src='n14-tf', frmt_dst='n37')
    logging.info('cord_tns_n37: %s / %s', cord_tns_n37.shape, cord_tns_n37.dtype)
    logging.info('cmsk_mat_n37: %s / %s', cmsk_mat_n37.shape, cmsk_mat_n37.dtype)
    logging.info('cord_tns_n37[:2]:\n%s', cord_tns_n37[:2])

    # validate the alternative routine w/ optimized implementation
    device = torch.device('cuda:0')
    time_v1, time_v2 = 0.0, 0.0  # accumulated time for different routines
    for _ in range(256):
        frmt_src, frmt_dst = random.sample(atom_mapper.atom_frmts, 2)
        n_atoms_src = int(frmt_src.split('-')[0][1:])
        n_atoms_dst = int(frmt_dst.split('-')[0][1:])
        cord_tns_src = torch.randn((len(aa_seq), n_atoms_src, 3), device=device)
        time_beg = timer()
        cord_tns_dst_v1 = atom_mapper.run(aa_seq, cord_tns_src, frmt_src, frmt_dst, method='v1')
        time_v1 += timer() - time_beg
        time_beg = timer()
        cord_tns_dst_v2 = atom_mapper.run(aa_seq, cord_tns_src, frmt_src, frmt_dst, method='v2')
        time_v2 += timer() - time_beg
        assert cord_tns_dst_v1.shape[1] == n_atoms_dst
        assert torch.norm(cord_tns_dst_v1 - cord_tns_dst_v2) < 1e-6
    logging.info('time consumption (ms): %.2f (v1) / %.2f (v2)', 1000.0 * time_v1, 1000.0 * time_v2)


if __name__ == '__main__':
    main()
