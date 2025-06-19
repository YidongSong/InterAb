"""Unit-tests for the <ClashChecker> class."""

import os
import logging
from timeit import default_timer as timer

import torch

from tfold.utils import tfold_init
from tfold.tools import ClashChecker
from tfold.tools import PdbParser


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_iters = 4
    prot_id = 'T1024-D1'
    device = torch.device('cpu')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')
    checker = ClashChecker(device=device)

    # check for steric clashes in the native structure
    logging.info('checking for steric clashes in the native structure ...')
    aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath_natv, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_natv}'
    loss, atpr_list = checker.run(aa_seq, cord_tns, cmsk_mat, rtn_atpr_list=True)
    logging.info('steric clash loss: %.4f', loss.item())
    checker.disp_atpr_list(atpr_list)

    # check for steric clashes in the decoy structure
    logging.info('checking for steric clashes in the decoy structure ...')
    aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath_decy, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_decy}'
    loss, atpr_list = checker.run(aa_seq, cord_tns, cmsk_mat, rtn_atpr_list=True)
    logging.info('steric clash loss: %.4f', loss.item())
    checker.disp_atpr_list(atpr_list)

    # measure the elapsed time per execution
    for _ in range(n_iters):
        checker.run(aa_seq, cord_tns, cmsk_mat)
    time_beg = timer()
    for _ in range(n_iters):
        checker.run(aa_seq, cord_tns, cmsk_mat)
    time_avg = (timer() - time_beg) / n_iters
    logging.info('average time per run: %.4f (s)', time_avg)


if __name__ == '__main__':
    main()
