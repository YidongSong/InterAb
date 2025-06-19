"""Unit-tests for the <ClashCheckerV2> class."""

import os
import logging
from timeit import default_timer as timer

import torch

from tfold.utils import tfold_init
from tfold.tools.clash_checker_v2 import ClashCheckerV2
from tfold.tools import PdbParser


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_iters = 16
    prot_id = 'T1024-D1'
    device = torch.device('cuda:0')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse 3D coordinates from native & decoy structures
    aa_seq, cord_tns_natv, cmsk_mat_natv, _, error_msg = \
        PdbParser.load(pdb_fpath_natv, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_natv}'
    aa_seq, cord_tns_decy, cmsk_mat_decy, _, error_msg = \
        PdbParser.load(pdb_fpath_decy, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_decy}'
    cord_tns_natv = cord_tns_natv.to(device)
    cmsk_mat_natv = cmsk_mat_natv.to(device)
    cord_tns_decy = cord_tns_decy.to(device)
    cmsk_mat_decy = cmsk_mat_decy.to(device)

    # check for steric clashes in native & decoy structures
    checker = ClashCheckerV2(debug=True)
    logging.info('checking for steric clashes in the native structure ...')
    loss, metrics = checker.run(aa_seq, cord_tns_natv, cmsk_mat_natv)
    logging.info('steric clash loss: %.4f', loss.item())
    logging.info('evaluation metrics: %s', metrics)
    logging.info('checking for steric clashes in the decoy structure ...')
    loss, metrics = checker.run(aa_seq, cord_tns_decy, cmsk_mat_decy)
    logging.info('steric clash loss: %.4f', loss.item())
    logging.info('evaluation metrics: %s', metrics)

    # measure the elapsed time per execution
    checker = ClashCheckerV2()  # disable the debug mode
    for _ in range(n_iters):
        checker.run(aa_seq, cord_tns_decy, cmsk_mat_decy)
    time_beg = timer()
    for _ in range(n_iters):
        checker.run(aa_seq, cord_tns_decy, cmsk_mat_decy)
    time_avg = (timer() - time_beg) / n_iters
    logging.info('average time per run: %.4f (s)', time_avg)


if __name__ == '__main__':
    main()
