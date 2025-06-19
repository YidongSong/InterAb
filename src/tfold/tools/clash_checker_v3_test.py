"""Unit-tests for the <ClashCheckerV3> class."""

import os
import logging
from collections import OrderedDict
from timeit import default_timer as timer

import torch

from tfold.utils import tfold_init
from tfold.utils import parse_fas_file_mult
from tfold.tools.clash_checker_v3 import ClashCheckerV3
from tfold.tools import PdbParser


def get_inputs(fas_fpath, pdb_fpath, device):
    """Get input tensors for structural violation checks."""

    aa_seq_dict = parse_fas_file_mult(fas_fpath)
    prot_data = OrderedDict()
    for chain_id, aa_seq in aa_seq_dict.items():
        _, cord_tns, cmsk_mat, _, error_msg = \
            PdbParser.load(pdb_fpath, aa_seq=aa_seq, chain_id=chain_id)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath} / {chain_id}'
        prot_data[chain_id] = {'seq': aa_seq, 'cord': cord_tns, 'cmsk': cmsk_mat}

    chain_ids = list(aa_seq_dict.keys())
    aa_seq = ''.join([prot_data[x]['seq'] for x in chain_ids])
    cord_tns = torch.cat([prot_data[x]['cord'] for x in chain_ids], dim=0).to(device)
    cmsk_mat = torch.cat([prot_data[x]['cmsk'] for x in chain_ids], dim=0).to(device)
    n_resds_list = [len(prot_data[x]['seq']) for x in chain_ids]
    asym_id = torch.cat([
        (idx + 1) * torch.ones(n_resds) for idx, n_resds in enumerate(n_resds_list)
    ], dim=0).to(device)

    return aa_seq, cord_tns, cmsk_mat, asym_id


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_iters = 16
    prot_id = '7wd1_D'
    device = torch.device('cuda:0')
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # check for steric clashes in native & decoy structures
    checker = ClashCheckerV3(debug=True)
    logging.info('checking for steric clashes in the native structure ...')
    aa_seq, cord_tns, cmsk_mat, asym_id = get_inputs(fas_fpath, pdb_fpath_natv, device)
    loss, metrics = checker.run(aa_seq, cord_tns, cmsk_mat, asym_id)
    logging.info('steric clash loss: %.4f', loss.item())
    logging.info('evaluation metrics: %s', metrics)
    logging.info('checking for steric clashes in the decoy structure ...')
    aa_seq, cord_tns, cmsk_mat, asym_id = get_inputs(fas_fpath, pdb_fpath_decy, device)
    loss, metrics = checker.run(aa_seq, cord_tns, cmsk_mat, asym_id)
    logging.info('steric clash loss: %.4f', loss.item())
    logging.info('evaluation metrics: %s', metrics)

    raise NotImplementedError

    # measure the elapsed time per execution
    checker = ClashCheckerV3()  # disable the debug mode
    for _ in range(n_iters):
        checker.run(aa_seq, cord_tns_decy, cmsk_mat_decy)
    time_beg = timer()
    for _ in range(n_iters):
        checker.run(aa_seq, cord_tns_decy, cmsk_mat_decy)
    time_avg = (timer() - time_beg) / n_iters
    logging.info('average time per run: %.4f (s)', time_avg)


if __name__ == '__main__':
    main()
