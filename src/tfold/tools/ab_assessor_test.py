"""Unit-tests for the <AbAssessor> class."""

import os
import logging

from tfold.utils import tfold_init
from tfold.tools.ab_assessor import AbAssessor


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples/ab_assessor')

    # initialization
    tfold_init()

    # initialize an assessor for antibody & nanobody structure predictions
    ab_assessor = AbAssessor(atom_set='bb', align_set='fr', scheme='chothia')

    # run <AbAssessor> to evaluate antibody structure predictions
    prot_id = '7df1_G_K'
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')
    metrics = ab_assessor.run(fas_fpath, pdb_fpath_natv, pdb_fpath_decy)
    logging.info('evaluation results for antibody <%s>:', prot_id)
    for key, val in metrics.items():
        logging.info('%s: %.4f', key, val)

    # run <AbAssessor> to evaluate nanobody structure predictions
    prot_id = '7b5g_H'
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')
    metrics = ab_assessor.run(fas_fpath, pdb_fpath_natv, pdb_fpath_decy)
    logging.info('evaluation results for nanobody <%s>:', prot_id)
    for key, val in metrics.items():
        logging.info('%s: %.4f', key, val)


if __name__ == '__main__':
    main()
