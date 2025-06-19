"""Unit-tests for the <PdbAssessor> class."""

import os
import logging
from multiprocessing import Manager, Pool

from tfold.utils import tfold_init
from tfold.tools import PdbAssessor


def main():
    """Main entry."""

    # configurations
    n_threads = 6
    prot_id = 'T1024-D1'
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')
    assessor = PdbAssessor()
    metric_names = assessor.metric_names

    # evaluate the decoy PDB file w/ various metrics
    args_list = []
    metrics = Manager().dict()
    for metric_name in metric_names:
        args_list.append((fas_fpath, pdb_fpath_natv, pdb_fpath_decy, metric_name, metrics))
    with Pool(processes=n_threads) as pool:
        pool.starmap(assessor.run, args_list)

    # display evaluation results
    for metric_name in metric_names:
        logging.info('%s: %.4f', metric_name, metrics[(pdb_fpath_decy, metric_name)])


if __name__ == '__main__':
    main()
