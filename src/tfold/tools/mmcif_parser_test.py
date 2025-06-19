"""Unit-tests for <mmCIFParser>."""

import os
import random
import logging

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.utils import find_files_by_suffix
from tfold.tools.mmcif_parser import mmCIFParser


def main():
    """Main entry."""

    # configurations
    n_files = 16  # number of mmCIF files for unit-tests
    cif_dpath = '/data/jonathanwu/Datasets/RCSB-PDB-20230102/snapshot'
    cif_fpaths = find_files_by_suffix(cif_dpath, suffix='.cif.gz')

    # initialization
    tfold_init()
    random.seed(42)

    # parse all the mmCIF files
    parser = mmCIFParser()
    cif_fpaths = random.sample(cif_fpaths, n_files)
    for cif_fpath in cif_fpaths:
        logging.info('parsing the mmCIF file: %s', cif_fpath)
        prot_data, meta_data, error_msg = parser.load(cif_fpath)
        inspect_data(prot_data, name='prot_data')
        logging.info('meta data: %s', meta_data)
        logging.info('parsing error: %s', error_msg)


if __name__ == '__main__':
    main()
