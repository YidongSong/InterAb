"""Unit-tests for the <DaLablBuilder> class."""

import os
import logging

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.tools import PdbParser
from tfold.tools import DaLablBuilder


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

    # parse the PDB file
    aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
    logging.info('sequence: %s', aa_seq)
    logging.info('cord_tns: %s / %s', cord_tns.shape, cord_tns.dtype)
    logging.info('cmsk_mat: %s / %s', cmsk_mat.shape, cmsk_mat.dtype)

    # test w/ the <DaLablBuilder> class
    da_labl_builder = DaLablBuilder()
    da_labl_dict = da_labl_builder.run(aa_seq, cord_tns, cmsk_mat)
    inspect_data(da_labl_dict, name='da_labl_dict')


if __name__ == '__main__':
    main()
