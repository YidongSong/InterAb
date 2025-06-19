"""Unit-tests for the <PdbParser> class."""

import os
import logging

from tfold.utils import tfold_init
from tfold.tools import PdbParser
from tfold.tools import PdbAssessor


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    prot_id = 'T1024-D1'
    metric_name = 'GDT-TS'
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_src = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_dst = os.path.join(data_dir, f'{prot_id}_reformat.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse the PDB file
    aa_seq, cord_tns, cmsk_mat, meta_data, error_msg = \
        PdbParser.load(pdb_fpath_src, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath_src}'
    logging.info('sequence: %s', aa_seq)
    logging.info('cord_tns: %s / %s', cord_tns.shape, cord_tns.dtype)
    logging.info('cmsk_mat: %s / %s', cmsk_mat.shape, cmsk_mat.dtype)
    for key, val in meta_data.items():
        logging.info('meta_data/%s: %s', key, val)

    # save the PDB file w/ standardized format
    PdbParser.save(aa_seq, cord_tns, cmsk_mat, pdb_fpath_dst)
    logging.info('PDB file re-generated: %s', pdb_fpath_dst)

    # compare two PDB files
    assessor = PdbAssessor()
    score = assessor.run(fas_fpath, pdb_fpath_src, pdb_fpath_dst, metric_name)
    logging.info('%s: %.4f', metric_name, score)

    # parse the decoy PDB file predicted by AlphaFold
    prot_id = 'UniRef50_A0A009EU90'
    data_dir = '/apdcephfs/share_1594716/jonathanwu/Datasets/UniRef50-SD-408k'
    pdb_fpath = os.path.join(data_dir, 'pdb.files.decoy.af', f'{prot_id}.pdb')
    aa_seq, cord_tns, cmsk_mat, meta_data, error_msg = PdbParser.load(pdb_fpath, has_plddt=True)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
    logging.info('sequence: %s', aa_seq)
    logging.info('cord_tns: %s / %s', cord_tns.shape, cord_tns.dtype)
    logging.info('cmsk_mat: %s / %s', cmsk_mat.shape, cmsk_mat.dtype)
    for key, val in meta_data.items():
        logging.info('meta_data/%s: %s', key, val)

    # parse the decoy PDB file predicted by tFold (v10)
    prot_id = 'UniRef50_A0A009EU90'
    data_dir = '/apdcephfs/share_1594716/jonathanwu/Datasets/UniRef50-SD-408k'
    pdb_fpath = os.path.join(data_dir, 'pdb.files.decoy.tf_v10', f'{prot_id}.pdb')
    aa_seq, cord_tns, cmsk_mat, meta_data, error_msg = PdbParser.load(pdb_fpath, has_plddt=True)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
    logging.info('sequence: %s', aa_seq)
    logging.info('cord_tns: %s / %s', cord_tns.shape, cord_tns.dtype)
    logging.info('cmsk_mat: %s / %s', cmsk_mat.shape, cmsk_mat.dtype)
    for key, val in meta_data.items():
        logging.info('meta_data/%s: %s', key, val)


if __name__ == '__main__':
    main()
