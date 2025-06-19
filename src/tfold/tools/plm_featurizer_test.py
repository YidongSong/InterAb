"""Unit-tests for the <PlmFeaturizer> class."""

import os
import logging

import torch

from tfold.utils import tfold_init
from tfold.tools import PlmFeaturizer


def main():
    """Main entry."""

    # configurations
    aa_seqs = [
        'ACDFKL',
        'ACDFKL' * 2,
        'GESLKISXAASGFTISGYGMNWVRQAPGKGLEWISYISINSVTKHYADSVQGRFTISRGNAKNSLYLQMNSLRDEDTAVYYCARDQSAYDTSGYLTWGRGTLVTVSS',
    ]
    mdl_dpath = '/data/jonathanwu/Pre-trained.Models'
    path_list = [
        os.path.join(mdl_dpath, 'ProtTrans-models/prot_xlnet'),
        os.path.join(mdl_dpath, 'ProtTrans-models/prot_t5_xl_uniref50'),
        os.path.join(mdl_dpath, 'AntiBERTy-models/antiberty'),
        os.path.join(mdl_dpath, 'ESM-models/esm1b_t33_650M_UR50S.pt'),
        os.path.join(mdl_dpath, 'ESM-models/esm1v_t33_650M_UR90S_1.pt'),
        os.path.join(mdl_dpath, 'ESM-models/esm2_t33_650M_UR50D.pt'),
        os.path.join(mdl_dpath, 'ESM-models/esm2_t36_3B_UR50D.pt'),
    ]
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # test w/ different PLMs
    for path in path_list:
        # build a PLM featurizer
        logging.info('PLM: %s', path)
        plm_featurizer = PlmFeaturizer(path, device, debug=True)

        # test the PLM featurizer w/o random masks
        logging.info('=== random masks disabled ===')
        sfea_mat_list, pfea_tns_list = plm_featurizer.run(aa_seqs)
        for aa_seq, sfea_mat, pfea_tns in zip(aa_seqs, sfea_mat_list, pfea_tns_list):
            logging.info('sequence: %s (%d AAs)', aa_seq, len(aa_seq))
            logging.info('sfea_mat: %s', sfea_mat.shape)
            logging.info('pfea_tns: %s', pfea_tns.shape)

        # test the PLM featurizer w/ random masks
        logging.info('=== random masks enabled ===')
        sfea_mat_list, pfea_tns_list, mask_vec_list = plm_featurizer.run(aa_seqs, mask_prob=0.15)
        for aa_seq, sfea_mat, pfea_tns, mask_vec in zip(aa_seqs, sfea_mat_list, pfea_tns_list, mask_vec_list):
            logging.info('sequence: %s (%d AAs)', aa_seq, len(aa_seq))
            logging.info('sfea_mat: %s', sfea_mat.shape)
            logging.info('pfea_tns: %s', pfea_tns.shape)
            logging.info('mask_vec: %s', mask_vec.shape)


if __name__ == '__main__':
    main()
