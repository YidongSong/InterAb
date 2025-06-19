"""Unit-tests for the <FramAnglNet> module."""

import random
import logging

import torch

from tfold.utils import tfold_init
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.modules.af2_smod.fram_angl_net import FramAnglNet
from tfold.modules.af2_smod.utils import init_qta_params


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 64
    config = {
        'n_dims_sfea': 384,
        'n_dims_encd': 32,
        'n_dims_hidd': 128,
        'aa_dep_tors': True,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    aa_seq = ''.join([random.choice(RESD_NAMES_1C) for _ in range(n_resds)])
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32)
    sfea_tns_init = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32)
    encd_tns = torch.randn((n_smpls, n_resds, config['n_dims_encd']), dtype=torch.float32)
    quat_tns, trsl_tns, _ = init_qta_params(n_smpls, n_resds, mode='random')
    logging.info('[inputs] aa_seq: %s', aa_seq)
    logging.info('[inputs] sfea_tns: %s', sfea_tns.shape)
    logging.info('[inputs] sfea_tns_init: %s', sfea_tns_init.shape)
    logging.info('[inputs] encd_tns: %s', encd_tns.shape)
    logging.info('[inputs] quat_tns: %s', quat_tns.shape)
    logging.info('[inputs] trsl_tns: %s', trsl_tns.shape)

    # test w/ the <FramAnglNet> module
    module = FramAnglNet(**config)
    quat_tns, trsl_tns, angl_tns, quat_tns_upd = \
        module(aa_seq, sfea_tns, sfea_tns_init, encd_tns, quat_tns, trsl_tns)
    logging.info('[outputs] quat_tns: %s', quat_tns.shape)
    logging.info('[outputs] trsl_tns: %s', trsl_tns.shape)
    logging.info('[outputs] angl_tns: %s', angl_tns.shape)
    logging.info('[outputs] quat_tns_upd: %s', quat_tns_upd.shape)


if __name__ == '__main__':
    main()
