"""Unit-tests for <PairPredictor>."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.misc import PairPredictor


def main():
    """Main entry."""

    # configurations
    n_smpls = 4
    n_resds = 64
    config = {
        'n_dims_pfea': 128,
        'n_bins_list': [37, 25, 25, 25],
    }
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # build pair features
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), device=device)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)

    # test w/ <PairPredictor>
    module = PairPredictor(**config).to(device)
    logt_tns_cb, logt_tns_om, logt_tns_th, logt_tns_ph = module(pfea_tns)
    logging.info('[outputs] logt_tns_cb: %s', logt_tns_cb.shape)
    logging.info('[outputs] logt_tns_om: %s', logt_tns_om.shape)
    logging.info('[outputs] logt_tns_th: %s', logt_tns_th.shape)
    logging.info('[outputs] logt_tns_ph: %s', logt_tns_ph.shape)


if __name__ == '__main__':
    main()
