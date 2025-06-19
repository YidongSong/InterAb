"""Unit-tests for the <PTmNet> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.modules.af2_smod.ptm_net import PTmNet


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 64
    n_resds_pri = 40
    n_resds_sec = n_resds - n_resds_pri
    config = {
        'n_dims_pfea': 256,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), dtype=torch.float32)
    asym_id = torch.cat([
        1 * torch.ones((n_resds_pri), dtype=torch.float32),
        2 * torch.ones((n_resds_sec), dtype=torch.float32),
    ], dim=0)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)
    logging.info('[inputs] asym_id: %s', asym_id.shape)

    # test w/ the <PTmNet> module
    module = PTmNet(**config)
    logging.info('=== w/o asymmetric unit ID ===')
    tmsc_dict = module(pfea_tns)
    inspect_data(tmsc_dict, name='tmsc_dict')
    logging.info('=== w/ asymmetric unit ID ===')
    tmsc_dict = module(pfea_tns, asym_id=asym_id)
    inspect_data(tmsc_dict, name='tmsc_dict')


if __name__ == '__main__':
    main()
