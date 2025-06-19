"""Unit-tests for the <InvPntAttn> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.af2_smod.inv_pnt_attn import InvPntAttn
from tfold.modules.af2_smod.utils import init_qta_params


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 64
    config = {
        'n_dims_sfea': 384,
        'n_dims_pfea': 256,
        'n_dims_attn': 16,
        'n_heads': 12,
        'n_qpnts': 4,
        'n_vpnts': 8,
        'drop_prob': 0.1,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), dtype=torch.float32)
    quat_tns, trsl_tns, _ = init_qta_params(n_smpls, n_resds, mode='random')
    logging.info('[inputs] sfea_tns: %s', sfea_tns.shape)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)
    logging.info('[inputs] quat_tns: %s', quat_tns.shape)
    logging.info('[inputs] trsl_tns: %s', trsl_tns.shape)

    # test w/ the <InvPntAttn> module
    module = InvPntAttn(**config)
    sfea_tns_out = module(sfea_tns, pfea_tns, quat_tns, trsl_tns)
    logging.info('[outputs] sfea_tns_out: %s', sfea_tns_out.shape)


if __name__ == '__main__':
    main()
