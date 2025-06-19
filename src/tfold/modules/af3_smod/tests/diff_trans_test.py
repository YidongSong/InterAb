"""Unit-tests for the <DiffusionTransformer> module."""


import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import get_peak_memory
from tfold.modules.af3_smod.diff_trans import DiffusionTransformer


def main():
    """Main entry."""

    # configurations
    n_smpls = 2
    n_resds = 300
    device = torch.device('cuda')

    config = {
        'n_lyrs': 3,
        'n_heads': 4,
        'dim': 128,
        'n_dims_cond': 384,
        'n_dims_pfea': 128,
        'attn_window_size': 27,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    nfea_tns = torch.randn((n_smpls, n_resds, config['dim']), dtype=torch.float32, device=device)
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_cond']), dtype=torch.float32, device=device)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), dtype=torch.float32, device=device)

    logging.info('[inputs] nfea_tns: %s', nfea_tns.shape)
    logging.info('[inputs] sfea_tns: %s', sfea_tns.shape)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)

    # test the AtomAttentionEncoder module
    diff_trans = DiffusionTransformer(**config).to(device)
    diff_trans.enable_activation_checkpoint(False)
    nfea_tns_updt = diff_trans(nfea_tns, sfea_tns, pfea_tns)

    print('peak GPU memory after diff_trans: %.2f (MB)' % get_peak_memory())
    logging.info('[outputs] nfea_tns: %s', nfea_tns_updt.shape)


if __name__ == '__main__':
    main()
