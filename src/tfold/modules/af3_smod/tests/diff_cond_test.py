"""Unit-tests for the <SingleConditioning> / <PairwiseConditioning> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.af3_smod.diff_cond import SingleConditioning
from tfold.modules.af3_smod.diff_cond import PairwiseConditioning


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 64
    device = torch.device('cuda')

    # config
    config = {
        'single': {
            'sigma_data': 16,
            'n_dims_sfea': 128,
            'n_dims_fourier': 64,
            'transition_expansion_factor': 2,
        },
        'pairwise': {
            'n_dims_pfea_trunk': 128,
            'n_dims_penc': 64,
            'n_dims_pfea': 32,
            'transition_expansion_factor': 2,
        }
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    times = torch.randn(1, ).to(device)
    inpt = torch.randn(
        (n_smpls, n_resds, config['single']['n_dims_sfea']), dtype=torch.float32, device=device)
    sfea_tns_trunk = torch.randn(
        (n_smpls, n_resds, config['single']['n_dims_sfea']), dtype=torch.float32, device=device)
    pfea_tns_trunk = torch.randn(
        (n_smpls, n_resds, n_resds, config['pairwise']['n_dims_pfea_trunk']), dtype=torch.float32, device=device)
    penc_tns = torch.randn(
        (n_smpls, n_resds, n_resds, config['pairwise']['n_dims_penc']), dtype=torch.float32, device=device)

    logging.info('[inputs] inpt: %s', inpt.shape)
    logging.info('[inputs] sfea_tns_trunk: %s', sfea_tns_trunk.shape)
    logging.info('[inputs] pfea_tns_trunk: %s', pfea_tns_trunk.shape)
    logging.info('[inputs] penc_tns: %s', penc_tns.shape)

    # test the SingleConditioning module
    single_conditioner = SingleConditioning(**config['single']).to(device)
    sfea_tns = single_conditioner(times, inpt, sfea_tns_trunk)
    logging.info('[outputs] sfea_tns: %s', sfea_tns.shape)

    # test the PairwiseConditioning module
    pairwise_conditioner = PairwiseConditioning(**config['pairwise']).to(device)
    pfea_tns = pairwise_conditioner(pfea_tns_trunk, penc_tns)
    logging.info('[outputs] pfea_tns: %s', pfea_tns.shape)


if __name__ == '__main__':
    main()
