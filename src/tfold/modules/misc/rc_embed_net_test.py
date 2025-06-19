"""Unit-tests for <RcEmbedNet>."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.modules.misc import RcEmbedNet


def main():
    """Main entry."""

    # configurations
    n_smpls = 4
    msa_depth = 16
    n_resds = 64
    config = {
        'n_dims_mfea': 192,
        'n_dims_pfea': 128,
    }
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # build pair features
    mfea_tns = torch.randn((n_smpls, msa_depth, n_resds, config['n_dims_mfea']), device=device)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), device=device)
    sfea_tns_rc = torch.randn((n_smpls, n_resds, config['n_dims_mfea']), device=device)
    pfea_tns_rc = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), device=device)
    cord_tns_rc = torch.zeros((n_smpls, n_resds, N_ATOMS_PER_RESD, 3), device=device)
    rc_inputs = {'sfea': sfea_tns_rc, 'pfea': pfea_tns_rc, 'cord': cord_tns_rc}
    logging.info('[inputs] mfea_tns: %s', mfea_tns.shape)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)
    inspect_data(rc_inputs, name='[inputs] rc_inputs')

    # test w/ <RcEmbedNet>
    module = RcEmbedNet(**config).to(device)
    mfea_tns, pfea_tns = module(mfea_tns, pfea_tns, rc_inputs=rc_inputs)
    logging.info('[outputs] mfea_tns: %s', mfea_tns.shape)
    logging.info('[outputs] pfea_tns: %s', pfea_tns.shape)


if __name__ == '__main__':
    main()
