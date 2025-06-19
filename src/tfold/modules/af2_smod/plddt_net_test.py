"""Unit-tests for the <PLddtNet> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.modules.af2_smod.plddt_net import PLddtNet


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 64
    config = {
        'n_dims_sfea': 384,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32)
    logging.info('[inputs] sfea_tns: %s', sfea_tns.shape)

    # test w/ the <PLddtNet> module
    module = PLddtNet(**config)
    plddt_dict = module(sfea_tns)
    inspect_data(plddt_dict, name='plddt_dict')


if __name__ == '__main__':
    main()
