"""Unit-tests for <RAMpnnAF2>."""

import random
import logging

import torch

from tfold.utils import tfold_init
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.modules.ra_mpnn.ra_mpnn_af2 import RAMpnnAF2


def main():
    """Main entry."""

    # configurations
    n_lyrs = 4
    n_resds_list = [64, 80]
    n_resds = sum(n_resds_list)
    n_dims_sfea = 48
    n_dims_pfea = 32
    version = 'v1'
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # randomly initialize AF2-like inputs
    aa_seq = ''.join([random.choice(RESD_NAMES_1C) for _ in range(n_resds)])
    sfea_tns = torch.randn((1, n_resds, n_dims_sfea), device=device)
    pfea_tns = torch.randn((1, n_resds, n_resds, n_dims_pfea), device=device)
    cord_tns = torch.randn((n_resds, N_ATOMS_PER_RESD, 3), device=device)
    asym_id = torch.cat([
        (idx + 1) * torch.ones(n_resds, device=device) for idx, n_resds in enumerate(n_resds_list)
    ], dim=0)

    # test w/ <RAMpnnAF2>
    module = RAMpnnAF2(
        n_lyrs=n_lyrs,
        n_dims_sfea=n_dims_sfea,
        n_dims_pfea=n_dims_pfea,
        version=version,
    ).to(device)
    cord_tns_out = module(aa_seq, sfea_tns, pfea_tns, cord_tns, asym_id=asym_id)
    logging.info('cord_tns_out: %s', cord_tns_out.shape)


if __name__ == '__main__':
    main()
