"""Unit-tests for the <AF2SMod> module."""

import random
import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.tools import ProtStruct
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.modules.af2_smod.af2_smod import AF2SMod


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 64
    config = {
        'n_lyrs': 8,
        'n_dims_sfea': 384,
        'n_dims_pfea': 256,
        'n_dims_encd': 32,
        'aa_dep_tors': True,
        'tmsc_pred': True,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    aa_seq = ''.join([random.choice(RESD_NAMES_1C) for _ in range(n_resds)])
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), dtype=torch.float32)
    encd_tns = torch.randn((n_smpls, n_resds, config['n_dims_encd']), dtype=torch.float32)
    cord_tns = torch.randn((n_resds, N_ATOMS_PER_RESD, 3), dtype=torch.float32)
    cmsk_mat = ProtStruct.get_cmsk_vld(aa_seq, cord_tns.device)

    # test w/ the <AF2SMod> module
    module = AF2SMod(**config)
    params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict = \
        module(aa_seq, sfea_tns, pfea_tns, encd_tns, cord_tns=cord_tns, cmsk_mat=cmsk_mat)
    inspect_data(params_list, name='params_list')
    inspect_data(plddt_list, name='plddt_list')
    inspect_data(cord_list, name='cord_list')
    logging.info('fram_tns_sc: %s / %s', fram_tns_sc.shape, fram_tns_sc.dtype)
    inspect_data(tmsc_dict, name='tmsc_dict')


if __name__ == '__main__':
    main()
