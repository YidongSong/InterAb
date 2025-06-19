"""Unit-tests for the <ConfidenceHead> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.utils import get_peak_memory
from tfold.modules.af3_smod.conf_head import ConfidenceHead


def main():
    """Main entry."""

    # configurations
    n_smpls = 5
    n_resds = 60
    n_resds_pri = 40
    device = torch.device('cuda')
    n_resds_sec = n_resds - n_resds_pri

    config = {
        'n_dims_inpt': 20,
        'n_dims_sfea': 128,
        'n_dims_pfea': 64,
        'n_lyrs': 4,
        'cal_pae': True,
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    inpt = torch.randn((n_smpls, n_resds, config['n_dims_inpt']), dtype=torch.float32, device=device)
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32, device=device)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), dtype=torch.float32, device=device)
    cord_tns = torch.randn((n_smpls, n_resds, 3), dtype=torch.float32, device=device)

    asym_id = torch.cat([
        0 * torch.ones((n_resds_pri), dtype=torch.float32),
        1 * torch.ones((n_resds_sec), dtype=torch.float32),
    ], dim=0).to(device)
    asym_id = asym_id.unsqueeze(0).expand(n_smpls, -1)

    logging.info('[inputs] inpt: %s', inpt.shape)
    logging.info('[inputs] sfea_tns: %s', sfea_tns.shape)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)
    logging.info('[inputs] cord_tns: %s', cord_tns.shape)
    logging.info('[inputs] asym_id: %s', asym_id.shape)

    # test the AF3 ConfidenceHead module
    module = ConfidenceHead(**config).to(device)
    conf_logts, conf_metric = module(inpt, sfea_tns, pfea_tns, cord_tns, asym_id)
    print('peak GPU memory after ConfidenceHead: %.2f (MB)' % get_peak_memory())
    inspect_data(conf_logts, name='conf_logts')
    inspect_data(conf_metric, name='conf_metric')


if __name__ == '__main__':
    main()
