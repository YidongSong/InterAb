"""Run benchmarks to measure the GPU memory consumption."""

import logging

import torch
from torch import nn
from ml_collections import ConfigDict

from tfold.utils import tfold_init
from tfold.utils import send_to_device
from tfold.utils import get_peak_memory
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.modules.evoformer import EvoformerStackSS


def build_inputs(config):
    """Build input tensors."""

    # build single & pair features
    sfea_tns = torch.randn((1, config.n_resds, config.n_dims_sfea))
    pfea_tns = torch.randn((1, config.n_resds, config.n_resds, config.n_dims_pfea))
    sfea_tns.requires_grad_()  # otherwise, checkpointing will fail
    pfea_tns.requires_grad_()

    # build per-atom 3D coordinates & asymmetric IDs
    cord_tns = torch.randn((config.n_resds, N_ATOMS_PER_RESD, 3))
    n_resds_per_chn = [int(config.n_resds * x + 0.5) for x in config.chn_size]
    asym_id_raw = []
    for idx_chn, n_resds in enumerate(n_resds_per_chn):
        asym_id_raw.append((idx_chn + 1) * torch.ones(n_resds))
    asym_id = torch.cat(asym_id_raw)[:config.n_resds]  # in case of rounding error

    # pack all into a dict
    inputs = {
        'sfea': sfea_tns,
        'pfea': pfea_tns,
        'cord': cord_tns,
        'asym': asym_id,
    }

    return inputs


def calc_loss(sfea_tns, pfea_tns):
    """Calculate the loss function."""

    loss_sfea = nn.L1Loss()(sfea_tns, torch.zeros_like(sfea_tns))
    loss_pfea = nn.L1Loss()(pfea_tns, torch.zeros_like(pfea_tns))
    loss = loss_sfea + loss_pfea

    return loss


def test_evoformer_stack_ss(config):
    """Test <EvoformerStackSS> w/ specified configurations."""

    # initialization
    device = torch.device('cuda:0')

    # build input tensors
    inputs = build_inputs(config)
    inputs = send_to_device(inputs, device)

    # build a <EvoformerStackSS> module
    module = EvoformerStackSS(
        num_layers=config.n_lyrs,
        c_s=config.n_dims_sfea,
        c_z=config.n_dims_pfea,
    ).to(device)
    module.enable_activation_checkpoint(True)
    module.train()

    # test w/ <EvoformerStackSS>
    for _ in range(config.n_iters):
        sfea_tns, pfea_tns = module(inputs['sfea'], inputs['pfea'])
        loss = calc_loss(sfea_tns, pfea_tns)
        loss.backward()
        logging.info('peak GPU memory: %.2f (MB)', get_peak_memory())


def main():
    """Main entry."""

    # configurations
    n_resds_base = 256
    config = ConfigDict({
        'n_iters': 4,
        'n_resds': None,
        'chn_size': [0.3, 0.5, 0.2],
        'n_lyrs': 16,
        'n_dims_sfea': 192,  # 384,
        'n_dims_pfea': 128,  # 256,
    })
    multipliers = [1, 2, 3, 4]

    # initialization
    tfold_init()

    # enumerate over all the width multiplers
    for multiplier in multipliers:
        config.n_resds = n_resds_base * multiplier
        logging.info('sequence length: %d', config.n_resds)
        logging.info('=== EvoformerStackSS ===')
        test_evoformer_stack_ss(config)


if __name__ == '__main__':
    main()
