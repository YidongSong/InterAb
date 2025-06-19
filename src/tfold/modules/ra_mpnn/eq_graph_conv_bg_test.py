"""Unit-tests for <EqGraphConvBG>."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.tools import SE3EquiValidator
from tfold.modules.ra_mpnn.eq_graph_conv_bg import EqGraphConvBG
from tfold.modules.ra_mpnn.utils import get_eidx_mat


def calc_outputs(module, inputs):
    """Calculate output tensors."""

    # extract input tensors
    nfea_mat_pri = inputs['nfea-p'][0]
    nfea_mat_sec = inputs['nfea-s'][0]
    ncrd_mat_pri = inputs['ncrd-p'][0]
    ncrd_mat_sec = inputs['ncrd-s'][0]
    eidx_mat = inputs['eidx'][0]
    efea_mat = inputs['efea'][0]
    nvlc_mat_sec = inputs['nvlc-s'][0]
    nfrc_mat_sec = inputs['nfrc-s'][0]
    ncum_vec_sec = inputs['ncum-s'][0]

    # perform the forward pass
    module.eval()  # for deterministic behaviors
    nfea_mat_sec, ncrd_mat_sec, nvlc_mat_sec, nfrc_mat_sec = module(
        nfea_mat_pri, nfea_mat_sec, ncrd_mat_pri, ncrd_mat_sec, eidx_mat,
        efea_mat=efea_mat, nvlc_mat_sec=nvlc_mat_sec,
        nfrc_mat_sec=nfrc_mat_sec, ncum_vec_sec=ncum_vec_sec,
    )

    # pack output tensors into a dict
    outputs = {
        'nfea-s': (nfea_mat_sec, 'RITI'),  # invariant to rotation / invariant to translation
        'ncrd-s': (ncrd_mat_sec, 'RETE'),  # equivariant to rotation / equivariant to translation
        'nvlc-s': (nvlc_mat_sec, 'RETI'),  # equivariant to rotation / invariant to translation
        'nfrc-s': (nfrc_mat_sec, 'RETI'),  # equivariant to rotation / invariant to translation
    }

    return outputs


def main():
    """Main entry."""

    # configurations
    n_nodes_pri = 128
    n_nodes_sec = 192
    n_edges = 1024
    n_dims_nfea_pri = 32
    n_dims_nfea_sec = 24
    n_dims_efea = 16
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # randomly generate node features, coordinates, and edge indices
    nfea_mat_pri = torch.randn((n_nodes_pri, n_dims_nfea_pri), dtype=torch.float32, device=device)
    nfea_mat_sec = torch.randn((n_nodes_sec, n_dims_nfea_sec), dtype=torch.float32, device=device)
    ncrd_mat_pri = torch.randn((n_nodes_pri, 3), dtype=torch.float32, device=device)
    ncrd_mat_sec = torch.randn((n_nodes_sec, 3), dtype=torch.float32, device=device)
    eidx_mat = get_eidx_mat(n_nodes_pri, n_nodes_sec, n_edges).to(device)
    efea_mat = torch.randn((n_edges, n_dims_efea), dtype=torch.float32, device=device)
    nvlc_mat_sec = torch.randn((n_nodes_sec, 3), dtype=torch.float32, device=device)
    nfrc_mat_sec = torch.randn((n_nodes_sec, 3), dtype=torch.float32, device=device)
    ncum_vec_sec = torch.ones(n_nodes_sec, dtype=torch.int8, device=device)
    logging.info('[inputs] nfea_mat_pri: %s', nfea_mat_pri.shape)
    logging.info('[inputs] nfea_mat_sec: %s', nfea_mat_sec.shape)
    logging.info('[inputs] ncrd_mat_pri: %s', ncrd_mat_pri.shape)
    logging.info('[inputs] ncrd_mat_sec: %s', ncrd_mat_sec.shape)
    logging.info('[inputs] eidx_mat: %s', eidx_mat.shape)
    logging.info('[inputs] efea_mat: %s', efea_mat.shape)
    logging.info('[inputs] nvlc_mat_sec: %s', nvlc_mat_sec.shape)
    logging.info('[inputs] nfrc_mat_sec: %s', nfrc_mat_sec.shape)
    logging.info('[inputs] ncum_vec_sec: %s', ncum_vec_sec.shape)

    # test w/ <EqGraphConvBG>
    module = EqGraphConvBG(n_dims_nfea_pri, n_dims_nfea_sec, n_dims_efea=n_dims_efea).to(device)
    nfea_mat_sec, ncrd_mat_sec, nvlc_mat_sec, nfrc_mat_sec = module(
        nfea_mat_pri, nfea_mat_sec, ncrd_mat_pri, ncrd_mat_sec, eidx_mat,
        efea_mat=efea_mat, nvlc_mat_sec=nvlc_mat_sec,
        nfrc_mat_sec=nfrc_mat_sec, ncum_vec_sec=ncum_vec_sec,
    )
    logging.info('[outputs] nfea_mat_sec: %s', nfea_mat_sec.shape)
    logging.info('[outputs] ncrd_mat_sec: %s', ncrd_mat_sec.shape)
    logging.info('[outputs] nvlc_mat_sec: %s', nvlc_mat_sec.shape)
    logging.info('[outputs] nfrc_mat_sec: %s', nfrc_mat_sec.shape)

    # validate SE(3) equivariance
    validator = SE3EquiValidator(device)
    inputs_orig = {
        'nfea-p': (nfea_mat_pri, 'RITI'),
        'nfea-s': (nfea_mat_sec, 'RITI'),
        'ncrd-p': (ncrd_mat_pri, 'RETE'),
        'ncrd-s': (ncrd_mat_sec, 'RETE'),
        'eidx': (eidx_mat, 'RITI'),
        'efea': (efea_mat, 'RITI'),
        'nvlc-s': (nvlc_mat_sec, 'RETI'),
        'nfrc-s': (nfrc_mat_sec, 'RETI'),
        'ncum-s': (ncum_vec_sec, 'RITI'),
    }
    inputs_tran = validator.trans_inputs(inputs_orig)
    outputs_orig = calc_outputs(module, inputs_orig)
    outputs_tran = calc_outputs(module, inputs_tran)
    validator.validate_outputs(outputs_orig, outputs_tran)


if __name__ == '__main__':
    main()
