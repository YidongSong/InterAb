"""Unit-tests for <EqGraphConv>."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.tools import SE3EquiValidator
from tfold.modules.ra_mpnn.eq_graph_conv import EqGraphConv
from tfold.modules.ra_mpnn.utils import get_eidx_mat


def calc_outputs(module, inputs):
    """Calculate output tensors."""

    # extract input tensors
    nfea_mat = inputs['nfea'][0]
    ncrd_mat = inputs['ncrd'][0]
    eidx_mat = inputs['eidx'][0]
    efea_mat = inputs['efea'][0]
    nvlc_mat = inputs['nvlc'][0]
    nfrc_mat = inputs['nfrc'][0]
    ncum_vec = inputs['ncum'][0]

    # perform the forward pass
    module.eval()  # for deterministic behaviors
    nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat = module(
        nfea_mat, ncrd_mat, eidx_mat,
        efea_mat=efea_mat, nvlc_mat=nvlc_mat, nfrc_mat=nfrc_mat, ncum_vec=ncum_vec,
    )

    # pack output tensors into a dict
    outputs = {
        'nfea': (nfea_mat, 'RITI'),  # invariant to rotation / invariant to translation
        'ncrd': (ncrd_mat, 'RETE'),  # equivariant to rotation / equivariant to translation
        'nvlc': (nvlc_mat, 'RETI'),  # equivariant to rotation / invariant to translation
        'nfrc': (nfrc_mat, 'RETI'),  # equivariant to rotation / invariant to translation
    }

    return outputs


def main():
    """Main entry."""

    # configurations
    n_nodes = 128
    n_edges = 1024
    n_dims_nfea = 32
    n_dims_efea = 16
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # randomly generate node features, coordinates, and edge indices
    nfea_mat = torch.randn((n_nodes, n_dims_nfea), dtype=torch.float32, device=device)
    ncrd_mat = torch.randn((n_nodes, 3), dtype=torch.float32, device=device)
    eidx_mat = get_eidx_mat(n_nodes, n_nodes, n_edges).to(device)
    efea_mat = torch.randn((n_edges, n_dims_efea), dtype=torch.float32, device=device)
    nvlc_mat = torch.randn((n_nodes, 3), dtype=torch.float32, device=device)
    nfrc_mat = torch.randn((n_nodes, 3), dtype=torch.float32, device=device)
    ncum_vec = torch.ones(n_nodes, dtype=torch.int8, device=device)
    logging.info('[inputs] nfea_mat: %s', nfea_mat.shape)
    logging.info('[inputs] ncrd_mat: %s', ncrd_mat.shape)
    logging.info('[inputs] eidx_mat: %s', eidx_mat.shape)
    logging.info('[inputs] efea_mat: %s', efea_mat.shape)
    logging.info('[inputs] nvlc_mat: %s', nvlc_mat.shape)
    logging.info('[inputs] nfrc_mat: %s', nfrc_mat.shape)
    logging.info('[inputs] ncum_vec: %s', ncum_vec.shape)

    # test w/ <EqGraphConv>
    module = EqGraphConv(n_dims_nfea, n_dims_efea=n_dims_efea).to(device)
    nfea_mat, ncrd_mat, nvlc_mat, nfrc_mat = module(
        nfea_mat, ncrd_mat, eidx_mat,
        efea_mat=efea_mat, nvlc_mat=nvlc_mat, nfrc_mat=nfrc_mat, ncum_vec=ncum_vec,
    )
    logging.info('[outputs] nfea_mat: %s', nfea_mat.shape)
    logging.info('[outputs] ncrd_mat: %s', ncrd_mat.shape)
    logging.info('[outputs] nvlc_mat: %s', nvlc_mat.shape)
    logging.info('[outputs] nfrc_mat: %s', nfrc_mat.shape)

    # validate SE(3) equivariance
    validator = SE3EquiValidator(device)
    inputs_orig = {
        'nfea': (nfea_mat, 'RITI'),
        'ncrd': (ncrd_mat, 'RETE'),
        'eidx': (eidx_mat, 'RITI'),
        'efea': (efea_mat, 'RITI'),
        'nvlc': (nfrc_mat, 'RETI'),
        'nfrc': (nfrc_mat, 'RETI'),
        'ncum': (ncum_vec, 'RITI'),
    }
    inputs_tran = validator.trans_inputs(inputs_orig)
    outputs_orig = calc_outputs(module, inputs_orig)
    outputs_tran = calc_outputs(module, inputs_tran)
    validator.validate_outputs(outputs_orig, outputs_tran)


if __name__ == '__main__':
    main()
