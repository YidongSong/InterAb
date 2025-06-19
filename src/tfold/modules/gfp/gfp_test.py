"""Unit-tests for the <GFP> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import quat2rot
from tfold.utils import apply_trans
from tfold.modules.gfp import GFP
from tfold.modules.gfp.utils import build_inputs


def get_trans(device, enbl_rota=True, enbl_trsl=True):
    """Get 3D transformation with rotation and/or translation enabled."""

    # generate a rotation matrix
    if not enbl_rota:
        rota_mat = torch.eye(3, dtype=torch.float32, device=device)
    else:
        quat_vec = torch.randn((3), dtype=torch.float32, device=device)  # partial quaternion
        rota_mat = quat2rot(quat_vec.unsqueeze(dim=0))[0]

    # generate a translation vector
    if not enbl_trsl:
        trsl_vec = torch.zeros((3), dtype=torch.float32, device=device)
    else:
        trsl_vec = torch.randn((3), dtype=torch.float32, device=device)

    return rota_mat, trsl_vec


def test_equiv(inputs, module, rota_mat, trsl_vec):  # pylint: disable=too-many-locals
    """Test equivariance w.r.t. rotation and/or translation."""

    # obtain input tensors
    graph = inputs['graph']
    node_feats = inputs['node_feats']
    node_cords_org = inputs['node_cords']
    edge_feats = inputs['edge_feats']
    node_masks = inputs['node_masks']

    # perform the forward pass w/ original & transformed node coordinates
    node_cords_trn = apply_trans(node_cords_org, rota_mat, trsl_vec).view(node_cords_org.shape)
    node_feats_out_org, node_cords_out_org = module(
        graph, node_feats, node_cords_org, edge_feats, node_masks=node_masks)
    node_feats_out_trn, node_cords_out_trn = module(
        graph, node_feats, node_cords_trn, edge_feats, node_masks=node_masks)
    node_cords_out_map = apply_trans(
        node_cords_out_org, rota_mat, trsl_vec).view(node_cords_out_org.shape)

    # measure the equivariance in node features & coordinates
    nfea_err_abs = torch.norm(node_feats_out_trn - node_feats_out_org).item()
    nfea_err_rlt = nfea_err_abs / torch.norm(node_feats_out_trn).item()
    ncrd_err_abs = torch.norm(node_cords_out_trn - node_cords_out_map).item()
    ncrd_err_rlt = ncrd_err_abs / torch.norm(node_cords_out_trn).item()
    logging.info('node_feats: %.2e (abs) / %.2e (rlt)', nfea_err_abs, nfea_err_rlt)
    logging.info('node_cords: %.2e (abs) / %.2e (rlt)', ncrd_err_abs, ncrd_err_rlt)


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_nodes = 1024
    n_edges_per_node = 9
    n_dims_nfea = 64
    n_dims_efea = 32
    n_grps_cord = 4
    nfcd_types = ['none', 'frcd-o', 'frcd-s', 'frcd-d']
    efcd_types = ['none', 'dist-s', 'dist-m', 'frcd-o', 'frcd-s', 'frcd-d']
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build inputs
    graph, node_feats, node_cords, edge_feats, node_masks = build_inputs(
        n_nodes, n_edges_per_node, n_dims_nfea, n_dims_efea, n_grps_cord, device)

    # test w/ the <GFP> module
    for nfcd_type in nfcd_types:
        for efcd_type in efcd_types:
            logging.info('=== encoding type: %s (node) / %s (edge) ===', nfcd_type, efcd_type)
            module = GFP(
                n_dims_nfea, n_dims_efea, n_grps_cord, nfcd_type=nfcd_type, efcd_type=efcd_type,
            ).to(device)
            node_feats_out, node_cords_out = module.forward(
                graph, node_feats, node_cords, edge_feats, node_masks=node_masks)
            logging.info('node_feats_out: %s / %s', node_feats_out.shape, node_feats_out.dtype)
            logging.info('node_cords_out: %s / %s', node_cords_out.shape, node_cords_out.dtype)

    # test equivariance w.r.t. rotation and/or translation
    inputs = {
        'graph': graph,
        'node_feats': node_feats,
        'node_cords': node_cords,
        'edge_feats': edge_feats,
        'node_masks': node_masks,
    }
    for nfcd_type in nfcd_types:
        for efcd_type in efcd_types:
            # build a module w/ specified node & edge feature encoding types
            logging.info('=== encoding type: %s (node) / %s (edge) ===', nfcd_type, efcd_type)
            module = GFP(
                n_dims_nfea, n_dims_efea, n_grps_cord, nfcd_type=nfcd_type, efcd_type=efcd_type,
            ).to(device)

            # test equivariance w.r.t. rotation and/or translation
            logging.info('=== rotation only ===')
            rota_mat, trsl_vec = get_trans(device, enbl_rota=True, enbl_trsl=False)
            test_equiv(inputs, module, rota_mat, trsl_vec)
            logging.info('=== translation only ===')
            rota_mat, trsl_vec = get_trans(device, enbl_rota=False, enbl_trsl=True)
            test_equiv(inputs, module, rota_mat, trsl_vec)
            logging.info('=== rotation + translation ===')
            rota_mat, trsl_vec = get_trans(device, enbl_rota=True, enbl_trsl=True)
            test_equiv(inputs, module, rota_mat, trsl_vec)


if __name__ == '__main__':
    main()
