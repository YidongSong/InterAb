"""Unit-tests for the <EGNN> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.gvp.utils import gen_trans
from tfold.modules.gvp.utils import apply_trans
from tfold.modules.egnn import EGNN
from tfold.modules.egnn.utils import build_data_dict
from tfold.modules.egnn.utils import test_module


def check_se3_eq(module, data_dict):
    """Check the SE(3) equivariance."""

    # initialization
    device = data_dict['graph'].device

    # generate a randomized 3D rotation & translation transformation
    rot_mat, tsl_vec = gen_trans()
    logging.info('rotation matrix: \n%s', rot_mat.detach().numpy())
    logging.info('translation vector: \n%s', tsl_vec.detach().numpy())
    rot_mat = rot_mat.to(device)
    tsl_vec = tsl_vec.to(device)

    # apply transformation on input 3D coordinates
    node_cords_org = data_dict['ncord']
    node_cords_trn = apply_trans(node_cords_org, rot_mat, tsl_vec)

    # perform the forward pass
    node_feats_out_org, node_cords_out_org = module(
        data_dict['graph'], data_dict['nfeat'], node_cords_org, data_dict['efeat'])
    node_feats_out_trn, node_cords_out_trn = module(
        data_dict['graph'], data_dict['nfeat'], node_cords_trn, data_dict['efeat'])

    # compare output scalar & vector features
    node_cords_out_map = apply_trans(node_cords_out_org, rot_mat, tsl_vec)
    logging.info('feat: %.4f', torch.norm(node_feats_out_org - node_feats_out_trn).item())
    logging.info('cord (org-trn): %.4f', torch.norm(node_cords_out_org - node_cords_out_trn).item())
    logging.info('cord (map-trn): %.4f', torch.norm(node_cords_out_map - node_cords_out_trn).item())


def main():
    """Main entry."""

    # configurations
    n_nodes = 4096
    n_edges_per_node = 9
    n_lyrs = 4
    n_dims_nfea = 40
    n_dims_nhid = 32
    n_dims_embd = 16
    n_dims_efea = 100
    n_dims_emsg = 24
    n_grps_cord = 1
    n_heads = 2
    n_dims_attn = 20
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build a dict of DGL graph, node & edge features, and 3D coordinates
    data_dict = build_data_dict(
        n_nodes, n_edges_per_node, n_dims_nfea, n_dims_efea, n_grps_cord, device)

    # test with the <EGNN> module built with <EGCL> layers
    logging.info('=== EGNN built with EGCL layers ===')
    module = EGNN(
        n_lyrs, n_dims_nfea, n_dims_embd, n_dims_efea,
        lyr_type='EGCL', n_dims_emsg=n_dims_emsg, n_dims_nhid=n_dims_nhid,
    ).to(device)
    test_module(module, data_dict, name='EGNN-EGCL')
    check_se3_eq(module, data_dict)

    # test with the <EGNN> module built with <MhaEGCL> layers
    logging.info('=== EGNN built with MhaEGCL layers ===')
    module = EGNN(
        n_lyrs, n_dims_nfea, n_dims_embd, n_dims_efea,
        lyr_type='MhaEGCL', n_dims_emsg=n_dims_emsg, n_dims_nhid=n_dims_nhid,
        n_grps_cord=n_grps_cord, n_heads=n_heads, n_dims_attn=n_dims_attn,
    ).to(device)
    test_module(module, data_dict, name='EGNN-MhaEGCL')
    check_se3_eq(module, data_dict)


if __name__ == '__main__':
    main()
