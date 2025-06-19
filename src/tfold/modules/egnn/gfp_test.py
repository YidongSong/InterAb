"""Unit-tests for the <GFP> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.tools import DistEncoder
from tfold.tools import FrcdEncoder
from tfold.modules.egnn import GFP
from tfold.modules.egnn.utils import build_data_dict


def main():
    """Main entry."""

    # configurations
    n_nodes = 4096
    n_edges_per_node = 9
    n_dims_nfea = 64
    n_dims_efea = 32
    n_grps_cord = 4
    encd_type = 'frcd'
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build a frame & coordinate encoder
    dist_encoder = DistEncoder()
    frcd_encoder = FrcdEncoder(dist_encoder, n_grps_cord)

    # test the <GFP> module
    data_dict = build_data_dict(
        n_nodes, n_edges_per_node, n_dims_nfea, n_dims_efea, n_grps_cord, device)
    module = GFP(
        n_dims_nfea, n_dims_efea, n_grps_cord,
        encd_type=encd_type, frcd_encoder=frcd_encoder, updt_cord=True,
    ).to(device)
    nfea_mat, ncrd_tns = module.forward(
        data_dict['graph'], data_dict['nfeat'], data_dict['ncord'], data_dict['efeat'])
    logging.info('nfea_mat: %s / %s', nfea_mat.shape, nfea_mat.dtype)
    logging.info('ncrd_tns: %s / %s', ncrd_tns.shape, ncrd_tns.dtype)


if __name__ == '__main__':
    main()
