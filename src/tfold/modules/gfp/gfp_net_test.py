"""Unit-tests for the <GFPNet> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.gfp import GFPNet
from tfold.modules.gfp.utils import build_inputs


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_nodes = 1024
    n_edges_per_node = 9
    n_blks = 4
    n_dims_nfea = 64
    n_dims_efea = 32
    n_grps_cord = 4
    n_dims_nemb = 128
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build inputs
    graph, node_feats, node_cords, edge_feats, node_masks = build_inputs(
        n_nodes, n_edges_per_node, n_dims_nfea, n_dims_efea, n_grps_cord, device)

    # test w/ the <GFPNet> module
    module = GFPNet(n_blks, n_dims_nfea, n_dims_efea, n_grps_cord, n_dims_nemb).to(device)
    node_feats_out, node_cords_out = module.forward(
        graph, node_feats, node_cords, edge_feats, node_masks=node_masks)
    logging.info('node_feats_out: %s / %s', node_feats_out.shape, node_feats_out.dtype)
    logging.info('node_cords_out: %s / %s', node_cords_out.shape, node_cords_out.dtype)


if __name__ == '__main__':
    main()
