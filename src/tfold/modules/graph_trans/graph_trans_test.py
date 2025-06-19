"""Unit-tests for the <GraphTrans> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.graph_trans import GraphTrans


def main():
    """Main entry."""

    # configurations
    n_smpls = 16
    n_nodes = 64
    n_dims_node = 24
    n_dims_edge = 12
    n_heads = 4
    n_dims_attn = 16
    aggr_fn_list = ['concat', 'mean']
    use_nlnr = True

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build node & edge features
    node_feats = torch.randn(n_smpls, n_nodes, n_dims_node, dtype=torch.float32)
    edge_feats = torch.randn(n_smpls, n_nodes, n_nodes, n_dims_edge, dtype=torch.float32)

    # run unit-tests for the <GraphTrans> module
    for aggr_fn in aggr_fn_list:
        module = GraphTrans(n_dims_node, n_dims_edge, n_heads, n_dims_attn, aggr_fn, use_nlnr)
        logging.info('module created: %s', module)
        node_feats_out = module(node_feats, edge_feats)
        logging.info('node_feats_out: %s', node_feats_out.shape)


if __name__ == '__main__':
    main()
