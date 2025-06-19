"""Utility functions."""

import logging
from timeit import default_timer as timer

import dgl
import numpy as np
import torch


def build_data_dict(n_nodes, n_edges_per_node, n_dims_node, n_dims_edge, n_grps_cord, device):  # pylint: disable=too-many-arguments
    """Build a dict of DGL graph, node & edge features, and 3D coordinates."""

    # construct randomized edges
    n_edges = n_edges_per_node * n_nodes
    idxs_node_src = np.random.randint(n_nodes, size=(n_edges))
    idxs_node_dst = np.repeat(np.arange(n_nodes), n_edges_per_node, axis=0)

    # create a DGL graph
    graph = dgl.graph((idxs_node_src, idxs_node_dst), num_nodes=n_nodes)

    # construct node/edge features and 3D coordinates
    node_feats = torch.randn(n_nodes, n_dims_node)
    edge_feats = torch.randn(n_edges, n_dims_edge)
    node_cords = torch.randn(n_nodes, n_grps_cord, 3)

    # pack all the elements into a dict
    data_dict = {
        'graph': graph.to(device),
        'nfeat': node_feats.to(device),
        'efeat': edge_feats.to(device),
        'ncord': node_cords.to(device),
    }

    return data_dict


def test_module(module, data_dict, name):
    """Test the specified module."""

    # configurations
    n_repts_wrmp = 16
    n_repts_chck = 16

    # warm-up the module, and then test its execution time
    for _ in range(n_repts_wrmp):
        module(data_dict['graph'], data_dict['nfeat'], data_dict['ncord'], data_dict['efeat'])
    time_beg = timer()
    for _ in range(n_repts_chck):
        module(data_dict['graph'], data_dict['nfeat'], data_dict['ncord'], data_dict['efeat'])
    time_avg = 1000.0 * (timer() - time_beg) / n_repts_chck  # convert to milli-seconds

    # final run
    node_feats, node_cords = module(
        data_dict['graph'], data_dict['nfeat'], data_dict['ncord'], data_dict['efeat'])

    # display results
    logging.info('%s/time: %.2f (ms)', name, time_avg)
    logging.info('%s/node_feats: %s / %s', name, node_feats.shape, node_feats.dtype)
    logging.info('%s/node_cords: %s / %s', name, node_cords.shape, node_cords.dtype)
