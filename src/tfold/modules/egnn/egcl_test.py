"""Unit-tests for the <EGCL> module."""

import torch

from tfold.utils import tfold_init
from tfold.modules.egnn import EGCL
from tfold.modules.egnn.utils import build_data_dict
from tfold.modules.egnn.utils import test_module


def main():
    """Main entry."""

    # configurations
    n_nodes = 4096
    n_edges_per_node = 9
    n_dims_node = 64
    n_dims_embd = 48
    n_dims_edge = 32
    n_grps_cord = 1
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build a dict of DGL graph, node & edge features, and 3D coordinates
    data_dict = build_data_dict(
        n_nodes, n_edges_per_node, n_dims_node, n_dims_edge, n_grps_cord, device)

    # test the <EGCL> module
    module = EGCL(n_dims_node, n_dims_embd, n_dims_edge).to(device)
    test_module(module, data_dict, name='EGCL')


if __name__ == '__main__':
    main()
