"""Unit-tests for the <MhaEGCL> module."""

import torch

from tfold.utils import tfold_init
from tfold.modules.egnn import MhaEGCL
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
    n_grps_cord_list = [1, 4]
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # test the <MhaEGCL> module
    for n_grps_cord in n_grps_cord_list:
        data_dict = build_data_dict(
            n_nodes, n_edges_per_node, n_dims_node, n_dims_edge, n_grps_cord, device)
        module = MhaEGCL(n_dims_node, n_dims_embd, n_dims_edge, n_grps_cord=n_grps_cord).to(device)
        test_module(module, data_dict, name='MhaEGCL')


if __name__ == '__main__':
    main()
