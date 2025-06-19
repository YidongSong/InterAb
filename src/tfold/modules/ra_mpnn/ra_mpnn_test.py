"""Unit-tests for <Mpnn> & <RAMpnn>."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.ra_mpnn.ra_mpnn import Mpnn
from tfold.modules.ra_mpnn.ra_mpnn import RAMpnn
from tfold.modules.ra_mpnn.ra_mpnn import RAMpnnMha
from tfold.modules.ra_mpnn.utils import get_homo_graph_config
from tfold.modules.ra_mpnn.utils import get_homo_graph
from tfold.modules.ra_mpnn.utils import get_hetero_graph_config
from tfold.modules.ra_mpnn.utils import get_hetero_graph


def main():
    """Main entry."""

    # configurations
    n_lyrs = 4
    use_checkpoint = True
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # test w/ <Mpnn>
    config = get_homo_graph_config()
    graph = get_homo_graph(config).to(device)
    logging.info('homogeneous graph (input): %s', graph)
    for version in ['v1', 'v2']:
        logging.info('Mpnn - %s', version)
        module = Mpnn(
            n_lyrs, config.n_dims_nfea,
            n_dims_efea=config.n_dims_efea,
            version=version,
        ).to(device)
        graph_out = module(graph, use_checkpoint=use_checkpoint)
        logging.info('homogeneous graph (output): %s', graph)

    # test w/ <RAMpnn>
    config = get_hetero_graph_config()
    graph = get_hetero_graph(config).to(device)
    logging.info('heterogeneous graph (input): %s', graph)
    for version in ['v1', 'v2']:
        logging.info('RAMpnn - %s', version)
        module = RAMpnn(
            n_lyrs, config.n_dims_resd, config.n_dims_atom,
            n_dims_efea_r2r=config.n_dims_efea_r2r,
            n_dims_efea_r2a=config.n_dims_efea_r2a,
            n_dims_efea_a2a=config.n_dims_efea_a2a,
            n_dims_efea_a2r=config.n_dims_efea_a2r,
            version=version,
        ).to(device)
        graph_out = module(graph, use_checkpoint=use_checkpoint)
        logging.info('heterogeneous graph (output): %s', graph)

    # test w/ <RAMpnnMha>
    config = get_hetero_graph_config()
    graph = get_hetero_graph(config).to(device)
    logging.info('heterogeneous graph (input): %s', graph)
    for version in ['v1', 'v2']:
        logging.info('RAMpnnMha - %s', version)
        module = RAMpnnMha(
            n_lyrs, config.n_dims_resd, config.n_dims_atom,
            n_dims_efea_r2r=config.n_dims_efea_r2r,
            n_dims_efea_r2a=config.n_dims_efea_r2a,
            n_dims_efea_a2a=config.n_dims_efea_a2a,
            n_dims_efea_a2r=config.n_dims_efea_a2r,
            version=version,
        ).to(device)
        graph_out = module(graph, use_checkpoint=use_checkpoint)
        logging.info('heterogeneous graph (output): %s', graph)


if __name__ == '__main__':
    main()
