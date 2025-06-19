"""Utility functions."""

import dgl
import numpy as np
import torch


def build_inputs(n_nodes, n_edges_per_node, n_dims_nfea, n_dims_efea, n_grps_cord, device):  # pylint: disable=too-many-arguments
    """Build inputs (DGL graph & node/edge features & 3D coordinates & validness masks)."""

    # construct randomized edges
    n_edges = n_edges_per_node * n_nodes
    idxs_node_src = np.random.randint(n_nodes, size=(n_edges))
    idxs_node_dst = np.repeat(np.arange(n_nodes), n_edges_per_node, axis=0)

    # create a DGL graph
    graph = dgl.graph((idxs_node_src, idxs_node_dst), num_nodes=n_nodes, device=device)

    # construct node/edge features & 3D coordinates & validness masks
    node_feats = torch.randn((n_nodes, n_dims_nfea), dtype=torch.float32, device=device)
    node_cords = torch.randn((n_nodes, n_grps_cord, 3), dtype=torch.float32, device=device)
    edge_feats = torch.randn((n_edges, n_dims_efea), dtype=torch.float32, device=device)
    node_masks = torch.randint(2, (n_nodes,), dtype=torch.int8, device=device)

    return graph, node_feats, node_cords, edge_feats, node_masks
