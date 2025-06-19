"""Utility functions."""

import torch
from ml_collections import ConfigDict
from torch_geometric.data import Data
from torch_geometric.data import HeteroData


def get_edges(n_nodes_pri, n_nodes_sec, n_edges, n_dims_efea=0):
    """Get edge indices (and edge features, if <n_dims_efea> is non-zero)."""

    eidx_mat = torch.cat([
        torch.randint(n_nodes_pri, size=(1, n_edges), dtype=torch.int64),
        torch.randint(n_nodes_sec, size=(1, n_edges), dtype=torch.int64),
    ], dim=0)
    efea_mat = torch.randn((n_edges, n_dims_efea))

    return eidx_mat if n_dims_efea == 0 else (eidx_mat, efea_mat)


def get_homo_graph_config():
    """Get configurations for generating a homogeneous graph for either residues or atoms."""

    config = ConfigDict({
        'n_nodes': 200,
        'n_edges': 4000,
        'n_dims_nfea': 48,
        'n_dims_efea': 12,
    })

    return config


def get_hetero_graph_config():
    """Get configurations for generating a heterogeneous graph of residues & atoms."""

    config = ConfigDict({
        'n_resds': 100,
        'n_atoms': 400,
        'n_edges_r2r': 1000,
        'n_edges_r2a': 4000,
        'n_edges_a2a': 16000,
        'n_edges_a2r': 4000,
        'n_dims_resd': 64,
        'n_dims_atom': 32,
        'n_dims_efea_r2r': 16,
        'n_dims_efea_r2a': 12,
        'n_dims_efea_a2a': 8,
        'n_dims_efea_a2r': 12,
    })

    return config


def get_homo_graph(config):
    """Get a homogeneous graph of either residues or atoms."""

    graph = Data()
    graph.x = torch.randn((config.n_nodes, config.n_dims_nfea))
    graph.pos = torch.randn((config.n_nodes, 3))
    graph.edge_index, graph.edge_attr = \
        get_edges(config.n_nodes, config.n_nodes, config.n_edges, config.n_dims_efea)

    return graph


def get_hetero_graph(config):
    """Get a heterogeneous graph of residues & atoms."""

    graph = HeteroData()
    graph['resd'].x = torch.randn((config.n_resds, config.n_dims_resd))
    graph['resd'].pos = torch.randn((config.n_resds, 3))
    graph['atom'].x = torch.randn((config.n_atoms, config.n_dims_atom))
    graph['atom'].pos = torch.randn((config.n_atoms, 3))
    graph['resd', 'link', 'resd'].edge_index, graph['resd', 'link', 'resd'].edge_attr = \
        get_edges(config.n_resds, config.n_resds, config.n_edges_r2r, config.n_dims_efea_r2r)
    graph['resd', 'link', 'atom'].edge_index, graph['resd', 'link', 'atom'].edge_attr = \
        get_edges(config.n_resds, config.n_atoms, config.n_edges_r2a, config.n_dims_efea_r2a)
    graph['atom', 'link', 'atom'].edge_index, graph['atom', 'link', 'atom'].edge_attr = \
        get_edges(config.n_atoms, config.n_atoms, config.n_edges_a2a, config.n_dims_efea_a2a)
    graph['atom', 'link', 'resd'].edge_index, graph['atom', 'link', 'resd'].edge_attr = \
        get_edges(config.n_atoms, config.n_resds, config.n_edges_a2r, config.n_dims_efea_a2r)

    return graph


def sp2ds_atom(afea_tns_sp, amsk_mat):
    """Convert per-atom features into the dense format.

    Args:
    * afea_tns_sp: per-atom features (in the sparse format) of size L_i x M (x D1 x D2 ...)
    * amsk_mat: per-atom validness masks of size L_i x M

    Returns:
    * afea_tns_ds: per-atom features (in the dense format) of size N (x D1 x D2 ...)
    """

    # initialization
    n_dims_addi = afea_tns_sp[0, 0].numel()

    # convert per-atom features into the dense format
    aidx_vec = torch.nonzero(amsk_mat.ravel())[:, 0]
    afea_tns_ds = torch.index_select(afea_tns_sp.view(-1, n_dims_addi), 0, aidx_vec)
    afea_tns_ds = afea_tns_ds.view([-1, *afea_tns_sp.shape[2:]])

    return afea_tns_ds


def ds2sp_atom(afea_tns_ds, amsk_mat):
    """Convert per-atom features into the sparse format.

    Args:
    * afea_tns_ds: per-atom features (in the dense format) of size N (x D1 x D2 ...)
    * amsk_mat: per-atom validness masks of size L_i x M

    Returns:
    * afea_tns_sp: per-atom features (in the sparse format) of size L_i x M (x D1 x D2 ...)
    """

    # initialization
    dtype = afea_tns_ds.dtype
    device = afea_tns_ds.device

    # convert per-atom features into the sparse format
    n_atoms = amsk_mat.numel()  # including invalid atoms
    aidx_vec = torch.nonzero(amsk_mat.ravel())[:, 0]
    afea_tns_sp = torch.zeros([n_atoms, *afea_tns_ds.shape[1:]], dtype=dtype, device=device)
    afea_tns_sp[aidx_vec] = afea_tns_ds
    afea_tns_sp = afea_tns_sp.view([*amsk_mat.shape, *afea_tns_ds.shape[1:]])

    return afea_tns_sp
