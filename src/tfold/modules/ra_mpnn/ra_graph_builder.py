"""The residue-atom graph builder."""

import logging

import torch
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData

from tfold.tools import ProtStruct
from tfold.modules.ra_mpnn.utils import sp2ds_atom


@torch.no_grad()
def cdist_fast(cord_mat, validate=False):
    """Calculate the pairwise Euclidean distance matrix w/ a faster implementation.

    Args:
    * cord_mat: coordinate matrix of size N x D
    * validate: (optional) whether to validate results w/ torch.cdist()

    Returns:
    * dist_mat: pairwise Euclidean distance matrix of size N x N
    """

    # calculate the pairwise Euclidean distance matrix
    cord_vec_nrm = torch.sum(torch.square(cord_mat), dim=1)  # squared L2-norm
    dist_mat_sqr = cord_vec_nrm.view(-1, 1) + cord_vec_nrm.view(1, -1) \
        - 2 * torch.matmul(cord_mat, cord_mat.transpose(0, 1))
    dist_mat = torch.sqrt(torch.clip(dist_mat_sqr, min=0.0))

    # validate results w/ torch.cdist()
    if validate:
        dist_mat_ref = cdist(cord_mat)  # which calls torch.cdist() for actual computation
        error = torch.norm(dist_mat - dist_mat_ref).item()
        logging.info('dist_mat: %.4e / %d', error, dist_mat.numel())

    return dist_mat


@torch.no_grad()
def find_edges_impl(dist_mat, degree, radius=None):
    """Find kNN edges - core implementation.

    Notes:
    * Edge indices are stored as (source_index, target_index), if the source node is within the kNN
        neighborhood of the target node.
    """

    # initialization
    device = dist_mat.device
    n_nodes = dist_mat.shape[0]
    assert n_nodes > degree, 'number of nodes must be larger than the kNN graph degree'

    # find kNN graphs
    idxs_mat = torch.argsort(dist_mat, dim=1)[:, degree:]  # find non-neighbor indices
    ic_vec = idxs_mat.flatten()
    ir_vec = torch.arange(n_nodes, device=device).view(-1, 1).repeat(1, n_nodes - degree).flatten()
    dist_mat[ir_vec, ic_vec] = 0.0
    if radius is not None:
        dist_mat[dist_mat > radius] = 0.0
    idxs_mat = torch.nonzero(dist_mat).transpose(0, 1).contiguous()  # no self-loops
    idxs_mat = torch.stack([idxs_mat[1], idxs_mat[0]], dim=0)  # swap source & target nodes

    return idxs_mat  # 2 x N_edges


@torch.no_grad()
def find_edges(cord_mat, n_resds_list, config):
    """Find kNN edges for intra-chain & inter-chain neighbors."""

    # initialization
    device = cord_mat.device
    n_resds_all = cord_mat.shape[0]

    # calculate the pairwise distance matrix
    dist_mat = cdist_fast(cord_mat)
    dist_max = torch.max(dist_mat)

    # build intra-chain & inter-chain masks
    idx_resd_beg = 0
    mask_mat_intra = torch.zeros((n_resds_all, n_resds_all), dtype=torch.int8, device=device)
    for n_resds in n_resds_list:
        idx_resd_end = idx_resd_beg + n_resds
        mask_mat_intra[idx_resd_beg:idx_resd_end, idx_resd_beg:idx_resd_end] = 1
        idx_resd_beg = idx_resd_end  # move the start of next chain
    mask_mat_inter = 1 - mask_mat_intra

    # find intra-chain kNN edges
    dist_mat_intra = dist_mat + dist_max * mask_mat_inter  # only intra-chain distance is valid
    idxs_mat_intra = find_edges_impl(dist_mat_intra, config['degree_intra'], config['radius_intra'])

    # find inter-chain kNN edges
    dist_mat_inter = dist_mat + dist_max * mask_mat_intra  # only inter-chain distance is valid
    idxs_mat_inter = find_edges_impl(dist_mat_inter, config['degree_inter'], config['radius_inter'])

    # concatenate intra-chain & inter-chain kNN edges
    idxs_mat = torch.cat([idxs_mat_intra, idxs_mat_inter], dim=1).contiguous()
    logging.debug('# of intra-chain edges: %d', idxs_mat_intra.shape[1])
    logging.debug('# of inter-chain edges: %d', idxs_mat_inter.shape[1])

    return idxs_mat


class RAGraphBuilder():
    """The residue-atom graph builder."""

    def __init__(self):
        """Constructor function."""

        # setup configurations

        # additional configurations
        self.config_resd = {
            'radius_intra': 15.0,
            'radius_inter': None,  # not limited
            'degree_intra': 24,
            'degree_inter': 24,
        }
        self.radius_atom = 5.0
        self.degree_atom = 8


    def run(self, aa_seq, rfea_mat, rpfe_tns, afea_mat, cord_tns, cmsk_mat, asym_id):
        """Build a heterogeneous graph of residues & atoms.

        Args:
        * aa_seq: amino-acid sequence
        * rfea_mat: per-residue features of size N_r x D_r
        * pfea_tns: residue-residue pairwise features of size N_r x N_r x D_rp
        * afea_mat: per-atom features of size N_a x D_a
        * cord_tns: per-atom 3D coordinates of size N_r x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size N_r x M
        * asym_id: asymmetric unit IDs (chain IDs) of size N_r

        Returns:
        * graph: heterogeneous graph of residues & atoms
        """

        # initialization
        device = rfea_mat.device
        chain_ids = torch.unique(asym_id)
        n_resds_list = [torch.sum((asym_id == x).to(torch.int32)).item() for x in chain_ids]

        # build per-residue & per-atom coordinates
        rcrd_mat = ProtStruct.get_atoms(aa_seq, cord_tns, ['CA'])
        acrd_mat = sp2ds_atom(cord_tns, cmsk_mat)

        # determine the number of residues & atoms
        n_resds = rfea_mat.shape[0]
        n_atoms = afea_mat.shape[0]
        logging.debug('# of residues: %d / # of atoms: %d', n_resds, n_atoms)

        # kNN graph between residues (intra-chain & inter-chain)
        eidx_mat_r2r = find_edges(rcrd_mat, n_resds_list, self.config_resd)
        eidx_mat_r2r = torch.cat([
            eidx_mat_r2r,
            torch.arange(n_resds, dtype=torch.int64, device=device).view(1, -1).repeat(2, 1),
        ], dim=1)  # add self-loops

        # extract edge features for the residue-wise kNN graph
        efea_mat_r2r = rpfe_tns[eidx_mat_r2r[0], eidx_mat_r2r[1]]

        # kNN graph between atoms (intra-chain & inter-chain)
        dist_mat_a2a = cdist_fast(acrd_mat)
        eidx_mat_a2a = find_edges_impl(dist_mat_a2a, self.degree_atom, self.radius_atom)
        eidx_mat_a2a = torch.cat([
            eidx_mat_a2a,
            torch.arange(n_atoms, dtype=torch.int64, device=device).view(1, -1).repeat(2, 1),
        ], dim=1)  # add self-loops

        # kNN graph between residues and atoms (intra-chain only)
        rsiz_vec = torch.sum(cmsk_mat, dim=1)
        ridx_vec = torch.repeat_interleave(
            torch.arange(n_resds, dtype=torch.int64, device=device), rsiz_vec, dim=0)
        aidx_vec = torch.arange(n_atoms, dtype=torch.int64, device=device)
        eidx_mat_r2a = torch.stack([ridx_vec, aidx_vec], dim=0)
        eidx_mat_a2r = torch.stack([aidx_vec, ridx_vec], dim=0)

        # validate node degrees in each graph
        logging.debug('R2R: %s', degree(eidx_mat_r2r[1], num_nodes=n_resds))
        logging.debug('A2A: %s', degree(eidx_mat_a2a[1], num_nodes=n_atoms))
        logging.debug('R2A: %s', degree(eidx_mat_r2a[1], num_nodes=n_atoms))
        logging.debug('A2R: %s', degree(eidx_mat_a2r[1], num_nodes=n_resds))

        # build a heterogeneous graph of residue & atoms
        graph = HeteroData()
        graph['resd'].x = rfea_mat
        graph['resd'].pos = rcrd_mat
        graph['atom'].x = afea_mat
        graph['atom'].pos = acrd_mat
        graph['resd', 'link', 'resd'].edge_index = eidx_mat_r2r
        graph['resd', 'link', 'resd'].edge_attr = efea_mat_r2r
        graph['resd', 'link', 'atom'].edge_index = eidx_mat_r2a
        graph['atom', 'link', 'atom'].edge_index = eidx_mat_a2a
        graph['atom', 'link', 'resd'].edge_index = eidx_mat_a2r
        logging.debug('heterogeneous graph: %s', graph)

        # validate the heterogeneous graph
        try:
            graph.validate(raise_on_error=True)
        except ValueError as err:
            logging.error(f'failed to build a heterogeneous graph for {mchn_data["id"]}')
            raise(err)

        return graph
