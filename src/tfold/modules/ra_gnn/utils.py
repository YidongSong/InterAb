"""Utility functions."""

import torch

from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


def sp2ds(aa_seq, atom_tns_sp, mask_mat=None):
    """Convert the per-atom tensor from the sparse format into dense.

    Args:
    * aa_seq: amino-acid sequence
    * atom_tns_sp: per-atom tensor in the sparse format of size L x M (x D1 x D2 ...)
    * mask_mat: (optional) per-atom validness masks of size L x M

    Returns:
    * atom_tns_ds: per-atom tensor in the dense format of size Nv (x D1 x D2 ...)
    """

    # initialization
    n_resds = len(aa_seq)
    device = atom_tns_sp.device
    n_atoms_all = n_resds * N_ATOMS_PER_RESD
    n_dims_addi = atom_tns_sp.numel() // n_atoms_all

    # build validness masks if not provided
    if mask_mat is None:
        mask_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8)
        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            mask_mat[idx_resd, :len(atom_names)] = 1
        mask_mat = mask_mat.to(device)

    # convert the per-atom tensor from the sparse format into dense
    idxs_atom = torch.nonzero(mask_mat.flatten(), as_tuple=True)[0]
    n_atoms_vld = idxs_atom.shape[0]
    atom_tns_ds = torch.gather(
        atom_tns_sp.view(n_atoms_all, -1), 0, idxs_atom.view(n_atoms_vld, 1).repeat(1, n_dims_addi),
    ).view(n_atoms_vld, *atom_tns_sp.size()[2:])

    return atom_tns_ds


def ds2sp(aa_seq, atom_tns_ds, mask_mat=None):
    """Convert the per-atom tensor from the dense format into sparse.

    Args:
    * aa_seq: amino-acid sequence
    * atom_tns_ds: per-atom tensor in the dense format of size Nv (x D1 x D2 ...)
    * mask_mat: (optional) per-atom validness masks of size L x M

    Returns:
    * atom_tns_sp: per-atom tensor in the sparse format of size L x M (x D1 x D2 ...)
    """

    # initialization
    n_resds = len(aa_seq)
    device = atom_tns_ds.device
    n_atoms_all = n_resds * N_ATOMS_PER_RESD
    n_atoms_vld = atom_tns_ds.shape[0]
    n_dims_addi = atom_tns_ds.numel() // n_atoms_vld

    # build validness masks if not provided
    if mask_mat is None:
        mask_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8)
        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            mask_mat[idx_resd, :len(atom_names)] = 1
        mask_mat = mask_mat.to(device)

    # convert the per-atom tensor from the dense format into sparse
    idxs_atom = torch.nonzero(mask_mat.flatten(), as_tuple=True)[0]
    atom_tns_sp = torch.scatter(
        torch.zeros((n_atoms_all, n_dims_addi), dtype=torch.float32, device=device),
        0, idxs_atom.view(n_atoms_vld, 1).repeat(1, n_dims_addi), atom_tns_ds.view(n_atoms_vld, -1),
    ).view(n_resds, N_ATOMS_PER_RESD, *atom_tns_ds.size()[1:])

    return atom_tns_sp
