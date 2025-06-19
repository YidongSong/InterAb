"""Calculate the Diffusion Loss for all atom proposed by AlphaFold-3."""

import logging

import torch
import torch.nn.functional as F

from einops import rearrange
from einops import einsum


def weight_loss(loss, sigma, sigma_data=0.5):
    """regular loss weight as defined in EDM paper.

    Args:
    * loss loss of size B x N x C
    * sigma of size B

    Returns:
    * loss: weighted loss
    """

    sigma = rearrange(sigma, 'b ->b 1 1')

    # Note: it is different from the EDM paper
    # weight = (sigma ** 2 + sigma_data ** 2) * (sigma + sigma_data) ** -2     # AF3
    weight = (sigma ** 2 + sigma_data ** 2) * (sigma * sigma_data) ** -2       # EDM
    loss = loss * weight

    return loss


def calc_loss_mse(atom_tns_pred, atom_tns_natv, amsk_mat, sigma, sigma_data, align_weights=None):
    """Calculate MSELoss.

    Args:
    * atom_tns_pred: predicted structure's per-atom 3D coordinates of size B x N x 3
    * atom_tns_natv: native structure's per-atom 3D coordinates of size B x N x 3
    * amsk_mat: per-atom 3D coordinates' validness masks of size B x N

    Returns:
    * loss: loss function
    * metrics: dict of evaluation metrics
    """

    if align_weights is None:
        align_weights = atom_tns_natv.new_ones(atom_tns_natv.shape[:2])

    # calculate msa loss
    losses = F.mse_loss(atom_tns_pred, atom_tns_natv, reduction='none') / 3
    losses = losses * align_weights.unsqueeze(-1)

    mse_loss = losses[amsk_mat.bool()].mean()
    metrics = {'Loss-MSE': mse_loss.detach()}

    weight_mse_loss = weight_loss(losses, sigma, sigma_data)
    weight_mse_loss = weight_mse_loss[amsk_mat.bool()].mean()
    metrics['Loss-WeightMSE'] = weight_mse_loss.detach()

    return weight_mse_loss, metrics


def calc_loss_bond(atom_tns_pred, atom_tns_natv, amsk_mat, sigma, sigma_data):
    """Calculate Extra Bond loss.

    Args:
    * atom_tns_pred: aligned predicted structure's per-atom 3D coordinates of size B x N x 3
    * atom_tns_natv: native structure's per-atom 3D coordinates of size B x N x 3
    * amsk_mat: per-atom 3D coordinates' validness masks of size B x N

    Returns:
    * loss: loss function
    * metrics: dict of evaluation metrics
    """

    pair_amsk_mat = amsk_mat.bool().unsqueeze(1) & amsk_mat.bool().unsqueeze(-1)

    dist_mat_pred = torch.cdist(atom_tns_pred, atom_tns_pred, p=2)
    dist_mat_natv = torch.cdist(atom_tns_natv, atom_tns_natv, p=2)

    bond_losses = F.mse_loss(dist_mat_pred, dist_mat_natv, reduction='none')
    bond_loss = bond_losses[pair_amsk_mat].mean()
    metrics = {'Loss-Bond': bond_loss.detach()}

    weight_bond_loss = weight_loss(bond_losses, sigma, sigma_data)
    weight_bond_loss = weight_bond_loss[pair_amsk_mat].mean()
    metrics['Loss-WeightBond'] = weight_bond_loss.detach()

    return bond_loss, metrics


def calc_loss_smooth_lddt(atom_tns_pred, atom_tns_natv, amsk_mat=None):
    """Calculate SmoothLDDTLoss.

    Args:
    * atom_tns_pred: predicted structure's per-atom 3D coordinates of size B x N x 3
    * atom_tns_natv: native structure's per-atom 3D coordinates of size B x N x 3
    * amsk_mat: per-atom 3D coordinates' validness masks of size B x N

    Returns:
    * loss: loss function
    * metrics: dict of evaluation metrics
    """

    # initialization
    dist_cutoff = 15.0

    # Compute distances between all pairs of atom
    dist_mat_pred = torch.cdist(atom_tns_pred, atom_tns_pred, p=2)
    dist_mat_natv = torch.cdist(atom_tns_natv, atom_tns_natv, p=2)

    # Compute distance difference for all pairs of atom
    dist_diff = torch.abs(dist_mat_natv - dist_mat_pred)

    # Compute epsilon values
    eps = (
        F.sigmoid(0.5 - dist_diff) +
        F.sigmoid(1.0 - dist_diff) +
        F.sigmoid(2.0 - dist_diff) +
        F.sigmoid(4.0 - dist_diff)
    ) / 4.0

    inclusion_radius = dist_mat_natv < dist_cutoff

    # Compute mean, avoiding self term
    mask = inclusion_radius & ~torch.eye(atom_tns_pred.shape[1], dtype=torch.bool, device=atom_tns_pred.device)

    # Take into account variable lengthed atoms in batch
    if amsk_mat is not None:
        paired_mask = amsk_mat.bool().unsqueeze(1) & amsk_mat.bool().unsqueeze(-1)
        mask = mask & paired_mask

    # Calculate masked averaging
    lddt_sum = (eps * mask).sum(dim=(-1, -2))
    lddt_count = mask.sum(dim=(-1, -2))
    lddt = lddt_sum / lddt_count.clamp(min=1)

    lddt_loss = 1. - lddt.mean()
    metrics = {'Loss-SmoothLDDT': lddt_loss.detach()}

    return lddt_loss, metrics


@torch.no_grad()
@torch.cuda.amp.autocast(enabled=False)
def weighted_rigid_align(atom_tns_pred, atom_tns_natv, weights, amsk_mat=None):
    """AF3 Algorithm 28

    Args:
    * atom_tns_pred: native structure's per-atom 3D coordinates of size B x N x 3
    * atom_tns_natv: native structure's per-atom 3D coordinates of size B x N x 3
    * amsk_mat: per-atom 3D coordinates' validness masks of size B x N
    * weights: weights for each atom of size B x N

    Returns:
    * atom_tns_pred: aligned atom of atom_tns_natv
    """

    # initialization
    batch_size, num_points, dim = atom_tns_pred.shape

    if amsk_mat is not None:
        # zero out all predicted and true coordinates where not an atom
        atom_tns_pred = atom_tns_pred * amsk_mat.unsqueeze(-1)
        atom_tns_natv = atom_tns_natv * amsk_mat.unsqueeze(-1)
        weights = weights * amsk_mat

    # Take care of weights broadcasting for coordinate dimension
    weights = rearrange(weights, 'b n -> b n 1')

    # Compute weights centroids
    pred_centroid = (atom_tns_pred * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)
    natv_centroid = (atom_tns_natv * weights).sum(dim=1, keepdim=True) / weights.sum(dim=1, keepdim=True)

    # Center the coordinates
    atom_tns_pred = atom_tns_pred - pred_centroid
    atom_tns_natv = atom_tns_natv - natv_centroid

    if num_points < (dim + 1):
        logging.warning(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(weights * atom_tns_natv, atom_tns_pred, 'b n i, b n j -> b i j')

    # Compute the SVD of the covariance matrix
    U, S, V = torch.svd(cov_matrix)
    U_T = U.transpose(-2, -1)

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        logging.warning(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    det = torch.det(einsum(V, U_T, 'b i j, b j k -> b i k'))
    # Ensure proper rotation matrix with determinant 1
    diag = torch.eye(dim, dtype=det.dtype, device=det.device)[None].repeat(batch_size, 1, 1)
    diag[:, -1, -1] = det
    rot_matrix = einsum(V, diag, U_T, "b i j, b j k, b k l -> b i l")

    # Apply the rotation and translation to atom_tns_natv
    atom_tns_natv_aligned = einsum(rot_matrix, atom_tns_natv, 'b i j, b n j -> b n i') + pred_centroid

    return atom_tns_natv_aligned
