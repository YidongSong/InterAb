"""Calculate the Confidence Loss."""

import torch
from torch import nn

from tfold.tools import LddtAssessor
from tfold.tools import ProtStruct
from tfold.tools import ProtConverter

from tfold.utils import apply_trans
from tfold.utils import cdist


def calc_loss_lddt(plddt_logit, aa_seq, cord_tns_pred, cord_tns_natv, cmsk_mat, eps=1e-8):
    """
    Args:
    * plddt_logit: predicted structure's per-residue lddt logit of size L x n_bins
    * aa_seq: amino-acid of sequence
    * cord_tns_pred: predicted structure's per-atom 3D coordinates of size L x M x 3
    * cord_tns_natv: native structure's per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * loss: loss function
    * metrics: dict of evaluation metrics
    """
    metrics = {}

    # additional configurations
    lddt_assessor = LddtAssessor()
    n_bins_lddt = 50

    plddt_vec_true, plmsk_vec, lddt_val_true = lddt_assessor.run(
        cord_tns_natv, cord_tns_pred, cmsk_mat, atom_set='ca'
    )

    labl_vec = torch.clip(
        torch.floor(n_bins_lddt * plddt_vec_true).to(torch.int64),
        min=0, max=(n_bins_lddt - 1),
    )

    loss = nn.CrossEntropyLoss(reduction='none')(plddt_logit, labl_vec)
    loss = torch.sum(plmsk_vec * loss) / (torch.sum(plmsk_vec) + eps)

    metrics['Loss-lDDT'] = loss.detach()
    metrics['lDDT'] = lddt_val_true.detach()

    return loss, metrics


def calc_loss_pae(pae_logits, aa_seq, cord_tns_pred, cord_tns_natv, cmsk_mat, eps=1e-8):
    """
    Args:
    * pae_logits: predicted structure's pae logit of size L x L x n_bins
    * aa_seq: amino-acid of sequence
    * cord_tns_pred: predicted structure's per-atom 3D coordinates of size L x M x 3
    * cord_tns_natv: native structure's per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * loss: loss function
    * metrics: dict of evaluation metrics
    """

    metrics = {}

    # extract 3D coordinates for CA-atom: L x 3 and local frame:  L x 4 x 3
    cord_mat_pred, fram_tns_true, fmsk_vec = prep_frame_from_cord(aa_seq, cord_tns_natv, cmsk_mat)
    cord_mat_natv, fram_tns_pred, _ = prep_frame_from_cord(aa_seq, cord_tns_pred)

    alignment_errors = compute_alignment_error(
        aa_seq, cord_mat_pred, cord_mat_natv, fram_tns_pred, fram_tns_true).detach()
    boundaries = torch.linspace(0, 31, steps=63, device=pae_logits.device)

    true_bins = torch.sum(alignment_errors[..., None] > boundaries, dim=-1)
    labl_vec = torch.nn.functional.one_hot(true_bins, 64)

    errors_vec = -torch.sum(labl_vec * torch.nn.functional.log_softmax(pae_logits, dim=-1), dim=-1)
    paired_mask = fmsk_vec[..., None] * fmsk_vec[..., None, :]

    # scale: help FP16 training along
    scale = 0.5
    loss = torch.sum(errors_vec * paired_mask, dim=-1)
    denom = torch.sum(scale * paired_mask, dim=(-1, -2)) + eps
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    # Average over the loss dimension
    loss = torch.mean(loss)

    metrics['Loss-PAE'] = loss.detach()

    return loss, metrics


def calc_loss_pde(pde_logits, aa_seq, cord_tns_pred, cord_tns_natv, cmsk_mat, eps=1e-8):
    """
    Args:
    * pde_logits: predicted structure's pde logit of size L x L x n_bins
    * aa_seq: amino-acid of sequence
    * cord_tns_pred: predicted structure's per-atom 3D coordinates of size L x M x 3
    * cord_tns_natv: native structure's per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * loss: loss function
    * metrics: dict of evaluation metrics
    """

    metrics = {}

    cord_mat_pred = ProtStruct.get_atoms(aa_seq, cord_tns_pred, ['CA']).view(-1, 3)
    cord_mat_natv = ProtStruct.get_atoms(aa_seq, cord_tns_natv, ['CA']).view(-1, 3)
    cmsk_mat_ca = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])

    dist_mat_pred = cdist(cord_mat_pred, cord_mat_pred)
    dist_mat_natv = cdist(cord_mat_natv, cord_mat_natv)
    dist_diff = torch.abs(dist_mat_natv - dist_mat_pred)

    boundaries = torch.linspace(0, 31, steps=63, device=pde_logits.device)
    true_bins = torch.sum(dist_diff[..., None] > boundaries, dim=-1)
    labl_vec = torch.nn.functional.one_hot(true_bins, 64)

    errors_vec = -torch.sum(labl_vec * torch.nn.functional.log_softmax(pde_logits, dim=-1), dim=-1)
    paired_mask = cmsk_mat_ca.unsqueeze(1) & cmsk_mat_ca.unsqueeze(0)

    # scale: help FP16 training along
    scale = 0.5
    loss = torch.sum(errors_vec * paired_mask, dim=-1)
    denom = torch.sum(scale * paired_mask, dim=(-1, -2)) + eps
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    # Average over the loss dimension
    loss = torch.mean(loss)
    metrics['Loss-PDE'] = loss.detach()

    return loss, metrics


def compute_alignment_error(aa_seq, cord_mat_pred, cord_mat_true, fram_tns_pred, fram_tns_natv, eps=1e-8):
    """
    Algorithm 30
        aa_seq: amino-acid of sequence
        cord_mat_pred: prediicted CA coordinates of size L x 3
        cord_mat_natv: native coordinates of size L x 3
        fram_tns_pred: predicted frames of size L x 4 x 3
        fram_tns_natv: native frames of size L x 4 x 3
    """

    # obtain local frames for the specified frame set
    fram_tns_true = fram_tns_natv.view(-1, 4, 3)
    fram_tns_pred = fram_tns_pred.view(-1, 4, 3)

    # decompose per-residue local frames into rotation matrices & translation vectors
    rot_tns_true, tsl_mat_true = fram_tns_true[:, :3], fram_tns_true[:, 3]
    rot_tns_pred, tsl_mat_pred = fram_tns_pred[:, :3], fram_tns_pred[:, 3]

    # align 3D coordinates under all the per-residue local frames
    n_atoms = cord_mat_true.shape[0]
    n_frams = fram_tns_true.shape[0]
    cord_tns_true_aln = apply_trans(
        cord_mat_true, rot_tns_true, tsl_mat_true, reverse=True).view(n_frams, n_atoms, 3)
    cord_tns_pred_aln = apply_trans(
        cord_mat_pred, rot_tns_pred, tsl_mat_pred, reverse=True).view(n_frams, n_atoms, 3)

    alignment_errors = torch.sqrt(
        torch.sum((cord_tns_true_aln - cord_tns_pred_aln) ** 2, dim=-1) + eps
    )

    return alignment_errors


def prep_frame_from_cord(aa_seq, cord_tns, cmsk_mat=None, eps=1e-8):
    """
    Build backbone local frames from 3D coordinates.
    Args:
    * aa_seq: amino-acid sequence
    * cord_tns: native structure's per-atom 3D coordinates of size L x M x 3
    * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

    Returns:
    * fram_tns: backbone local frames of size L x 4 x 3
    """

    if cmsk_mat is None:
        cmsk_mat = ProtStruct.get_cmsk_vld(aa_seq, device=cord_tns.device)

    cord_tns_mean = torch.sum(cmsk_mat.view(-1, 1) * cord_tns.view(-1, 3), dim=0) / (torch.sum(cmsk_mat) + eps)

    # Center the coordinate
    cord_tns_cen = cord_tns - cord_tns_mean.view(1, 1, 3)

    # initialize the structure
    prot_struct = ProtStruct()
    prot_converter = ProtConverter()

    # calculate the backbone local frames & torsion angles
    prot_struct.init_from_cord(aa_seq, cord_tns_cen, cmsk_mat)
    prot_struct.build_fram_n_angl(prot_converter, build_sc=False)
    prot_struct.build_mask()

    cord_mat = ProtStruct.get_atoms(aa_seq, cord_tns_cen, ['CA']).view(-1, 3)
    fram_tns = prot_struct.fram_tns_bb.view(-1, 4, 3)
    fmsk_vec = prot_struct.fmsk_mat_bb.view(-1)

    return cord_mat, fram_tns, fmsk_vec
