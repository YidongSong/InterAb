"""Utility functions."""

import torch

from tfold.utils import quat2rot
from tfold.utils import rot2quat
from tfold.tools.prot_constants import N_ANGLS_PER_RESD


def init_qta_params(n_smpls, n_resds, mode='black-hole', dtype=None, device=None):
    """Initialize quaternion-translation-angle (QTA) parameters.

    Args:
    * n_smpls: number of samples (default: 1)
    * n_resds: number of residues
    * mode: initialization mode (choices: 'black-hole' / 'random')
    * dtype: (optional) data type
    * device: (optional) computational device

    Returns:
    * quat_tns: quaternion vectors of size N x L x 4
    * trsl_tns: translation vectors of size N x L x 3
    * angl_tns: torsion angle matrices of size N x L x K x 2 (K=7)
    """

    # determine the default data type & computational device
    if dtype is None:
        dtype = torch.float32
    if device is None:
        device = torch.device('cpu')

    # initialize quaternion-translation-angle (QTA) parameters
    if mode == 'black-hole':
        quat_tns = torch.cat([
            torch.ones((n_smpls, n_resds, 1), dtype=dtype, device=device),
            torch.zeros((n_smpls, n_resds, 3), dtype=dtype, device=device),
        ], dim=2)
        trsl_tns = torch.zeros((n_smpls, n_resds, 3), dtype=dtype, device=device)
        angl_tns = torch.zeros((n_smpls, n_resds, N_ANGLS_PER_RESD, 2), dtype=dtype, device=device)
    elif mode == 'random':
        quat_tns = torch.randn((n_smpls, n_resds, 4), dtype=dtype, device=device)
        quat_tns *= torch.sign(quat_tns[:, :, :1])  # qr: non-negative
        quat_tns /= torch.norm(quat_tns, dim=2, keepdim=True)  # unit L2-norm
        trsl_tns = torch.randn((n_smpls, n_resds, 3), dtype=dtype, device=device)
        angl_tns = torch.zeros((n_smpls, n_resds, N_ANGLS_PER_RESD, 2), dtype=dtype, device=device)
    else:
        raise ValueError('unrecognized initialization mode for local frames: {mode}')

    return quat_tns, trsl_tns, angl_tns


def update_se3_trans(quat_tns_old, trsl_tns_old, quat_tns_upd, trsl_tns_upd):
    """Update SE(3) transformations.

    Args:
    * quat_tns_old: old quaternion vectors of size N x L x 4
    * trsl_tns_old: old translation vectors of size N x L x 3
    * quat_tns_upd: update terms of quaternion vectors of size N x L x 4
    * trsl_tns_upd: update terms of translation vectors of size N x L x 3

    Returns*
    * quat_tns_new: new quaternion vectors of size N x L x 4
    * trsl_tns_new: new translation vectors of size N x L x 3
    """

    # initialization
    n_smpls, n_resds, _ = quat_tns_old.shape

    # obtain previous rotation matrices and their update terms
    rota_tns_old = quat2rot(quat_tns_old[0]).unsqueeze(dim=0)  # N x L x 3 x 3
    rota_tns_upd = quat2rot(quat_tns_upd[0]).unsqueeze(dim=0)

    # obtain new rotation matrices
    rota_tns_new = torch.bmm(rota_tns_old[0], rota_tns_upd[0]).unsqueeze(dim=0)

    # obtain new quaternion & translation vectors
    quat_tns_new = rot2quat(rota_tns_new[0], quat_type='full').view(n_smpls, n_resds, -1)
    trsl_tns_new = trsl_tns_old + \
        torch.bmm(rota_tns_old[0], trsl_tns_upd.view(-1, 3, 1)).view(n_smpls, n_resds, -1)

    return quat_tns_new, trsl_tns_new
