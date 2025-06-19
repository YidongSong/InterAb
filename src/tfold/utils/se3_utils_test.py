"""Unit-tests for SE(3)-related utility functions."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import quat2rot
from tfold.utils import rot2quat
from tfold.utils import rtax2rot
from tfold.utils import rot2rtax
from tfold.utils import apply_trans
from tfold.utils import kabsch


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_frams = 64
    n_atoms = 256
    quat_types = ['full', 'part']

    # initialization
    tfold_init(verb_levl='DEBUG')

    # test the Kabsch algorithm
    cord_mat_pri = torch.randn((n_atoms, 3), dtype=torch.float32)
    cord_mat_pri -= torch.mean(cord_mat_pri, dim=0, keepdim=True)
    quat_vec_true = torch.randn((4), dtype=torch.float32)
    rota_mat_true = quat2rot(quat_vec_true.unsqueeze(dim=0))[0]
    cord_mat_sec = torch.sum(rota_mat_true.unsqueeze(dim=0) * cord_mat_pri.unsqueeze(dim=1), dim=2)
    cord_mat_sec += 0.1 * torch.randn_like(cord_mat_sec)
    cord_mat_sec -= torch.mean(cord_mat_sec, dim=0, keepdim=True)
    rota_mat_pred = kabsch(cord_mat_pri, cord_mat_sec)
    logging.info('ground-truth rotation matrix: \n%s', rota_mat_true.detach().cpu().numpy())
    logging.info('estimated rotation matrix: \n%s', rota_mat_pred.detach().cpu().numpy())
    cord_mat_map = torch.sum(rota_mat_true.unsqueeze(dim=0) * cord_mat_pri.unsqueeze(dim=1), dim=2)
    crmsd_true = torch.mean(torch.norm(cord_mat_map - cord_mat_sec, dim=1))
    cord_mat_map = torch.sum(rota_mat_pred.unsqueeze(dim=0) * cord_mat_pri.unsqueeze(dim=1), dim=2)
    crmsd_pred = torch.mean(torch.norm(cord_mat_map - cord_mat_sec, dim=1))
    logging.info('cRMSD: %.4f (true) / %.4f (estimated)', crmsd_true.item(), crmsd_pred.item())

    # test conversion routines for rotation axis vectors
    rtax_vecs_old = torch.randn((n_frams, 3), dtype=torch.float32)
    rot_mats_old = rtax2rot(rtax_vecs_old)
    rtax_vecs_new = rot2rtax(rot_mats_old)
    rot_mats_new = rtax2rot(rtax_vecs_new)
    logging.info('diff. in rotation axis vectors: %.4f', torch.norm(rtax_vecs_new - rtax_vecs_old))
    logging.info('diff. in rotation matrices: %.4f', torch.norm(rot_mats_new - rot_mats_old))

    # test conversion routines for full / partial quaternion vectors
    for quat_type in quat_types:
        # initialization
        n_dims_quat = 4 if quat_type == 'full' else 3
        logging.info('=== testing conversion routines for <%s> quaternion vectors ===', quat_type)

        # test the conversion between quaternion vectors and rotation matrices
        quat_vecs_old = torch.randn((n_frams, n_dims_quat), dtype=torch.float32)
        rot_mats_old = quat2rot(quat_vecs_old)
        quat_vecs_new = rot2quat(rot_mats_old, quat_type)
        rot_mats_new = quat2rot(quat_vecs_new)
        logging.info('diff. in quaternion vectors: %.4f', torch.norm(quat_vecs_new - quat_vecs_old))
        logging.info('diff. in rotation matrices: %.4f', torch.norm(rot_mats_new - rot_mats_old))

        # test the global transformation on 3D coordinates
        cord_tns_old = torch.randn((n_atoms, 3), dtype=torch.float32)
        quat_tns = torch.randn((1, n_dims_quat), dtype=torch.float32)
        trsl_tns = torch.randn((1, 3), dtype=torch.float32)
        rot_tns = quat2rot(quat_tns)
        cord_tns_med = apply_trans(cord_tns_old, rot_tns, trsl_tns).view(n_atoms, 3)
        cord_tns_new = apply_trans(cord_tns_med, rot_tns, trsl_tns, reverse=True).view(n_atoms, 3)
        logging.info('diff. in 3D coordinates: %.4f', torch.norm(cord_tns_new - cord_tns_old))


if __name__ == '__main__':
    main()
