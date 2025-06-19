"""Unit-tests for <CPNetVx> modules."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import quat2rot
from tfold.utils import apply_trans
from tfold.modules.gfp import CPNetV1
from tfold.modules.gfp import CPNetV2
from tfold.modules.gfp import CPNetV3


def get_trans(device, enbl_rota=True, enbl_trsl=True):
    """Get 3D transformation with rotation and/or translation enabled."""

    # generate a rotation matrix
    if not enbl_rota:
        rota_mat = torch.eye(3, dtype=torch.float32, device=device)
    else:
        quat_vec = torch.randn((3), dtype=torch.float32, device=device)  # partial quaternion
        rota_mat = quat2rot(quat_vec.unsqueeze(dim=0))[0]

    # generate a translation vector
    if not enbl_trsl:
        trsl_vec = torch.zeros((3), dtype=torch.float32, device=device)
    else:
        trsl_vec = torch.randn((3), dtype=torch.float32, device=device)

    return rota_mat, trsl_vec


def test_equiv(cord_tns_org, module, rota_mat, trsl_vec):
    """Test equivariance w.r.t. rotation and/or translation."""

    # perform the forward pass w/ original & transformed node coordinates
    cord_tns_trn = apply_trans(cord_tns_org, rota_mat, trsl_vec).view(cord_tns_org.shape)
    cord_tns_out_org = module(cord_tns_org)
    cord_tns_out_trn = module(cord_tns_trn)
    cord_tns_out_map = apply_trans(
        cord_tns_out_org, rota_mat, trsl_vec).view(cord_tns_out_org.shape)

    # measure the equivariance in 3D coordinates
    cord_err_abs = torch.norm(cord_tns_out_trn - cord_tns_out_map).item()
    cord_err_rlt = cord_err_abs / torch.norm(cord_tns_out_trn).item()
    logging.info('cord_tns: %.2e (abs) / %.2e (rlt)', cord_err_abs, cord_err_rlt)


def main():
    """Main entry."""

    # configurations
    n_nodes = 64
    n_grps_cord_src = 4
    n_grps_cord_dst = 8
    device = torch.device('cuda:0')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # build input 3D coordinates
    cord_tns = torch.randn((n_nodes, n_grps_cord_src, 3), dtype=torch.float32, device=device)

    # test w/ <CPNetVx> modules
    for CPNet in [CPNetV1, CPNetV2, CPNetV3]:
        logging.info('=== %s ===', CPNet)
        module = CPNet(n_grps_cord_src, n_grps_cord_dst).to(device)
        logging.info('=== rotation only ===')
        rota_mat, trsl_vec = get_trans(device, enbl_rota=True, enbl_trsl=False)
        test_equiv(cord_tns, module, rota_mat, trsl_vec)
        logging.info('=== translation only ===')
        rota_mat, trsl_vec = get_trans(device, enbl_rota=False, enbl_trsl=True)
        test_equiv(cord_tns, module, rota_mat, trsl_vec)
        logging.info('=== rotation + translation ===')
        rota_mat, trsl_vec = get_trans(device, enbl_rota=True, enbl_trsl=True)
        test_equiv(cord_tns, module, rota_mat, trsl_vec)


if __name__ == '__main__':
    main()
