"""Unit-tests for the <GVP> module."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.modules.gvp import GVP
from tfold.modules.gvp.utils import gen_trans
from tfold.modules.gvp.utils import apply_trans


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_nodes = 64
    n_dims_is = 16
    n_dims_iv = 4
    n_dims_os = 24
    n_dims_ov = 6
    n_dims_hv = 8

    # initialization
    tfold_init(verb_levl='DEBUG')

    # generate a randomized 3D rotation & translation transformation
    rot_mat, tsl_vec = gen_trans()
    logging.info('rotation matrix: \n%s', rot_mat.detach().numpy())
    logging.info('translation vector: \n%s', tsl_vec.detach().numpy())

    # build input scalar & vector features (original & rotated)
    sfea_mat_in = torch.randn((n_nodes, n_dims_is), dtype=torch.float32)
    vfea_tns_in_org = torch.randn((n_nodes, n_dims_iv, 3), dtype=torch.float32)
    vfea_tns_in_trn = apply_trans(vfea_tns_in_org, rot_mat, tsl_vec)
    vmsk_mat = torch.randint(2, (n_nodes, n_dims_iv), dtype=torch.int8)

    # perform the forward pass
    module = GVP(n_dims_is, n_dims_iv, n_dims_os, n_dims_ov, n_dims_hv=n_dims_hv)
    sfea_mat_out_org, vfea_tns_out_org = module(sfea_mat_in, vfea_tns_in_org, vmsk_mat)
    sfea_mat_out_rot, vfea_tns_out_trn = module(sfea_mat_in, vfea_tns_in_trn, vmsk_mat)

    # compare output scalar & vector features
    vfea_tns_out_map = apply_trans(vfea_tns_out_org, rot_mat, tsl_vec)
    logging.info('sfea: %.4f', torch.norm(sfea_mat_out_org - sfea_mat_out_rot).item())
    logging.info('vfea (org-trn): %.4f', torch.norm(vfea_tns_out_org - vfea_tns_out_trn).item())
    logging.info('vfea (map-trn): %.4f', torch.norm(vfea_tns_out_map - vfea_tns_out_trn).item())


if __name__ == '__main__':
    main()
