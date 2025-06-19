"""Utility functions."""

import torch
import numpy as np


def gen_trans():
    """Generate a randomized 3D rotation & translation transformation.

    Args: n/a

    Returns:
    * rot_mat: 3D rotation matrix of size 3 x 3
    * tsl_vec: translation vector of size 3
    """

    # generate randomized yaw, pitch, and roll angles
    yaw = np.random.uniform(low=-np.pi, high=np.pi)
    ptc = np.random.uniform(low=-np.pi, high=np.pi)
    rll = np.random.uniform(low=-np.pi, high=np.pi)

    # build 3D rotation matrices around Z, Y, and X axes
    rot_mat_z = torch.tensor([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw), np.cos(yaw), 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=torch.float32)
    rot_mat_y = torch.tensor([
        [np.cos(ptc), 0.0, np.sin(ptc)],
        [0.0, 1.0, 0.0],
        [-np.sin(ptc), 0.0, np.cos(ptc)],
    ], dtype=torch.float32)
    rot_mat_x = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(rll), -np.sin(rll)],
        [0.0, np.sin(rll), np.cos(rll)],
    ], dtype=torch.float32)

    # build the final 3D rotation matrix
    rot_mat = torch.matmul(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))

    # generate a randomized translation vector
    tsl_vec = torch.randn((3), dtype=torch.float32)

    return rot_mat, tsl_vec


def apply_trans(cord_tns, rot_mat, tsl_vec):
    """Apply the 3D rotation & translation transformation on 3D coordinates.

    Args:
    * cord_tns: 3D coordinates of size * x 3 ('*' could contain any positive number of dimensions)
    * rot_mat: 3D rotation matrix of size 3 x 3
    * tsl_vec: translation vector of size 3

    Returns:
    * cord_tns_out: transformed 3D coordinates of same size as <cord_tns>
    """

    cord_tns_out = torch.reshape(
        tsl_vec.view(1, 3) + torch.sum(rot_mat.view(1, 3, 3) * cord_tns.reshape(-1, 1, 3), dim=2)
    , cord_tns.shape)

    return cord_tns_out
