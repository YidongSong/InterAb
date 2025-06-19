"""Unit-tests for base encoders."""

import logging

import torch
import numpy as np

from tfold.utils import tfold_init
from tfold.utils import quat2rot
from tfold.utils import apply_trans
from tfold.utils import calc_rot_n_tsl_batch
from tfold.tools.base_encoders import PosiEncoder
from tfold.tools.base_encoders import DistEncoder
from tfold.tools.base_encoders import AnglEncoder
from tfold.tools.base_encoders import FramEncoder
from tfold.tools.base_encoders import FrcdEncoder


def gen_inputs(n_resds, n_grps_cord, device):
    """Generate inputs (3D coordinates, rotation matrices, and translation vectors)."""

    # randomly initialize the global rotation & translation
    quat_vec = torch.randn((3), dtype=torch.float32, device=device)  # partial quaternion
    rota_mat = quat2rot(quat_vec.unsqueeze(dim=0))[0]
    trsl_vec = torch.randn((3), dtype=torch.float32, device=device)

    # randomly generate 3D coordinates and then apply the global transformation
    inputs_src = {
        'cord-p': torch.randn((n_resds, n_grps_cord, 3), dtype=torch.float32, device=device),
        'cord-s': torch.randn((n_resds, n_grps_cord, 3), dtype=torch.float32, device=device),
    }  # pre-transformation
    inputs_dst = {
        'cord-p': apply_trans(
            inputs_src['cord-p'], rota_mat, trsl_vec).view(n_resds, n_grps_cord, 3),
        'cord-s': apply_trans(
            inputs_src['cord-s'], rota_mat, trsl_vec).view(n_resds, n_grps_cord, 3),
    }  # post-transformation

    # build rotation matrices and translation vectors
    for inputs in [inputs_src, inputs_dst]:
        rota_tns, trsl_mat = calc_rot_n_tsl_batch(inputs['cord-p'][:, :3])
        inputs.update({'rota-p': rota_tns, 'trsl-p': trsl_mat})
        rota_tns, trsl_mat = calc_rot_n_tsl_batch(inputs['cord-s'][:, :3])
        inputs.update({'rota-s': rota_tns, 'trsl-s': trsl_mat})

    return inputs_src, inputs_dst


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_resds = 64
    n_grps_cord = 4
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # generate inputs (3D coordinates, rotation matrices, and translation vectors)
    inputs_src, inputs_dst = gen_inputs(n_resds, n_grps_cord, device)

    # test w/ <PosiEncoder>
    logging.info('=== PosiEncoder ===')
    posi_encoder = PosiEncoder()
    idxs_vec = torch.arange(n_resds, dtype=torch.int64, device=device)
    encd_mat = posi_encoder.run(idxs_vec)
    logging.info('n_dims: %d', posi_encoder.n_dims)
    logging.info('encd_mat: %s / %s', encd_mat.shape, encd_mat.dtype)

    # test w/ <DistEncoder>
    logging.info('=== DistEncoder ===')
    dist_encoder = DistEncoder()
    dist_vec = 30.0 * torch.rand(n_resds, dtype=torch.float32, device=device)
    encd_mat = dist_encoder.run(dist_vec)
    logging.info('n_dims: %d', dist_encoder.n_dims)
    logging.info('encd_mat: %s / %s', encd_mat.shape, encd_mat.dtype)

    # test w/ <AnglEncoder>
    logging.info('=== AnglEncoder ===')
    angl_encoder = AnglEncoder()
    angl_vec = np.pi * (2.0 * torch.rand(n_resds, dtype=torch.float32, device=device) - 1.0)
    encd_mat = angl_encoder.run(angl_vec)
    logging.info('n_dims: %d', angl_encoder.n_dims)
    logging.info('encd_mat: %s / %s', encd_mat.shape, encd_mat.dtype)

    # test w/ <FramEncoder>
    logging.info('=== FramEncoder ===')
    fram_encoder = FramEncoder(dist_encoder)
    encd_mat_src = fram_encoder.run(
        inputs_src['rota-p'], inputs_src['trsl-p'], inputs_src['rota-s'], inputs_src['trsl-s'])
    encd_mat_dst = fram_encoder.run(
        inputs_dst['rota-p'], inputs_dst['trsl-p'], inputs_dst['rota-s'], inputs_dst['trsl-s'])
    logging.info('n_dims: %d', fram_encoder.n_dims)
    logging.info('encd_mat: %s / %s', encd_mat_src.shape, encd_mat_src.dtype)
    logging.info('SE(3)-invariance: %.4f', torch.norm(encd_mat_dst - encd_mat_src).item())

    # test w/ <FrcdEncoder>
    logging.info('=== FrcdEncoder ===')
    frcd_encoder = FrcdEncoder(dist_encoder, n_grps_cord)
    encd_mat_src = frcd_encoder.run(
        inputs_src['rota-p'], inputs_src['trsl-p'], inputs_src['cord-p'],
        inputs_src['rota-s'], inputs_src['trsl-s'], inputs_src['cord-s'],
    )
    encd_mat_dst = frcd_encoder.run(
        inputs_dst['rota-p'], inputs_dst['trsl-p'], inputs_dst['cord-p'],
        inputs_dst['rota-s'], inputs_dst['trsl-s'], inputs_dst['cord-s'],
    )
    logging.info('n_dims: %d', frcd_encoder.n_dims)
    logging.info('encd_mat: %s / %s', encd_mat_src.shape, encd_mat_src.dtype)
    logging.info('SE(3)-invariance: %.4f', torch.norm(encd_mat_dst - encd_mat_src).item())


if __name__ == '__main__':
    main()
