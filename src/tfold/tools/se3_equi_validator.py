"""The SE(3) equivariance validator.

Notes:
* Three types of SE(3) equivariance can be validated:
  > RITI - Rotational invariance & translational invariance (e.g., atomic features)
  > RETE - Rotational equivariance & translational equivariance (e.g., atomic coordinates)
  > RETI - Rotational equivariance & translational invariance (e.g., atomic forces)
"""

import torch

from tfold.utils import quat2rot


class SE3EquiValidator():
    """The SE(3) equivariance validator."""

    def __init__(self, device):
        """Constructor function."""

        # randomly generate a rotation matrix & translation vector
        quat_vec = torch.randn(3, device=device)  # partial quaternion
        self.rota_mat = quat2rot(quat_vec.unsqueeze(dim=0))[0]
        self.trsl_vec = torch.randn(3, device=device)


    def trans_inputs(self, inputs_orig):
        """Apply an SE(3) transformation on inputs."""

        inputs_tran = {}
        for key, (val_orig, equi_type) in inputs_orig.items():
            val_tran = self.__apply_trans(val_orig, equi_type)
            inputs_tran[key] = (val_tran, equi_type)

        return inputs_tran


    def validate_outputs(self, outputs_orig, outputs_tran):
        """Validate the SE(3) equivariance of outputs."""

        for key, (val_orig, equi_type) in outputs_orig.items():
            val_tran = self.__apply_trans(val_orig, equi_type)
            norm_orig = torch.norm(val_orig - outputs_tran[key][0]).item()
            norm_tran = torch.norm(val_tran - outputs_tran[key][0]).item()
            print(f'{key}: {norm_orig:.2e} / {norm_tran:.2e}')


    def __apply_trans(self, val_orig, equi_type):
        """Apply SE(3)-transformations."""

        if equi_type == 'RITI':
            val_tran = val_orig
        elif equi_type == 'RETE':
            assert val_orig.shape[-1] == 3
            val_tran = (
                torch.sum(self.rota_mat.view(1, 3, 3) * val_orig.view(-1, 1, 3), dim=2)
                + self.trsl_vec.view(1, 3)
            ).view(val_orig.shape)
        elif equi_type == 'RETI':
            val_tran = torch.sum(
                self.rota_mat.view(1, 3, 3) * val_orig.view(-1, 1, 3), dim=2).view(val_orig.shape)
        else:
            raise ValueError(f'unrecognized SE(3) equivariance type: {equi_type}')

        return val_tran
