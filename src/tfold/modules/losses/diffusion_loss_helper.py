"""The helper for calculating diffusion loss functions."""

import torch

from einops import repeat

from tfold.modules.losses.diffusion_loss import weighted_rigid_align
from tfold.modules.losses.diffusion_loss import calc_loss_mse
from tfold.modules.losses.diffusion_loss import calc_loss_bond
from tfold.modules.losses.diffusion_loss import calc_loss_smooth_lddt


class DiffusionLossHelper():
    """The helper class for calculating the loss and evaluation metrics related to <AF3SMod>."""

    def __init__(
        self,
        sigma_data=16,  # number of sampling steps
        wc_mse=1.0,  # weighting coefficient for mean squared error (MSE)
        wc_bond=0.0,   # weighting coefficient for bond loss
        wc_smooth_lddt=1.0,   # weighting coefficient for smooth lddt loss
        skip_loss=False,  # whether to skip certain loss terms based on the task type
    ):

        """Constructor function."""

        # basic configurations
        self.sigma_data = sigma_data
        self.wc_mse = wc_mse
        self.wc_bond = wc_bond
        self.wc_smooth_lddt = wc_smooth_lddt
        self.skip_loss = skip_loss

    def run(self, inputs, outputs):

        """Run the loss helper to calculate various loss functions.

        Args:
        * inputs: dict of input tensors
        * outputs: dict of output tensors

        Returns:
        * loss: loss function value
        * metrics: dict of evaluation metrics
        """

        # initialization
        loss_list = []
        metrics = {}

        atom_tns_natv = inputs['base']['atom']
        atom_tns_pred = outputs['3d']['atom']
        amsk_mat = inputs['base']['amsk']
        align_weights = atom_tns_natv.new_ones(outputs['3d']['atom'].shape[:2])
        # take care of augmentation
        batch_size = outputs['3d']['atom'].shape[0]
        if batch_size > 1:
            (atom_tns_natv, amsk_mat) = tuple(
                repeat(t, 'b ... -> (b a) ...', a=batch_size)
                for t in (atom_tns_natv, amsk_mat) if t is not None
            )

        # align the rigid
        atom_tns_natv = weighted_rigid_align(
            atom_tns_pred,
            atom_tns_natv,
            align_weights,
            amsk_mat
        )

        # calculate the MSELoss
        if self.wc_mse > 0.0 and not self.skip_loss:
            loss_mse, metrics_mse = calc_loss_mse(
                atom_tns_pred, atom_tns_natv, amsk_mat, outputs['sigmas'], sigma_data=self.sigma_data)
            loss_list.append(self.wc_mse * loss_mse)
            metrics.update(metrics_mse)

        # calculate the Bond Loss
        if self.wc_bond > 0.0 and not self.skip_loss:
            loss_bond, metrics_bond = calc_loss_bond(
                atom_tns_pred, atom_tns_natv, amsk_mat, outputs['sigmas'], sigma_data=self.sigma_data)
            loss_list.append(self.wc_bond * loss_bond)
            metrics.update(metrics_bond)

        # calculate the SmoothLDDTLoss
        if self.wc_smooth_lddt > 0.0 and not self.skip_loss:
            loss_lddt, metrics_lddt = calc_loss_smooth_lddt(atom_tns_pred, atom_tns_natv, amsk_mat)
            loss_list.append(self.wc_smooth_lddt * loss_lddt)
            metrics.update(metrics_lddt)

        # calculate the overall loss
        loss = torch.sum(torch.stack(loss_list))
        metrics['DiffusionLoss'] = loss.detach()

        return loss, metrics
