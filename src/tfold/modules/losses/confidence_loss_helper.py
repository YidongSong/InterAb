"""The helper for calculating confidence loss functions."""

import torch

from tfold.modules.losses.confidence_loss import calc_loss_lddt
from tfold.modules.losses.confidence_loss import calc_loss_pae
from tfold.modules.losses.confidence_loss import calc_loss_pde


class ConfidenceLossHelper():
    """The helper class for calculating the loss and evaluation metrics related to <AF3SMod>."""

    def __init__(
        self,
        wc_lddt=1.0,  # weighting coefficient for lddt loss
        wc_pae=1.0,   # weighting coefficient for pae loss
        wc_pde=1.0,   # weighting coefficient for pde loss
        skip_loss=False  # Whether to skip certain loss terms based on task type
    ):

        """Constructor function."""

        # basic configurations
        self.wc_lddt = wc_lddt
        self.wc_pae = wc_pae
        self.wc_pde = wc_pde
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

        cord_tns_natv = inputs['base']['cord'][0]
        cmsk_mat = inputs['base']['cmsk'][0]
        aa_seq = inputs['base']['seq']

        confidence_logts = outputs['conf_logts']
        cord_tns_pred = outputs['3d']['cord']

        # lddt loss
        if self.wc_lddt > 0.0 and not self.skip_loss:
            loss_lddt, metrics_lddt = calc_loss_lddt(
                confidence_logts['plddt'], aa_seq, cord_tns_pred, cord_tns_natv, cmsk_mat)
            loss_list.append(self.wc_lddt * loss_lddt)
            metrics.update(metrics_lddt)

        # pae loss
        if self.wc_pae > 0.0 and confidence_logts['pae'] is not None and not self.skip_loss:
            loss_pae, metrics_pae = calc_loss_pae(
                confidence_logts['pae'], aa_seq, cord_tns_pred, cord_tns_natv, cmsk_mat)
            loss_list.append(self.wc_pae * loss_pae)
            metrics.update(metrics_pae)

        # pde loss
        if self.wc_pde > 0.0 and confidence_logts['pde'] is not None and not self.skip_loss:
            loss_pde, metrics_pde = calc_loss_pde(
                confidence_logts['pde'], aa_seq, cord_tns_pred, cord_tns_natv, cmsk_mat)
            loss_list.append(self.wc_pde * loss_pde)
            metrics.update(metrics_pde)

        # calculate the overall loss
        loss = torch.sum(torch.stack(loss_list))
        metrics['ConfidenceLoss'] = loss.detach()

        return loss, metrics
