"""The focal loss.

Reference:
Lin et al., Focal Loss for Dense Object Detection. ICCV 2017.
"""

import torch
from torch import nn


class FocalLoss(nn.Module):
    """The focal loss."""

    def __init__(self, alpha=0.995, gamma=2.0):
        """Constructor function."""

        super().__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.eps = 1e-6
        self.freq_vec = None  # torch.ones((n_classes), dtype=torch.float32) / n_classes
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, pred_mat, labl_vec, mask_vec):
        """Perform the forward pass.

        Args:
        * pred_mat: predicted classification logits of size N x C
        * labl_vec: ground-truth classification labels of size N (ranges from 0 to C - 1)
        * mask_vec: ground-truth classification labels' validness masks of size N

        Returns:
        * loss: focal loss
        """

        # initialization
        n_classes = pred_mat.shape[1]

        # update the frequency vector
        hist_vec = torch.bincount(mask_vec * labl_vec, minlength=n_classes)
        hist_vec[0] -= torch.sum(mask_vec == 0)
        freq_vec = hist_vec / (torch.sum(mask_vec) + self.eps)
        if self.freq_vec is None:
            self.freq_vec = freq_vec
        else:
            self.freq_vec = self.alpha * self.freq_vec + (1.0 - self.alpha) * freq_vec

        # determine per-class weighting coefficients
        wei_vec_pc = (1.0 / n_classes) / (self.freq_vec + self.eps)

        # extract classification probabilities for the correct class
        prob_mat = self.softmax(pred_mat)
        prob_vec = torch.gather(prob_mat, 1, labl_vec.view(-1, 1)).view(-1) + self.eps

        # determine per-sample weighting coefficients
        #wei_vec_ps = mask_vec  # standard cross-entropy loss
        wei_vec_ps = mask_vec * \
            torch.take(wei_vec_pc, labl_vec) * torch.pow(1.0 - prob_vec, self.gamma)
        loss = torch.sum(-wei_vec_ps * torch.log(prob_vec)) / (torch.sum(wei_vec_ps) + self.eps)

        return loss
