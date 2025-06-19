"""Unit-tests for the <FocalLoss> class."""

import logging

import torch
from torch import nn

from tfold.utils import tfold_init
from tfold.modules.losses import FocalLoss


def main():
    """Main entry."""

    # configurations
    n_batches = 4
    n_smpls = 64
    n_classes = 16
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # build multiple mini-batches for unit-tests
    data_list = []
    for _ in range(n_batches):
        pred_mat = torch.randn(
            (n_smpls, n_classes), dtype=torch.float32, device=device, requires_grad=True)
        labl_vec = torch.randint(n_classes, (n_smpls,), dtype=torch.int64, device=device)
        mask_vec = torch.randint(2, (n_smpls,), dtype=torch.int8, device=device)
        data_list.append((pred_mat, labl_vec, mask_vec))

    # perform unit-tests w/ <FocalLoss>
    loss_fn = FocalLoss()
    for pred_mat, labl_vec, mask_vec in data_list:
        loss = loss_fn(pred_mat, labl_vec, mask_vec)
        logging.info('focal loss: %.4f', loss.item())
        loss.backward()
        logging.info('pred_mat.grad: %s', pred_mat.grad.shape)

        loss_vec = nn.CrossEntropyLoss(reduction='none')(pred_mat, labl_vec)
        loss = torch.sum(mask_vec * loss_vec) / (torch.sum(mask_vec) + 1e-6)
        logging.info('cross-entropy loss: %.4f', loss.item())


if __name__ == '__main__':
    main()
