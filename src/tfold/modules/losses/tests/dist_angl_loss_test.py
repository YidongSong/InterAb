"""Unit-tests for the inter-residue distance & angle loss."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.modules.losses.dist_angl_loss import calc_loss_dtag
from tfold.modules.losses.utils import disp_loss_n_metrics


def main():
    """Main entry."""

    # configurations
    n_smpls = 4
    n_resds = 64
    da_names = ['cb', 'om', 'th', 'ph']
    wc_schemes = ['uniform', 'binary', 'multi']
    n_bins_list = [37, 25, 25, 25]

    # initialization
    tfold_init()

    # randomly initialize ground-truth labels & predictions
    labl_dict = {}
    pred_dict = {}
    for da_name, n_bins in zip(da_names, n_bins_list):
        labl_dict[f'{da_name}-idx'] = \
            torch.randint(n_bins, size=(n_resds, n_resds), dtype=torch.int64)
        labl_dict[f'{da_name}-msk'] = torch.randint(2, size=(n_resds, n_resds), dtype=torch.int8)
        pred_dict[da_name] = torch.randn((n_smpls, n_bins, n_resds, n_resds), dtype=torch.float32)
    inspect_data(labl_dict, name='[inputs] labl_dict')
    inspect_data(pred_dict, name='[inputs] pred_dict')

    # test w/ <calc_loss_dtag>
    for wc_scheme in wc_schemes:
        logging.info('=== weighting scheme: %s ===', wc_scheme)
        loss, metrics = calc_loss_dtag(labl_dict, pred_dict, wc_scheme, n_bins_list)
        disp_loss_n_metrics(loss, metrics, name='DTAG')


if __name__ == '__main__':
    main()
