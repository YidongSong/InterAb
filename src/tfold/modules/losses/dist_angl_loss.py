"""Calculate the inter-residue distance & angle loss."""

import torch
from torch import nn


def calc_wc_tns_bc(labl_tns, mask_tns, n_bins_posi):
    """Calculate weighting coefficients for the <binary> scheme.

    Notes:
    * This is assumed that the first <n_bins_posi> bins correspond to contacting residue pairs.
    """

    # configurations
    eps = 1e-6
    wc_max = 10.0  # to avoid extremely unbalance labels

    # determine per-class weighting coefficients
    mask_tns_posi = mask_tns * torch.lt(labl_tns, n_bins_posi).to(torch.int8)
    mask_tns_nega = mask_tns - mask_tns_posi
    wc_posi = torch.clip(torch.sum(mask_tns) / (2.0 * torch.sum(mask_tns_posi) + eps), max=wc_max)
    wc_nega = torch.clip(torch.sum(mask_tns) / (2.0 * torch.sum(mask_tns_nega) + eps), max=wc_max)

    # determine weighting coefficients for all the residue pairs
    wc_tns = wc_posi * mask_tns_posi + wc_nega * mask_tns_nega

    return wc_tns


def calc_wc_tns_mc(labl_tns, mask_tns, n_bins):
    """Calculate weighting coefficients for the <multi> scheme."""

    # configurations
    eps = 1e-6
    wc_max = 10.0  # to avoid extremely unbalance labels

    # determine per-class weighting coefficients
    onht_tns = mask_tns.unsqueeze(dim=-1) * nn.functional.one_hot(labl_tns, n_bins)
    hist_vec = torch.sum(onht_tns.view(-1, n_bins), dim=0)
    wc_vec = torch.clip(torch.sum(mask_tns) / (n_bins * hist_vec + eps), max=wc_max)

    # determine weighting coefficients for all the residue pairs
    wc_tns = torch.sum(wc_vec.view(1, 1, 1, -1) * onht_tns, dim=3)

    return wc_tns


def calc_loss_dtag(labl_dict, pred_dict, wc_scheme='multi', n_bins_list=None):
    """Calculate the inter-residue distance & angle loss.

    Args:
    * labl_dict: dict of ground-truth labels & validness masks, each of size L x L (or N x L x L)
    * pred_dict: dict of inter-residue distance & angle predictions, each of size N x C x L x L
    * wc_scheme: (optional) weighting scheme (choices: 'uniform' / 'binary' / 'multi')
    * n_bins_list: (optional) list of number of bins in inter-residue distance & angle predictions

    Returns:
    * loss: loss function value
    * metrics: dict of evaluation metrics
    """

    # configurations
    eps = 1e-6
    n_bins_posi = 12  # first 12 distance bins correspond to contacting residue pairs
    if n_bins_list is None:
        n_bins_list = [37, 25, 25, 25]

    # add a pseudo batch dimension if needed
    if labl_dict['cb-idx'].ndim == 2:
        n_smpls = pred_dict['cb'].shape[0]
        labl_dict = {k: v.unsqueeze(dim=0).repeat(n_smpls, 1, 1) for k, v in labl_dict.items()}

    # calculate weighting coefficients for all the residue pairs
    if wc_scheme == 'uniform':
        wc_tns_cb = labl_dict['cb-msk']
        wc_tns_om = labl_dict['om-msk']
        wc_tns_th = labl_dict['th-msk']
        wc_tns_ph = labl_dict['ph-msk']
    elif wc_scheme == 'binary':
        wc_tns_cb = calc_wc_tns_bc(labl_dict['cb-idx'], labl_dict['cb-msk'], n_bins_posi)
        wc_tns_om = labl_dict['om-msk'] * wc_tns_cb
        wc_tns_th = labl_dict['th-msk'] * wc_tns_cb
        wc_tns_ph = labl_dict['ph-msk'] * wc_tns_cb
    elif wc_scheme == 'multi':
        wc_tns_cb = calc_wc_tns_mc(labl_dict['cb-idx'], labl_dict['cb-msk'], n_bins=n_bins_list[0])
        wc_tns_om = calc_wc_tns_mc(labl_dict['om-idx'], labl_dict['om-msk'], n_bins=n_bins_list[1])
        wc_tns_th = calc_wc_tns_mc(labl_dict['th-idx'], labl_dict['th-msk'], n_bins=n_bins_list[2])
        wc_tns_ph = calc_wc_tns_mc(labl_dict['ph-idx'], labl_dict['ph-msk'], n_bins=n_bins_list[3])
    else:
        raise ValueError(f'unrecognized weighting scheme: {wc_scheme}')

    # loss function - inter-residue distance predictions
    loss_tns_cb = nn.CrossEntropyLoss(reduction='none')(pred_dict['cb'], labl_dict['cb-idx'])
    loss_cb = torch.sum(wc_tns_cb * loss_tns_cb) / (torch.sum(wc_tns_cb) + eps)

    # loss function - inter-residue orientation predictions
    loss_tns_om = nn.CrossEntropyLoss(reduction='none')(pred_dict['om'], labl_dict['om-idx'])
    loss_tns_th = nn.CrossEntropyLoss(reduction='none')(pred_dict['th'], labl_dict['th-idx'])
    loss_tns_ph = nn.CrossEntropyLoss(reduction='none')(pred_dict['ph'], labl_dict['ph-idx'])
    loss_om = torch.sum(wc_tns_om * loss_tns_om) / (torch.sum(wc_tns_om) + eps)
    loss_th = torch.sum(wc_tns_th * loss_tns_th) / (torch.sum(wc_tns_th) + eps)
    loss_ph = torch.sum(wc_tns_ph * loss_tns_ph) / (torch.sum(wc_tns_ph) + eps)

    # aggregrate all the loss functions & evaluation metrics
    loss = loss_cb + (loss_om + loss_th + loss_ph) / 3.0
    metrics = {
        'Loss-CB': loss_cb.detach(),
        'Loss-OM': loss_om.detach(),
        'Loss-TH': loss_th.detach(),
        'Loss-PH': loss_ph.detach(),
    }

    return loss, metrics
