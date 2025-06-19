"""PyTorch-related utility functions."""

import os
import logging

import torch
import numpy as np


def get_tensor_size(tensor):
    """Get the PyTorch tensor's memory consumption (in MB).

    Args:
    * tensor: PyTorch tensor

    Returns:
    * mem: PyTorch tensor's memory consumption (in MB)
    """

    mem = tensor.element_size() * torch.numel(tensor) / 1024.0 / 1024.0

    return mem


def get_peak_memory():
    """Get the peak memory consumption (in MB).

    Args: n/a

    Returns:
    * mem: peak memory consumption (in MB)
    """

    mem = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024.0 / 1024.0

    return mem


def send_to_device(data, device):
    """Send the data object to the target device.

    Args:
    * data: data object (list / dict / torch.Tensor / etc.)
    * device: target device

    Returns:
    * data: data object stored in the target device
    """

    if isinstance(data, list) or isinstance(data, tuple):
        return [send_to_device(x, device) for x in data]
    if isinstance(data, dict):
        return {k: send_to_device(v, device) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def inspect_data(data, name):
    """Inspect the data object.

    Args:
    * data: data object (list / dict / torch.Tensor / etc.)
    * name: data object name

    Returns: n/a
    """

    if isinstance(data, list):
        for idx, val in enumerate(data):
            inspect_data(val, f'{name}/{idx}')
    elif isinstance(data, dict):
        for key, val in data.items():
            inspect_data(val, f'{name}/{key}')
    elif isinstance(data, np.ndarray):
        logging.info('%s: %s / %s', name, data.shape, data.dtype)
    elif isinstance(data, torch.Tensor):
        logging.info('%s: %s / %s / %s', name, data.shape, data.dtype, data.device)
    else:
        logging.info('%s: %s', name, data)


def clone_data(data):
    """Clone the data object w/ gradients detached.

    Args:
    * data: source data object (list / dict / torch.Tensor / etc.)

    Returns:
    * data: cloned data object (list / dict / torch.Tensor / etc.)
    """

    if isinstance(data, list):
        return [clone_data(x) for x in data]
    if isinstance(data, dict):
        return {k: clone_data(v) for k, v in data.items()}
    if isinstance(data, torch.Tensor):
        return data.detach().clone()

    return data


def report_abnormal_keys(keys_miss, keys_uexp):
    """Report abnormal keys (if any) in restoring model parameters.

    Args:
    * keys_miss: list of missing keys
    * keys_uexp: list of unexpected keys

    Returns: n/a
    """

    if len(keys_miss) != 0:
        logging.warning('# of missing keys: %d', len(keys_miss))
        logging.warning('missing keys: %s', ','.join(keys_miss))
    if len(keys_uexp) != 0:
        logging.warning('# of unexpected keys: %d', len(keys_uexp))
        logging.warning('unexpected keys: %s', ','.join(keys_uexp))


def get_state_dict_wo_plm(model):
    """Get a state dict w/o variables belonging to pre-trained language model (PLM).

    Args:
    * model: nn.Module() object

    Returns:
    * state_dict: state dict w/o PLM variables
    """

    return {k: v for k, v in model.state_dict().items() if not k.startswith('plm_model')}


def save_model(model, path):
    """Save the model to a checkpoint file.

    Args:
    * model: nn.Module() object to be saved
    * path: file path to a model checkpoint

    Returns: n/a
    """

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    torch.save(model.state_dict(), path)
    logging.info('model saved to %s', path)


def load_model(model, path, strict=True):
    """Restore model parameters from the checkpoint file.

    Args:
    * model: nn.Module() object to be loaded
    * path: file path to a model checkpoint
    * strict: (optional) whether missing and/or unexpected keys are allowed

    Returns: n/a
    """

    if os.path.exists(path):
        snapshot = torch.load(path, map_location='cpu')
        keys_miss, keys_uexp = model.load_state_dict(snapshot, strict=strict)
        report_abnormal_keys(keys_miss, keys_uexp)
        logging.info('model loaded from %s', path)
    else:
        logging.warning('model checkpoint file not found: %s', path)


def save_snapshot(
        mdl_dpath, model_base, model_trgt, optimizer, scheduler,
        metrics_list_intra, metrics_list_inter, scaler=None, save_wo_plm=False):
    """Save snapshot files for models and evaluation metrics.

    Args:
    * mdl_dpath: directory path to snapshot files
    * model_base: base model
    * model_trgt: target model (updated via EMA)
    * optimizer: model optimizer
    * scheduler: learning rate scheduler
    * metrics_list_intra: list of intra-epoch evaluation metrics (one dict per iteration)
    * metrics_list_inter: list of inter-epoch evaluation metrics (one dict per epoch)
    * scaler: (optional) gradient scaler for mixed-precision training
    * save_wo_plm: (optional) whether to save models w/o PLM variables

    Returns: n/a
    """

    # initialization
    os.makedirs(mdl_dpath, exist_ok=True)

    # save a snapshot for models
    snapshot = {
        'model_base': get_state_dict_wo_plm(model_base) if save_wo_plm else model_base.state_dict(),
        'model_trgt': get_state_dict_wo_plm(model_trgt) if save_wo_plm else model_trgt.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
    }
    if scaler is not None:
        snapshot['scaler'] = scaler.state_dict()
    pth_fpath = os.path.join(mdl_dpath, 'snapshot_models.pth')
    torch.save(snapshot, pth_fpath)
    logging.info('model snapshot saved to %s', pth_fpath)

    # save a snapshot for evaluation metrics
    snapshot = {
        'metrics_list_intra': metrics_list_intra,
        'metrics_list_inter': metrics_list_inter,
    }
    pth_fpath = os.path.join(mdl_dpath, 'snapshot_metrics.pth')
    torch.save(snapshot, pth_fpath)
    logging.info('metric snapshot saved to %s', pth_fpath)


def load_snapshot(mdl_dpath, model_base, model_trgt, optimizer=None, scheduler=None, scaler=None, strict=True):
    """Load snapshot files for models and evaluation metrics.

    Args:
    * mdl_dpath: directory path to snapshot files
    * model_base: base model
    * model_trgt: target model (updated via EMA)
    * optimizer: (optional) model optimizer
    * scheduler: (optimizer) learning rate scheduler
    * scaler: (optional) gradient scaler for mixed-precision training
    * strict: (optional) whether missing and/or unexpected keys are allowed

    Returns:
    * metrics_list_intra: list of intra-epoch evaluation metrics (one dict per iteration)
    * metrics_list_inter: list of inter-epoch evaluation metrics (one dict per epoch)
    """

    # restore the snapshot for models
    pth_fpath = os.path.join(mdl_dpath, 'snapshot_models.pth')
    if os.path.exists(pth_fpath):
        # restore base & target models
        snapshot = torch.load(pth_fpath, map_location='cpu')
        keys_miss, keys_uexp = model_base.load_state_dict(snapshot['model_base'], strict=strict)
        report_abnormal_keys(keys_miss, keys_uexp)
        keys_miss, keys_uexp = model_trgt.load_state_dict(snapshot['model_trgt'], strict=strict)
        report_abnormal_keys(keys_miss, keys_uexp)

        # restore optimizer & LR scheduler
        if (optimizer is not None) and ('optimizer' in snapshot) and strict:
            optimizer.load_state_dict(snapshot['optimizer'])
        if (scheduler is not None) and ('scheduler' in snapshot):
            scheduler.load_state_dict(snapshot['scheduler'])
        if (scaler is not None) and ('scaler' in snapshot) and strict:
            scaler.load_state_dict(snapshot['scaler'])
        logging.info('model snapshot loaded from %s', pth_fpath)
    else:
        logging.warning('model snapshot file not found: %s', pth_fpath)

    # restore the snapshot for evaluation metrics
    pth_fpath = os.path.join(mdl_dpath, 'snapshot_metrics.pth')
    if os.path.exists(pth_fpath):
        snapshot = torch.load(pth_fpath, map_location='cpu')
        metrics_list_intra = snapshot['metrics_list_intra']
        metrics_list_inter = snapshot['metrics_list_inter']
        logging.info('metric snapshot loaded from %s', pth_fpath)
    else:
        metrics_list_intra, metrics_list_inter = [], []
        logging.warning('metric snapshot file not found: %s', pth_fpath)

    return metrics_list_intra, metrics_list_inter


def load_snapshot_ckpt(mdl_dpath, model_base, model_trgt, strict=True):
    """Load snapshot files for models and evaluation metrics.

    Args:
    * mdl_dpath: directory path to snapshot files
    * model_base: base model
    * model_trgt: target model (updated via EMA)
    * strict: (optional) whether missing and/or unexpected keys are allowed

    Returns:
    """

    # restore the snapshot for models
    pth_fpath = mdl_dpath
    logging.info('model params loaded from %s', pth_fpath)
    if os.path.exists(pth_fpath):
        # restore base & target models
        snapshot = torch.load(pth_fpath, map_location='cpu')
        keys_miss, keys_uexp = model_base.load_state_dict(snapshot, strict=strict)
        report_abnormal_keys(keys_miss, keys_uexp)
        keys_miss, keys_uexp = model_trgt.load_state_dict(snapshot, strict=strict)
        report_abnormal_keys(keys_miss, keys_uexp)
    else:
        exit()

    return
