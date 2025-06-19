"""Utility functions."""

import logging


def disp_loss_n_metrics(loss, metrics, name=''):
    """Display the loss function value & evaluation metrics.

    Args:
    * loss: loss function value
    * metrics: dict of evaluation metrics
    * name: (optional) loss function name

    Returns: n/a
    """

    logging.info('[%s] Loss: %.2e', name, loss.item())
    for key, val in metrics.items():
        logging.info('[%s] %s: %.2e', name, key, val)
