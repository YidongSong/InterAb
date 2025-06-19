"""Checkpoint recorder for multi-model ensembles."""

import os
import logging

from tfold.utils import save_model


class CkptRecorder():
    """Checkpoint recorder for multi-model ensembles."""

    def __init__(
            self,
            mdl_dpath,  # directory path to the multi-model ensemble
            mode='min',  # lower evaluation metric value corresponds to a better model
            topk=4,  # number of models in the ensemble
        ):
        """Constructor function."""

        # setup configurations
        self.mdl_dpath = mdl_dpath
        self.mode = mode
        self.topk = topk
        assert self.mode in ['min', 'max'], f'unrecognized model selection mode: {self.mode}'

        # additional configurations
        self.vals_opt = []  # list of optimal models' evaluation metric values


    def restore(self, vals_all):
        """Restore <vals_opt> from a list of evaluation metric values."""

        vals_all.sort(reverse=(self.mode == 'max'))
        self.vals_opt = vals_all[:self.topk]


    def update(self, model, val_curr):
        """Update the multi-model ensemble."""

        # report the performance of mult-model ensemble & current model
        logging.info('multi-model ensemble: %s', ', '.join([f'{x:.4f}' for x in self.vals_opt]))
        logging.info('current model: %.4f', val_curr)

        # find the first model in the ensemble w/ better performance than the current one
        for idx_pvt in range(len(self.vals_opt) - 1, -2, -1):
            if idx_pvt == -1:  # all the models in the ensemble are inferior to the current one
                break
            if (self.mode == 'max') and (self.vals_opt[idx_pvt] > val_curr):
                break
            if (self.mode == 'min') and (self.vals_opt[idx_pvt] < val_curr):
                break

        # early exit if the multi-model ensemble does not need to be updated
        if (len(self.vals_opt) == self.topk) and (idx_pvt == len(self.vals_opt) - 1):
            logging.info('multi-model ensemble is not updated')
            return

        # rename all the subsequent models (after the pivot one) in the ensemble
        if len(self.vals_opt) < self.topk:
            self.vals_opt.append(0.0)
        for idx_src in range(len(self.vals_opt) - 2, idx_pvt, -1):
            idx_dst = idx_src + 1
            pth_fpath_src = os.path.join(self.mdl_dpath, f'model_{idx_src}.pth')
            pth_fpath_dst = os.path.join(self.mdl_dpath, f'model_{idx_dst}.pth')
            os.rename(pth_fpath_src, pth_fpath_dst)
            self.vals_opt[idx_dst] = self.vals_opt[idx_src]

        # save the current model into the ensemble
        pth_fpath = os.path.join(self.mdl_dpath, f'model_{idx_pvt + 1}.pth')
        save_model(model, pth_fpath)
        self.vals_opt[idx_pvt + 1] = val_curr

        # report the performance of updated multi-model ensemble
        logging.info('multi-model ensemble: %s', ', '.join([f'{x:.4f}' for x in self.vals_opt]))
