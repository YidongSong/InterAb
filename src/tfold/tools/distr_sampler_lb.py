"""Distributed sampler w/ load balance."""

import numpy as np
import torch
from torch.utils.data import Sampler


class DistrSamplerLB(Sampler):
    """Distributed sampler w/ load balance."""

    def __init__(self, dataset, num_replicas=None, rank=None, drop_last=False):
        """Constructor function."""

        super().__init__(dataset)

        # setup configurations
        self.dataset = dataset
        self.n_workers = num_replicas
        self.idx_worker = rank
        self.drop_last = drop_last

        # additional configurations
        self.epoch = 0
        self.cost_list = self.dataset.get_cost_list()
        if self.drop_last:
            self.n_smpls_lcl = len(dataset) // self.n_workers
        else:
            self.n_smpls_lcl = (len(dataset) + self.n_workers - 1) // self.n_workers


    def __len__(self):
        """Get the epoch length (for the current worker)."""

        return self.n_smpls_lcl


    def __iter__(self):
        """Get an iterator of sample indices."""

        # randomly shuffle sample indices
        generator = torch.Generator()
        generator.manual_seed(self.epoch)
        idxs_smpl_ttl = torch.randperm(len(self.dataset), generator=generator).tolist()

        # handle the last mini-batch
        n_smpls_ttl = self.n_smpls_lcl * self.n_workers
        if self.drop_last:
            idxs_smpl_ttl = idxs_smpl_ttl[:n_smpls_ttl]
        else:
            n_repts = (n_smpls_ttl + len(idxs_smpl_ttl) - 1) // len(idxs_smpl_ttl)
            idxs_smpl_ttl = (n_repts * idxs_smpl_ttl)[:n_smpls_ttl]

        # sub-sample sample indices
        idxs_smpl_lcl = idxs_smpl_ttl[self.idx_worker::self.n_workers]
        assert len(idxs_smpl_lcl) == self.n_smpls_lcl

        # re-arrange sample indices to balance the computational cost
        cost_list_ttl = self.dataset.get_cost_list()
        cost_list_lcl = [cost_list_ttl[x] for x in idxs_smpl_lcl]
        idxs_sort = np.argsort(cost_list_lcl).tolist()
        n_smpls_hlf = self.n_smpls_lcl // 2
        idxs_sort_head = idxs_sort[:n_smpls_hlf]
        idxs_sort_tail = idxs_sort[-n_smpls_hlf:]
        idxs_smpl_lcl_new = []
        for idx in torch.randperm(n_smpls_hlf, generator=generator).tolist():
            idxs_smpl_lcl_new.append(idxs_smpl_lcl[idxs_sort_head[idx]])
            idxs_smpl_lcl_new.append(idxs_smpl_lcl[idxs_sort_tail[-idx]])
        if self.n_smpls_lcl % 2 == 1:
            idxs_smpl_lcl_new.append(idxs_smpl_lcl[idxs_sort[n_smpls_hlf]])
        idxs_smpl_lcl = idxs_smpl_lcl_new

        return iter(idxs_smpl_lcl)


    def set_epoch(self, epoch):
        """Set the epoch to ensure different orderings for different epochs."""

        self.epoch = epoch
