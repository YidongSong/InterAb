"""Distributed sampler w/ load balance - v3.

Notes:
* The main difference between <DistrSamplerLBV2> and <DistrSamplerLBV3> is that the latter one
    supports sampling padding for the last non-full batch, while the former one always drops the
    last K elements. This will cause inaccurate results, especially for small validation subsets.
"""

import random

import torch
from torch.utils.data import Sampler


class DistrSamplerLBV3(Sampler):
    """Distributed sampler w/ load balance - v3."""

    def __init__(self, dataset, num_replicas=None, rank=None, seed=0, drop_last=False):
        """Constructor function."""

        super().__init__(dataset)

        # setup configurations
        self.dataset = dataset
        self.n_workers = num_replicas
        self.idx_worker = rank
        self.seed = seed
        self.drop_last = drop_last

        # additional configurations
        self.idx_epoch = -1
        self.ds_len = len(self.dataset)
        if self.drop_last or (self.ds_len % self.n_workers == 0):
            self.n_smpls_lcl = self.ds_len // self.n_workers
            self.n_smpls_all = self.n_smpls_lcl * self.n_workers
        else:
            self.n_smpls_lcl = self.ds_len // self.n_workers + 1
            self.n_smpls_all = self.n_smpls_lcl * self.n_workers
            self.n_smpls_pad = self.n_smpls_all - self.ds_len


    def __len__(self):
        """Get the epoch length (for the current worker)."""

        return self.n_smpls_lcl


    def __iter__(self):
        """Get an iterator of sample indices."""

        # initialization
        seed = self.seed + self.idx_epoch  # random seed for the current epoch
        random.seed(seed)

        # sample a subset of data samples for the current epoch
        cost_list = self.dataset.sample_subset(seed)
        assert len(cost_list) == self.ds_len, 'unexpected length of computational cost list'
        if self.ds_len > self.n_smpls_all:
            cost_list = random.sample(cost_list, self.n_smpls_all)
        elif self.ds_len < self.n_smpls_all:
            n_repts = (self.n_smpls_pad + self.ds_len - 1) // self.ds_len
            cost_list += random.sample(cost_list * n_repts, self.n_smpls_pad)
        cost_list.sort(key=lambda x: x[1])

        # group sample indices based on the number of workers
        idxs_smpl_list = []
        for idx_smpl_beg in range(0, self.n_smpls_all, self.n_workers):
            idx_smpl_end = idx_smpl_beg + self.n_workers
            idxs_smpl_list.append([x[0] for x in cost_list[idx_smpl_beg:idx_smpl_end]])

        # determine sample indices for the current worker
        random.shuffle(idxs_smpl_list)
        idxs_smpl_lcl = [x[self.idx_worker] for x in idxs_smpl_list]

        return iter(idxs_smpl_lcl)


    def set_epoch(self, idx_epoch):
        """Set the epoch index to ensure different orderings for different epochs."""

        self.idx_epoch = idx_epoch
