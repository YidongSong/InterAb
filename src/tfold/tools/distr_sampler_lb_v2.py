"""Distributed sampler w/ load balance - v2.

Notes:
* This data sampler supports load-balanced data sampling w/ subset selection.
* The dataset class should implement following methods:
  > get_candidates: get a full list of candidate data samples
  > sample_subset: sample a subset of data samples from all the candidates
* For simplicity, we always drop the last K elements that cannot be evenly divided.

Workflow:
* Get a full candidate list and sub-sampling function from the dataset.
* In each epoch:
  > Apply the sub-sampling function on the full candidate list to select data samples.
  > Arrange data samples in a load-balanced manner.
  > Assign data samples to each worker (GPUs, not data loader's underlying workers).
"""

import random

import torch
from torch.utils.data import Sampler


class DistrSamplerLBV2(Sampler):
    """Distributed sampler w/ load balance - v2."""

    def __init__(self, dataset, num_replicas=None, rank=None, seed=0):
        """Constructor function."""

        super().__init__(dataset)

        # setup configurations
        self.dataset = dataset
        self.n_workers = num_replicas
        self.idx_worker = rank
        self.seed = seed

        # additional configurations
        self.idx_epoch = -1
        self.n_smpls_lcl = len(self.dataset) // self.n_workers  # last K elements are dropped
        self.n_smpls_all = self.n_smpls_lcl * self.n_workers


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
        if len(cost_list) != self.n_smpls_all:
            cost_list = random.sample(cost_list, self.n_smpls_all)
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
