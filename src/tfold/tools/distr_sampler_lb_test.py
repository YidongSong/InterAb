"""Unit-tests for <DistrSamplerLB>."""

import random
import logging

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from tfold.utils import tfold_init
from tfold.tools import DistrSamplerLB


class DemoDataset(Dataset):
    """Demonstration dataset."""

    def __init__(self):
        """Constructor function."""

        super().__init__()

        self.len_min = 50
        self.len_max = 400
        self.n_items = 4096
        self.item_list = [random.randint(self.len_min, self.len_max) for _ in range(self.n_items)]


    def get_cost_list(self):
        """Get a list of estimated computational cost."""

        return [x ** 2 for x in self.item_list]  # assume O(N^2) complexity


    def __len__(self):
        """Get the number of elements in the dataset."""

        return self.n_items


    def __getitem__(self, idx):
        """Get the i-th element in the dataset."""

        return self.item_list[idx]


def main():
    """Main entry."""

    # configurations
    n_workers = 8
    idx_worker = 0
    n_epochs = 4
    n_batches = 16

    # initialization
    tfold_init()

    # test w/ <DistrSamplerLB>
    dataset = DemoDataset()
    sampler = DistrSamplerLB(dataset, n_workers, idx_worker)
    data_loader = DataLoader(
        dataset, batch_size=1, sampler=sampler,
        num_workers=1, collate_fn=lambda x: x[0], prefetch_factor=1,
    )
    for idx_epoch in range(n_epochs):
        logging.info('epoch #%d', idx_epoch + 1)
        sampler.set_epoch(idx_epoch)
        for idx_batch, item in enumerate(data_loader):
            logging.info('sequence length: %d', item)
            if idx_batch + 1 == n_batches:
                break


if __name__ == '__main__':
    main()
