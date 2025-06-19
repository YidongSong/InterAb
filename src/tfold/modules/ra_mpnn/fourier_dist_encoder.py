"""Fourier distance encoder.

Notes:
* The reference distance values (as denominators in sine & cosine functions) are:
    power(base, t) / cord_scale, t = 0, 1, ..., n_dims // 2 - 1
  where <cord_scale> is the scaling factor applied on 3D coordinates.
"""

import torch


class FourierDistEncoder():  # pylint: disable=too-few-public-methods
    """Fourier distance encoder."""

    def __init__(self, n_dims, base=2.0, cord_scale=1.0):
        """Constructor function."""

        # initialization
        self.n_dims = n_dims
        self.base = base
        self.cord_scale = cord_scale
        assert self.n_dims % 2 == 0, '# of dimensions in distance encodings must be even'

        # additional configurations
        self.n_freqs = self.n_dims // 2
        self.freq_vec = torch.pow(self.base, torch.arange(self.n_freqs)) / self.cord_scale


    def run(self, dist_tns):
        """Run the distance encoder.

        Args:
        * dist_tns: distance values of size D1 (x D2 x D3 x ...)

        Returns:
        * encd_tns: distance encodings of size D1 (x D2 x D3 x ...) x De
        """

        if self.freq_vec.device != dist_tns.device:
            self.freq_vec = self.freq_vec.to(dist_tns.device)

        encd_tns = torch.cat([
            torch.sin(dist_tns.unsqueeze(dim=-1) / self.freq_vec),
            torch.cos(dist_tns.unsqueeze(dim=-1) / self.freq_vec),
        ], dim=-1)

        return encd_tns
