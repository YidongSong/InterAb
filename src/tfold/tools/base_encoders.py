"""Base encoders."""

import numpy as np
import torch

from tfold.utils import rot2quat


class PosiEncoder():  # pylint: disable=too-few-public-methods
    """Positional encoder."""

    def __init__(self, n_dims=32, pos_max=1024):
        """Constructor function."""

        # initialization
        assert n_dims % 2 == 0, 'number of dimensions in positional encodings must be even'
        self.n_dims = n_dims
        self.pos_max = pos_max

        # additional configurations
        self.n_freqs = self.n_dims // 2
        self.freq_vec = torch.pow(self.pos_max, torch.arange(self.n_freqs) / (self.n_freqs - 1))


    def run(self, idxs_vec):
        """Run the positional encoder.

        Args:
        * idxs_vec: residue indices of size N

        Returns:
        * encd_mat: positional encodings of size N x D
        """

        if self.freq_vec.device != idxs_vec.device:
            self.freq_vec = self.freq_vec.to(idxs_vec.device)

        encd_mat = torch.cat([
            torch.sin(idxs_vec.view(-1, 1) / self.freq_vec.view(1, -1)),
            torch.cos(idxs_vec.view(-1, 1) / self.freq_vec.view(1, -1)),
        ], dim=1)

        return encd_mat


class DistEncoder():  # pylint: disable=too-few-public-methods
    """Distance encoder."""

    def __init__(self, n_dims=16, dist_min=0.0, dist_max=20.0):
        """Constructor function."""

        # initialization
        self.n_dims = n_dims
        self.dist_min = dist_min
        self.dist_max = dist_max

        # additional configurations
        self.bin_wid = (self.dist_max - self.dist_min) / self.n_dims
        self.freq_vec = self.dist_min + self.bin_wid * (torch.arange(self.n_dims) + 0.5)
        self.div_fctr = np.sqrt(2 * np.pi) * self.bin_wid


    def run(self, dist_vec):
        """Run the distance encoder.

        Args:
        * dist_vec: distance vector of size N

        Returns:
        * encd_mat: distance encodings of size N x D
        """

        if self.freq_vec.device != dist_vec.device:
            self.freq_vec = self.freq_vec.to(dist_vec.device)

        diff_mat = dist_vec.view(-1, 1) - self.freq_vec.view(1, -1)
        encd_mat = torch.exp(-torch.square(diff_mat) / (2 * self.bin_wid ** 2)) / self.div_fctr

        return encd_mat


class AnglEncoder():  # pylint: disable=too-few-public-methods
    """Angle encoder."""

    def __init__(self):
        """Constructor function."""

        # initialization
        self.n_dims = 2


    def run(self, angl_vec):
        """Run the angle encoder.

        Args:
        * angl_vec: angle vector of size N

        Returns:
        * encd_mat: angle encodings of size N x 2
        """

        encd_mat = torch.stack([torch.sin(angl_vec), torch.cos(angl_vec)], dim=1)

        return encd_mat


class FramEncoder():  # pylint: disable=too-few-public-methods
    """Frame encoder."""

    def __init__(self, dist_encoder):
        """Constructor function."""

        # initialization
        self.dist_encoder = dist_encoder

        # additional configurations
        self.eps = 1e-6
        self.n_dims_quat = 4  # quaternion encodings
        self.n_dims_dist = self.dist_encoder.n_dims  # distance encodings
        self.n_dims_drct = 3  # direction encodings
        self.n_dims = self.n_dims_quat + self.n_dims_dist + self.n_dims_drct


    def run(self, rota_tns_pri, trsl_mat_pri, rota_tns_sec, trsl_mat_sec):
        """Run the orientation encoder.

        Args:
        * rota_tns_pri: rotation matrices of size N x 3 x 3
        * trsl_mat_pri: translation vectors of size N x 3
        * rota_tns_sec: rotation matrices of size N x 3 x 3
        * trsl_mat_sec: translation vectors of size N x 3

        Returns:
        * encd_mat: frame & coordinate encodings of size N x D
        """

        # quaternion encodings
        rota_tns_rlt = torch.sum(
            rota_tns_pri.permute(0, 2, 1).unsqueeze(dim=3) * rota_tns_sec.unsqueeze(dim=1), dim=2)
        encd_mat_quat = rot2quat(rota_tns_rlt, quat_type='full')

        # distance encodings
        dist_vec = torch.norm(trsl_mat_sec - trsl_mat_pri, dim=1)
        encd_mat_dist = self.dist_encoder.run(dist_vec)

        # direction encodings
        dcrd_mat = trsl_mat_sec - trsl_mat_pri
        dcrd_mat_norm = dcrd_mat / (torch.norm(dcrd_mat, dim=1, keepdim=True) + self.eps)
        encd_mat_drct = torch.sum(
            rota_tns_pri.permute(0, 2, 1) * dcrd_mat_norm.unsqueeze(dim=1), dim=2)

        # concatenate direction & quaternion encodings together
        encd_mat = torch.cat([encd_mat_quat, encd_mat_dist, encd_mat_drct], dim=1)

        return encd_mat


class FrcdEncoder():  # pylint: disable=too-few-public-methods
    """Frame & coordinate encoder."""

    def __init__(self, dist_encoder, n_grps_cord):
        """Constructor function."""

        # initialization
        self.dist_encoder = dist_encoder
        self.n_grps_cord = n_grps_cord

        # additional configurations
        self.eps = 1e-6
        self.n_dims_quat = 4  # quaternion encodings
        self.n_dims_dist = self.dist_encoder.n_dims  # distance encodings
        self.n_dims_drct = 3 * self.n_grps_cord  # direction encodings
        self.n_dims = self.n_dims_quat + self.n_dims_dist + self.n_dims_drct


    def run(self, rota_tns_pri, trsl_mat_pri, cord_tns_pri,
            rota_tns_sec, trsl_mat_sec, cord_tns_sec,
        ):  # pylint: disable=too-many-arguments,too-many-locals
        """Run the orientation encoder.

        Args:
        * rota_tns_pri: rotation matrices of size N x 3 x 3
        * trsl_mat_pri: translation vectors of size N x 3
        * cord_tns_pri: 3D coordinates of size N x G x 3
        * rota_tns_sec: rotation matrices of size N x 3 x 3
        * trsl_mat_sec: translation vectors of size N x 3
        * cord_tns_sec: 3D coordinates of size N x G x 3

        Returns:
        * encd_mat: frame & coordinate encodings of size N x D
        """

        # initialization
        n_frams = rota_tns_pri.shape[0]
        assert cord_tns_pri.shape[1] == self.n_grps_cord
        assert cord_tns_sec.shape[1] == self.n_grps_cord

        # quaternion encodings
        rota_tns_rlt = torch.sum(
            rota_tns_pri.permute(0, 2, 1).unsqueeze(dim=3) * rota_tns_sec.unsqueeze(dim=1), dim=2)
        encd_mat_quat = rot2quat(rota_tns_rlt, quat_type='full')

        # distance encodings
        dist_vec = torch.norm(trsl_mat_sec - trsl_mat_pri, dim=1)
        encd_mat_dist = self.dist_encoder.run(dist_vec)

        # direction encodings
        dcrd_tns = cord_tns_sec - cord_tns_pri
        dcrd_tns_norm = dcrd_tns / (torch.norm(dcrd_tns, dim=2, keepdim=True) + self.eps)
        encd_mat_drct = torch.sum(
            rota_tns_pri.permute(0, 2, 1).unsqueeze(dim=1) * dcrd_tns_norm.unsqueeze(dim=2)
        , dim=3).view(n_frams, self.n_dims_drct)

        # concatenate direction & quaternion encodings together
        encd_mat = torch.cat([encd_mat_quat, encd_mat_dist, encd_mat_drct], dim=1)

        return encd_mat
