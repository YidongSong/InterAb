"""Protein visualizer (from NPZ & PDB files)."""

import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from tfold.utils import cdist
from tfold.tools.pdb_parser import PdbParser
from tfold.tools.prot_struct import ProtStruct


def softmax(x_tns, axis=-1):
    """Softmax function."""

    x_tns_sub = x_tns - np.max(x_tns, axis=axis, keepdims=True)
    z_tns = np.exp(x_tns_sub) / np.sum(np.exp(x_tns_sub), axis=axis, keepdims=True)

    return z_tns


class ProtVisualizer():  # pylint: disable=too-few-public-methods
    """Protein visualizer (from NPZ & PDB files)."""

    def __init__(self, nctc_pos='first'):
        """Constructor function."""

        self.nctc_pos = nctc_pos  # position of the non-contacting bin (choices: 'first' / 'last')
        assert self.nctc_pos in ['first', 'last'], \
            f'unrecognized position of the non-contacting bin: {self.nctc_pos}'
        self.eps = 1e-6
        self.dist_max = 20.0


    def run(self, npz_fpath, pdb_fpath, png_fpath):  # pylint: disable=too-many-locals
        """Run the protein visualizer."""

        # restore raw predictions from the NPZ file
        with np.load(npz_fpath) as npz_data:
            pred_tns_cb = npz_data['cb'][0].transpose(1, 2, 0)  # L x L x 37
            pred_tns_om = npz_data['om'][0].transpose(1, 2, 0)  # L x L x 25
            pred_tns_th = npz_data['th'][0].transpose(1, 2, 0)  # L x L x 25
            pred_tns_ph = npz_data['ph'][0].transpose(1, 2, 0)  # L x L x 25

        # restore 3D coordinates from the PDB file
        aa_seq, cord_tns, _, _, error_msg = PdbParser.load(pdb_fpath)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'

        # recover inter-residue CB-CB distance predictions
        dist_mat_npz = self.__calc_dist_mat_from_logits(pred_tns_cb)
        dist_mat_pdb = self.__calc_dist_mat_from_coords(aa_seq, cord_tns)

        # recover inter-residue omega, theta, and phi angle predictions
        angl_mat_om = self.__calc_angl_mat(pred_tns_om)
        angl_mat_th = self.__calc_angl_mat(pred_tns_th)
        angl_mat_ph = self.__calc_angl_mat(pred_tns_ph)

        # visualize distance & angle predictions, and export the figure to a PNG file
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))

        img = axes[0][0].imshow(dist_mat_npz)
        fig.colorbar(img, ax=axes[0][0])
        axes[0][0].set_title('CB-CB distance (NPZ)')
        img = axes[0][1].imshow(dist_mat_pdb)
        fig.colorbar(img, ax=axes[0][1])
        axes[0][1].set_title('CB-CB distance (PDB)')
        axes[0][2].axis('off')

        img = axes[1][0].imshow(angl_mat_om)
        fig.colorbar(img, ax=axes[1][0])
        axes[1][0].set_title('CA-CB-CB\'-CA\' dihedral angle')
        img = axes[1][1].imshow(angl_mat_th)
        fig.colorbar(img, ax=axes[1][1])
        axes[1][1].set_title('N-CA-CB-CB\' dihedral angle')
        img = axes[1][2].imshow(angl_mat_ph)
        fig.colorbar(img, ax=axes[1][2])
        axes[1][2].set_title('CA-CB-CB\' planar angle')

        os.makedirs(os.path.dirname(os.path.realpath(png_fpath)), exist_ok=True)
        plt.savefig(png_fpath)


    def __calc_dist_mat_from_logits(self, pred_tns):
        """Calculate the distance matrix from classification logits."""

        # configurations
        dist_min = 2.0
        dist_max = 20.0
        n_bins = 37
        bin_wid = (dist_max - dist_min) / (n_bins - 1)
        dist_vals = dist_min + bin_wid * (np.arange(n_bins - 1) + 0.5)

        # calculate the distance matrix
        prob_tns = softmax(pred_tns)
        if self.nctc_pos == 'first':
            prob_mat_nctc = prob_tns[:, :, 0]  # non-contacting bin
            prob_tns_cntc = prob_tns[:, :, 1:]  # contacting bins
        else:
            prob_tns_cntc = prob_tns[:, :, :-1]
            prob_mat_nctc = prob_tns[:, :, -1]
        dist_mat = \
            dist_max * prob_mat_nctc + np.sum(dist_vals[None, None, :] * prob_tns_cntc, axis=-1)

        return dist_mat


    def __calc_dist_mat_from_coords(self, aa_seq, cord_tns):
        """Calculate the distance matrix from 3D coordinates."""

        # initialization
        n_resds = len(aa_seq)
        device = cord_tns.device
        atom_names = ['CA', 'CB']

        # obtain 3D coordinates for CA & CB atoms
        cord_tns_sel = ProtStruct.get_atoms(aa_seq, cord_tns, atom_names)
        x_ca, x_cb = [torch.squeeze(x, dim=1) for x in torch.split(cord_tns_sel, 1, dim=1)]

        # use GLY's CA atom as the replacement for its missing CB atom
        is_gly = torch.tensor(
            [1 if aa_seq[x] == 'G' else 0 for x in range(n_resds)], dtype=torch.int8, device=device)
        x_cab = is_gly[:, None] * x_ca + (1 - is_gly[:, None]) * x_cb

        # calculate the CB-CB distance matrix (CA for Glycine)
        dist_mat = torch.clip(cdist(x_cab), 0.0, self.dist_max)

        return dist_mat


    def __calc_angl_mat(self, pred_tns):
        """Calculate the angle matrix."""

        # configurations
        angl_min = -np.pi
        angl_max = np.pi
        n_bins = 25
        bin_wid = (angl_max - angl_min) / (n_bins - 1)
        angl_vals = angl_min + bin_wid * (np.arange(n_bins - 1) + 0.5)

        # calculate the angle matrix
        prob_tns = softmax(pred_tns)
        prob_tns_cntc = prob_tns[:, :, 1:] if self.nctc_pos == 'first' else prob_tns[:, :, :-1]
        angl_mat = np.sum(angl_vals[None, None, :] * prob_tns_cntc, axis=-1) \
            / (np.sum(prob_tns_cntc, axis=-1) + self.eps)

        return angl_mat


def main():
    """Main entry."""

    # configurations
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    prot_id = 'UniRef50_A0A059X6M2'
    npz_fpath = os.path.join(data_dir, f'{prot_id}.npz')
    pdb_fpath = os.path.join(data_dir, f'{prot_id}.pdb')
    png_fpath = os.path.join(data_dir, f'{prot_id}.png')

    # test w/ <ProtVisualizer>
    visualizer = ProtVisualizer()
    visualizer.run(npz_fpath, pdb_fpath, png_fpath)


if __name__ == '__main__':
    main()
