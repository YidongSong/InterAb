"""Assessor for long-range contact predictions."""

import numpy as np
from scipy.spatial.distance import cdist


class CntcAssessor():
    """Assessor for long-range contact predictions."""

    def __init__(self):
        """Constructor function."""

        self.eps = 1e-6
        self.dist_thres = 8.0
        self.idx_bin_beg = 1  # 2.0 - 2.5 A (inclusive)
        self.idx_bin_end = 12  # 7.5 - 8.0 A (inclusive)
        self.sep = 24  # residue separation for long-range contacts


    def calc_prec_w_cord(self, cord_mat, mask_vec, prob_mat):
        """Calculate the top-L precision for long-range contact predictions from 3D coordinates.

        Args:
        * cord_mat: ground-truth 3D coordinates of size L x 3
        * mask_vec: ground-truth 3D coordinates' validness masks of size L
        * prob_mat: predicted inter-residue contact probabilities of size L x L

        Returns:
        * prec: top-L precision for long-range contact predictions
        """

        # calculate binary-values masks for inter-residue contacts
        mask_mat = np.outer(mask_vec, mask_vec)
        dist_mat = cdist(cord_mat, cord_mat, metric='euclidean')
        cmsk_mat = mask_mat * (dist_mat <= self.dist_thres).astype(np.int8)

        # calculate the top-L precision for long-range contact predictions
        prec = self.__calc_prec_impl(mask_mat, cmsk_mat, prob_mat)

        return prec


    def calc_prec_w_labl(self, labl_mat, mask_mat, prob_mat):
        """Calculate the top-L precision for long-range contact predictions from categorical labels.

        Args:
        * labl_mat: ground-truth categorical labels of size L x L
        * mask_mat: ground-truth categorical labels' validness masks of size L x L
        * prob_mat: predicted inter-residue contact probabilities of size L x L

        Returns:
        * prec: top-L precision for long-range contact predictions

        Note:
        The categorical label ranges from 0 to 36, inclusive.
        """

        # calculate binary-values masks for inter-residue contacts
        cmsk_mat = (labl_mat >= self.idx_bin_beg).astype(np.int8) \
            * (labl_mat <= self.idx_bin_end).astype(np.int8) * mask_mat

        # calculate the top-L precision for long-range contact predictions
        prec = self.__calc_prec_impl(mask_mat, cmsk_mat, prob_mat)

        return prec


    def __calc_prec_impl(self, vmsk_mat, cmsk_mat, prob_mat):
        """Calculate the top-L precision for long-range contact predictions."""

        # initialization
        n_resds = vmsk_mat.shape[0]

        # find-out top-L predicted inter-residue contacts
        cntc_infos = []
        for idx1 in range(n_resds):
            for idx2 in range(idx1 + self.sep, n_resds):
                if vmsk_mat[idx1, idx2] == 1:
                    cntc_infos.append((idx1, idx2, prob_mat[idx1, idx2]))
        cntc_infos.sort(key=lambda x: x[2], reverse=True)

        # count the number of correct predictions
        n_pairs_true = 0
        n_pairs_full = min(n_resds, len(cntc_infos))
        for idx1, idx2, _ in cntc_infos[:n_pairs_full]:
            if cmsk_mat[idx1, idx2] == 1:
                n_pairs_true += 1

        # calculcate the top-L precision for long-range contact predictions
        prec = n_pairs_true / (n_pairs_full + self.eps)

        return prec
