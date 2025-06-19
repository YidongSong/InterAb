"""Protein-related encoders."""

import itertools

import torch
from torch import nn
import numpy as np

from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


class OnhtEncoder():
    """One-hot encoder."""

    def __init__(self, vocab):
        """Constructor function."""

        self.vocab = vocab


    @property
    def n_dims(self):
        """Get the number of dimensions needed for one-hot encodings."""

        return len(self.vocab)


    def name2idx(self, names):
        """Convert names into indices.

        Args:
        * names: list of names of length L

        Returns:
        * idxs_vec: indices of size L
        """

        idxs_vec = torch.tensor([self.vocab.index(x) for x in names], dtype=torch.int64)

        return idxs_vec


    def name2onht(self, names):
        """Convert names into one-hot encodings.

        Args:
        * names: list of names of length L

        Returns:
        * onht_mat: one-hot encodings of size L x D
        """

        idxs_vec = self.name2idx(names)
        onht_mat = nn.functional.one_hot(idxs_vec, self.n_dims)

        return onht_mat


    def idx2name(self, idxs_vec):
        """Convert indices into names.

        Args:
        * idxs_vec: indices of size L

        Returns:
        * names: list of names of length L
        """

        idxs_vec_np = idxs_vec.detach().cpu().numpy()
        names = [self.vocab[x] for x in np.nditer(idxs_vec_np)]

        return names


    def onht2name(self, onht_mat):
        """Convert one-hot encodings into residue names.

        Args:
        * onht_mat: one-hot encodings of size L x D

        Returns:
        * names: list of names of length L

        Note:
        * It is also okay to feed predicted probabilities for conversion.
        """

        idxs_vec = torch.argmax(onht_mat, dim=1)
        names = self.idx2name(idxs_vec)

        return names


class ResdEncoderV2(OnhtEncoder):
    """One-hot encoder for residue names."""

    def __init__(self):
        """Constructor function."""

        super().__init__(RESD_NAMES_1C)


class AtomEncoderV2(OnhtEncoder):
    """One-hot encoder for atom names."""

    def __init__(self):
        """Constructor function."""

        atom_names = sorted(list(set(itertools.chain.from_iterable(ATOM_NAMES_PER_RESD.values()))))
        super().__init__(atom_names)


class ElemEncoderV2(OnhtEncoder):
    """One-hot encoder for chemical element names."""

    def __init__(self):
        """Constructor function."""

        elem_names = ['C', 'N', 'O', 'S']
        super().__init__(elem_names)


class ResdEncoder():
    """One-hot encoder for residue names."""

    def __init__(self):
        """Constructor function."""

        self.resd_names = RESD_NAMES_1C


    @property
    def n_dims(self):
        """Get the number of dimensions needed for one-hot encodings."""

        return len(self.resd_names)


    def name2onht(self, resd_names):
        """Convert residue names into one-hot encodings.

        Args:
        * resd_names: list of residue names

        Returns:
        * onht_mat: one-hot encodings of size L x D
        """

        idxs_vec = torch.tensor([self.resd_names.index(x) for x in resd_names], dtype=torch.int64)
        onht_mat = nn.functional.one_hot(idxs_vec, self.n_dims)

        return onht_mat


    def onht2name(self, onht_mat):
        """Convert one-hot encodings into residue names.

        Args:
        * onht_mat: one-hot encodings of size L x D

        Returns:
        * resd_names: list of residue names

        Note:
        * It is also okay to feed predicted probabilities for conversion.
        """

        idxs_vec_np = torch.argmax(onht_mat, dim=1).detach().cpu().numpy()
        resd_names = [self.resd_names[x] for x in np.nditer(idxs_vec_np)]

        return resd_names


class AtomEncoder():
    """Encoder for element types & atom names."""

    def __init__(self):
        """Constructor function."""

        self.atom_elems = ['C', 'N', 'O', 'S']
        atom_names = set(itertools.chain.from_iterable(ATOM_NAMES_PER_RESD.values()))
        self.atom_names = sorted(list(atom_names))


    @property
    def n_dims_elem(self):
        """Get the number of dimensions needed for one-hot encodings of atom elements."""

        return len(self.atom_elems)


    @property
    def n_dims_name(self):
        """Get the number of dimensions needed for one-hot encodings of atom names."""

        return len(self.atom_names)


    def elem2onht(self, atom_elems):
        """Convert atom elements into one-hot encodings.

        Args:
        * atom_elems: list of atom elements

        Returns:
        * onht_mat: one-hot encodings of size L x D
        """

        idxs_vec = torch.tensor([self.atom_elems.index(x) for x in atom_elems], dtype=torch.int64)
        onht_mat = nn.functional.one_hot(idxs_vec, self.n_dims_elem)

        return onht_mat


    def name2onht(self, atom_names):
        """Convert atom names into one-hot encodings.

        Args:
        * atom_names: list of atom names

        Returns:
        * onht_mat: one-hot encodings of size L x D
        """

        idxs_vec = torch.tensor([self.atom_names.index(x) for x in atom_names], dtype=torch.int64)
        onht_mat = nn.functional.one_hot(idxs_vec, self.n_dims_name)

        return onht_mat


    def onht2elem(self, onht_mat):
        """Convert one-hot encodings into atom elements.

        Args:
        * onht_mat: one-hot encodings of size L x D

        Returns:
        * atom_elems: list of atom elements
        """

        idxs_vec_np = torch.argmax(onht_mat, dim=1).detach().cpu().numpy()
        atom_elems = [self.atom_elems[x] for x in np.nditer(idxs_vec_np)]

        return atom_elems


    def onht2name(self, onht_mat):
        """Convert one-hot encodings into atom names.

        Args:
        * onht_mat: one-hot encodings of size L x D

        Returns:
        * atom_names: list of atom names
        """

        idxs_vec_np = torch.argmax(onht_mat, dim=1).detach().cpu().numpy()
        atom_names = [self.atom_names[x] for x in np.nditer(idxs_vec_np)]

        return atom_names
