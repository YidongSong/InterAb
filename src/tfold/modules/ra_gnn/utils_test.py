"""Unit-tests for utility functions."""

import random
import logging

import torch

from tfold.utils import tfold_init
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD
from tfold.modules.ra_gnn.utils import sp2ds
from tfold.modules.ra_gnn.utils import ds2sp


def main():
    """Main entry."""

    # configurations
    n_resds = 32
    aa_seq = ''.join([random.choice(RESD_NAMES_1C) for _ in range(n_resds)])

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize per-atom 3D coordinates & validness masks
    cord_tns = torch.randn((n_resds, N_ATOMS_PER_RESD, 3), dtype=torch.float32)
    cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8)
    for idx_resd, resd_name in enumerate(aa_seq):
        atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
        cmsk_mat[idx_resd, :len(atom_names)] = 1
    cmsk_mat = torch.minimum(cmsk_mat, torch.randint(2, cmsk_mat.shape, dtype=torch.int8))
    cord_tns *= cmsk_mat.unsqueeze(dim=2)

    # test w/ sparse-to-dense (and vice-versa) conversion routines
    cord_tns_ds = sp2ds(aa_seq, cord_tns, cmsk_mat)
    cord_tns_sp = ds2sp(aa_seq, cord_tns_ds, cmsk_mat)
    logging.info('conversion error: %.4f', torch.norm(cord_tns_sp - cord_tns).item())


if __name__ == '__main__':
    main()
