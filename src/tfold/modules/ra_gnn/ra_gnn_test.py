"""Unit-tests for <ProtModelV2>."""

import random
import logging

import torch

from tfold.utils import tfold_init
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD
from tfold.modules.ra_gnn.ra_gnn import ResdAtomGNN


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    n_resds_all = 64
    n_resds_unk = 8  # number of residues w/ unknown amino-acid types
    n_resds_mis = 8  # number of residues w/ missing per-atom 3D coordinates

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize the amino-acid sequence w/ unknown residues
    resd_names = [random.choice(RESD_NAMES_1C) for _ in range(n_resds_all)]
    idxs_resd_unk = random.sample(range(n_resds_all), n_resds_unk)
    for idx_resd in idxs_resd_unk:
        resd_names[idx_resd] = 'X'  # replace w/ an unknown residue
    aa_seq = ''.join(resd_names)

    # build per-atom 3D coordinates' valid-or-not masks
    idxs_resd_mis = random.sample(range(n_resds_all), n_resds_mis)
    vmsk_mat = torch.zeros((n_resds_all, N_ATOMS_PER_RESD), dtype=torch.int8)
    for idx_resd, resd_name in enumerate(aa_seq):
        if resd_name != 'X':
            atom_names = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
        else:
            atom_names = ['C', 'CA', 'N', 'O']
        vmsk_mat[idx_resd, :len(atom_names)] = 1
    cmsk_mat = torch.minimum(vmsk_mat, torch.randint(2, vmsk_mat.shape, dtype=torch.int8))
    for idx_resd in idxs_resd_mis:
        cmsk_mat[idx_resd] = 0

    # randomly initialize per-atom 3D coordinates
    cord_tns = torch.randn((n_resds_all, N_ATOMS_PER_RESD, 3), dtype=torch.float32)
    cord_tns *= cmsk_mat.unsqueeze(dim=2)

    # randomly initialize per-atom 3D coordinates' update-or-not masks
    umsk_mat = torch.minimum(cmsk_mat, torch.randint(2, cmsk_mat.shape, dtype=torch.int8))

    # inspect input tensors
    logging.info('aa_seq: %s', aa_seq)
    logging.info('residues w/ missing atoms: %s', sorted(list(idxs_resd_mis)))
    logging.info('cord_tns: %s / %s', cord_tns.shape, cord_tns.dtype)
    logging.info('cmsk_mat: %s / %s', cmsk_mat.shape, cmsk_mat.dtype)
    logging.info('umsk_mat: %s / %s', umsk_mat.shape, umsk_mat.dtype)

    # test w/ <ResdAtomGNN>
    module = ResdAtomGNN()
    rfea_mat, afea_tns, acrd_tns = module(aa_seq, cord_tns, cmsk_mat, umsk_mat)
    logging.info('rfea_mat: %s / %s', rfea_mat.shape, rfea_mat.dtype)
    logging.info('afea_tns: %s / %s', afea_tns.shape, afea_tns.dtype)
    logging.info('acrd_tns: %s / %s', acrd_tns.shape, acrd_tns.dtype)

    # chech which atoms' 3D coordinates are updated
    dist_mat = torch.norm(acrd_tns - cord_tns, dim=2)
    for idx_resd in range(dist_mat.shape[0]):
        for idx_atom in range(dist_mat.shape[1]):
            if dist_mat[idx_resd, idx_atom] < 1e-6:
                continue
            header = f'R{idx_resd}|A{idx_atom}'
            if umsk_mat[idx_resd, idx_atom] == 1:
                logging.info('%s - update explicitly specified atoms', header)
            elif vmsk_mat[idx_resd, idx_atom] == 1 and cmsk_mat[idx_resd, idx_atom] == 0:
                logging.info('%s - update missing atoms', header)
            else:
                logging.warning('%s - unknown update operation', header)
                raise NotImplementedError


if __name__ == '__main__':
    main()
