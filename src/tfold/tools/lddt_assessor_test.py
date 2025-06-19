"""Unit-tests for the <LddtAssessor> class."""

import os
import logging

from tfold.utils import tfold_init
from tfold.tools import LddtAssessor
from tfold.tools import ProtStruct
from tfold.tools import ProtConverter


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    prot_id = 'T1024-D1'
    atom_sets = ['ca', 'fa']
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, 'examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath_natv = os.path.join(data_dir, f'{prot_id}_native.pdb')
    pdb_fpath_decy = os.path.join(data_dir, f'{prot_id}_decoy.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')
    assessor = LddtAssessor()
    converter = ProtConverter()

    # initialize protein structures with native & decoy PDB files
    struct_ref = ProtStruct()
    struct_ref.init_from_file(fas_fpath, pdb_fpath_natv)
    struct_qry = ProtStruct()
    struct_qry.init_from_file(fas_fpath, pdb_fpath_decy)

    # calculate per-residue lDDT scores (Ca-only / full-atom)
    cmsk_mat = struct_ref.cmsk_mat * struct_qry.cmsk_mat
    for atom_set in atom_sets:
        logging.info('=== atom set: %s ===', atom_set)
        plddt_vec, plmsk_vec, clddt_val = \
            assessor.run(struct_ref.cord_tns, struct_qry.cord_tns, cmsk_mat, atom_set)
        logging.info('plddt_vec: %s / %s', plddt_vec.shape, plddt_vec.dtype)
        logging.info('plmsk_vec: %s / %s', plmsk_vec.shape, plmsk_vec.dtype)
        logging.info('clddt_val: %.4f', clddt_val.item())

    # calculate per-residue lDDT scores (Ca-only / full-atom) w/ symmetric renaming
    logging.info('renaming symmetric ground-truth atoms in the query structure')
    struct_ref.build_fram_n_angl(converter, build_sc=True)
    struct_ref.rename_sym_atoms(struct_qry.cord_tns, struct_qry.cmsk_mat, converter)
    for atom_set in atom_sets:
        logging.info('=== atom set: %s ===', atom_set)
        plddt_vec, plmsk_vec, clddt_val = \
            assessor.run(struct_ref.cord_tns, struct_qry.cord_tns, cmsk_mat, atom_set)
        logging.info('plddt_vec: %s / %s', plddt_vec.shape, plddt_vec.dtype)
        logging.info('plmsk_vec: %s / %s', plmsk_vec.shape, plmsk_vec.dtype)
        logging.info('clddt_val: %.4f', clddt_val.item())


if __name__ == '__main__':
    main()
