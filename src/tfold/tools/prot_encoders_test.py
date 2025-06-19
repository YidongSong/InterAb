"""Unit-tests for protein-related encoders."""

import random
import logging

from tfold.utils import tfold_init
from tfold.tools import ResdEncoder
from tfold.tools import AtomEncoder
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD


def main():
    """Main entry."""

    # configurations
    resd_names = 'TLAEKELELIASWEHFAILNLIRMKTFKPEPEWIAERLALPLEKVQQSLELLLDLGFIK'
    atom_names = [random.choice(ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[x]]) for x in resd_names]
    #atom_names = ['CA', 'CB', 'CG1', 'CG2', 'N', 'S', 'OD1', 'OD2']
    atom_elems = [x[0] for x in atom_names]

    # initialization
    tfold_init(verb_levl='DEBUG')

    # test w/ <ResdEncoder>
    encoder = ResdEncoder()
    onht_mat = encoder.name2onht(resd_names)
    resd_names_new = encoder.onht2name(onht_mat)
    logging.info('residue names (old): %s', ''.join(resd_names))
    logging.info('residue names (new): %s', ''.join(resd_names_new))

    # test w/ <AtomEncoder>
    encoder = AtomEncoder()
    onht_mat = encoder.elem2onht(atom_elems)
    atom_elems_new = encoder.onht2elem(onht_mat)
    logging.info('atom elements (old): %s', ''.join(atom_elems))
    logging.info('atom elements (new): %s', ''.join(atom_elems_new))
    onht_mat = encoder.name2onht(atom_names)
    atom_names_new = encoder.onht2name(onht_mat)
    logging.info('atom names (old): %s', '|'.join(atom_names))
    logging.info('atom names (new): %s', '|'.join(atom_names_new))


if __name__ == '__main__':
    main()
