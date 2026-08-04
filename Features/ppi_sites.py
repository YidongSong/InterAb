
import torch
from Bio import BiopythonWarning
import warnings
import logging
import os
import gzip
from collections import defaultdict
from collections import OrderedDict
from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException
from datetime import datetime
import random
import hashlib
import numpy as np
from torch import nn
from Bio import PDB
import pdb
import pickle
from tqdm import tqdm
import pandas as pd
import argparse


RESD_MAP_1TO3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL',
}
RESD_MAP_3TO1 = {v: k for k, v in RESD_MAP_1TO3.items()}
RESD_NAMES_1C = sorted(list(RESD_MAP_1TO3.keys()))
RESD_NAMES_3C = sorted(list(RESD_MAP_1TO3.values()))
N_ATOMS_PER_RESD = 14  # TRP
N_ANGLS_PER_RESD = 7  # TRP (omega, phi, psi, chi1, chi2, chi3, and chi4)
ATOM_NAMES_PER_RESD = {
    'ALA': ['C', 'CA', 'CB', 'N', 'O'],
    'ARG': ['C', 'CA', 'CB', 'CG', 'CD', 'CZ', 'N', 'NE', 'O', 'NH1', 'NH2'],
    'ASP': ['C', 'CA', 'CB', 'CG', 'N', 'O', 'OD1', 'OD2'],
    'ASN': ['C', 'CA', 'CB', 'CG', 'N', 'ND2', 'O', 'OD1'],
    'CYS': ['C', 'CA', 'CB', 'N', 'O', 'SG'],
    'GLU': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O', 'OE1', 'OE2'],
    'GLN': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'NE2', 'O', 'OE1'],
    'GLY': ['C', 'CA', 'N', 'O'],
    'HIS': ['C', 'CA', 'CB', 'CG', 'CD2', 'CE1', 'N', 'ND1', 'NE2', 'O'],
    'ILE': ['C', 'CA', 'CB', 'CG1', 'CG2', 'CD1', 'N', 'O'],
    'LEU': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'N', 'O'],
    'LYS': ['C', 'CA', 'CB', 'CG', 'CD', 'CE', 'N', 'NZ', 'O'],
    'MET': ['C', 'CA', 'CB', 'CG', 'CE', 'N', 'O', 'SD'],
    'PHE': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O'],
    'PRO': ['C', 'CA', 'CB', 'CG', 'CD', 'N', 'O'],
    'SER': ['C', 'CA', 'CB', 'N', 'O', 'OG'],
    'THR': ['C', 'CA', 'CB', 'CG2', 'N', 'O', 'OG1'],
    'TRP': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'N', 'NE1', 'O'],  # pylint: disable=line-too-long
    'TYR': ['C', 'CA', 'CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ', 'N', 'O', 'OH'],
    'VAL': ['C', 'CA', 'CB', 'CG1', 'CG2', 'N', 'O']
}
ANGL_INFOS_PER_RESD = {
    'ALA': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
    ],
    'ARG': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', False, ['CB', 'CG', 'CD', 'NE']],
        ['chi4', False, ['CG', 'CD', 'NE', 'CZ']],
    ],
    'ASN': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'OD1']],
    ],
    'ASP': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', True, ['CA', 'CB', 'CG', 'OD1']],
    ],
    'CYS': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'SG']],
    ],
    'GLN': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', False, ['CB', 'CG', 'CD', 'OE1']],
    ],
    'GLU': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', True, ['CB', 'CG', 'CD', 'OE1']],
    ],
    'GLY': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
    ],
    'HIS': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'ND1']],
    ],
    'ILE': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG1']],
        ['chi2', False, ['CA', 'CB', 'CG1', 'CD1']],
    ],
    'LEU': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'LYS': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
        ['chi3', False, ['CB', 'CG', 'CD', 'CE']],
        ['chi4', False, ['CG', 'CD', 'CE', 'NZ']],
    ],
    'MET': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'SD']],
        ['chi3', False, ['CB', 'CG', 'SD', 'CE']],
    ],
    'PHE': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', True, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'PRO': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD']],
    ],
    'SER': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'OG']],
    ],
    'THR': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'OG1']],
    ],
    'TRP': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', False, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'TYR': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG']],
        ['chi2', True, ['CA', 'CB', 'CG', 'CD1']],
    ],
    'VAL': [
        ['psi', False, ['N', 'CA', 'C', 'O']],
        ['chi1', False, ['N', 'CA', 'CB', 'CG1']],
    ],
}
ATOM_INFOS_PER_RESD = {
    'ALA': [
        ['N', 0, (-0.525, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.529, -0.774, -1.205)],
        ['O', 3, (0.627, 1.062, 0.000)],
    ],
    'ARG': [
        ['N', 0, (-0.524, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.524, -0.778, -1.209)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.616, 1.390, -0.000)],
        ['CD', 5, (0.564, 1.414, 0.000)],
        ['NE', 6, (0.539, 1.357, -0.000)],
        ['NH1', 7, (0.206, 2.301, 0.000)],
        ['NH2', 7, (2.078, 0.978, -0.000)],
        ['CZ', 7, (0.758, 1.093, -0.000)],
    ],
    'ASN': [
        ['N', 0, (-0.536, 1.357, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.531, -0.787, -1.200)],
        ['O', 3, (0.625, 1.062, 0.000)],
        ['CG', 4, (0.584, 1.399, 0.000)],
        ['ND2', 5, (0.593, -1.188, 0.001)],
        ['OD1', 5, (0.633, 1.059, 0.000)],
    ],
    'ASP': [
        ['N', 0, (-0.525, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, 0.000, -0.000)],
        ['CB', 0, (-0.526, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.593, 1.398, -0.000)],
        ['OD1', 5, (0.610, 1.091, 0.000)],
        ['OD2', 5, (0.592, -1.101, -0.003)],
    ],
    'CYS': [
        ['N', 0, (-0.522, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, 0.000)],
        ['CB', 0, (-0.519, -0.773, -1.212)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['SG', 4, (0.728, 1.653, 0.000)],
    ],
    'GLN': [
        ['N', 0, (-0.526, 1.361, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.779, -1.207)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.615, 1.393, 0.000)],
        ['CD', 5, (0.587, 1.399, -0.000)],
        ['NE2', 6, (0.593, -1.189, -0.001)],
        ['OE1', 6, (0.634, 1.060, 0.000)],
    ],
    'GLU': [
        ['N', 0, (-0.528, 1.361, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, -0.000, -0.000)],
        ['CB', 0, (-0.526, -0.781, -1.207)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG', 4, (0.615, 1.392, 0.000)],
        ['CD', 5, (0.600, 1.397, 0.000)],
        ['OE1', 6, (0.607, 1.095, -0.000)],
        ['OE2', 6, (0.589, -1.104, -0.001)],
    ],
    'GLY': [
        ['N', 0, (-0.572, 1.337, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.517, -0.000, -0.000)],
        ['O', 3, (0.626, 1.062, -0.000)],
    ],
    'HIS': [
        ['N', 0, (-0.527, 1.360, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.525, -0.778, -1.208)],
        ['O', 3, (0.625, 1.063, 0.000)],
        ['CG', 4, (0.600, 1.370, -0.000)],
        ['CD2', 5, (0.889, -1.021, 0.003)],
        ['ND1', 5, (0.744, 1.160, -0.000)],
        ['CE1', 5, (2.030, 0.851, 0.002)],
        ['NE2', 5, (2.145, -0.466, 0.004)],
    ],
    'ILE': [
        ['N', 0, (-0.493, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.536, -0.793, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.534, 1.437, -0.000)],
        ['CG2', 4, (0.540, -0.785, -1.199)],
        ['CD1', 5, (0.619, 1.391, 0.000)],
    ],
    'LEU': [
        ['N', 0, (-0.520, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.773, -1.214)],
        ['O', 3, (0.625, 1.063, -0.000)],
        ['CG', 4, (0.678, 1.371, 0.000)],
        ['CD1', 5, (0.530, 1.430, -0.000)],
        ['CD2', 5, (0.535, -0.774, 1.200)],
    ],
    'LYS': [
        ['N', 0, (-0.526, 1.362, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, 0.000)],
        ['CB', 0, (-0.524, -0.778, -1.208)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.619, 1.390, 0.000)],
        ['CD', 5, (0.559, 1.417, 0.000)],
        ['CE', 6, (0.560, 1.416, 0.000)],
        ['NZ', 7, (0.554, 1.387, 0.000)],
    ],
    'MET': [
        ['N', 0, (-0.521, 1.364, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, 0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.210)],
        ['O', 3, (0.625, 1.062, -0.000)],
        ['CG', 4, (0.613, 1.391, -0.000)],
        ['SD', 5, (0.703, 1.695, 0.000)],
        ['CE', 6, (0.320, 1.786, -0.000)],
    ],
    'PHE': [
        ['N', 0, (-0.518, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, 0.000, -0.000)],
        ['CB', 0, (-0.525, -0.776, -1.212)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.377, 0.000)],
        ['CD1', 5, (0.709, 1.195, -0.000)],
        ['CD2', 5, (0.706, -1.196, 0.000)],
        ['CE1', 5, (2.102, 1.198, -0.000)],
        ['CE2', 5, (2.098, -1.201, -0.000)],
        ['CZ', 5, (2.794, -0.003, -0.001)],
    ],
    'PRO': [
        ['N', 0, (-0.566, 1.351, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, 0.000)],
        ['CB', 0, (-0.546, -0.611, -1.293)],
        ['O', 3, (0.621, 1.066, 0.000)],
        ['CG', 4, (0.382, 1.445, 0.0)],
        ['CD', 5, (0.477, 1.424, 0.0)],
    ],
    'SER': [
        ['N', 0, (-0.529, 1.360, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, -0.000)],
        ['CB', 0, (-0.518, -0.777, -1.211)],
        ['O', 3, (0.626, 1.062, -0.000)],
        ['OG', 4, (0.503, 1.325, 0.000)],
    ],
    'THR': [
        ['N', 0, (-0.517, 1.364, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.526, 0.000, -0.000)],
        ['CB', 0, (-0.516, -0.793, -1.215)],
        ['O', 3, (0.626, 1.062, 0.000)],
        ['CG2', 4, (0.550, -0.718, -1.228)],
        ['OG1', 4, (0.472, 1.353, 0.000)],
    ],
    'TRP': [
        ['N', 0, (-0.521, 1.363, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.525, -0.000, 0.000)],
        ['CB', 0, (-0.523, -0.776, -1.212)],
        ['O', 3, (0.627, 1.062, 0.000)],
        ['CG', 4, (0.609, 1.370, -0.000)],
        ['CD1', 5, (0.824, 1.091, 0.000)],
        ['CD2', 5, (0.854, -1.148, -0.005)],
        ['CE2', 5, (2.186, -0.678, -0.007)],
        ['CE3', 5, (0.622, -2.530, -0.007)],
        ['NE1', 5, (2.140, 0.690, -0.004)],
        ['CH2', 5, (3.028, -2.890, -0.013)],
        ['CZ2', 5, (3.283, -1.543, -0.011)],
        ['CZ3', 5, (1.715, -3.389, -0.011)],
    ],
    'TYR': [
        ['N', 0, (-0.522, 1.362, 0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.524, -0.000, -0.000)],
        ['CB', 0, (-0.522, -0.776, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG', 4, (0.607, 1.382, -0.000)],
        ['CD1', 5, (0.716, 1.195, -0.000)],
        ['CD2', 5, (0.713, -1.194, -0.001)],
        ['CE1', 5, (2.107, 1.200, -0.002)],
        ['CE2', 5, (2.104, -1.201, -0.003)],
        ['OH', 5, (4.168, -0.002, -0.005)],
        ['CZ', 5, (2.791, -0.001, -0.003)],
    ],
    'VAL': [
        ['N', 0, (-0.494, 1.373, -0.000)],
        ['CA', 0, (0.000, 0.000, 0.000)],
        ['C', 0, (1.527, -0.000, -0.000)],
        ['CB', 0, (-0.533, -0.795, -1.213)],
        ['O', 3, (0.627, 1.062, -0.000)],
        ['CG1', 4, (0.540, 1.429, -0.000)],
        ['CG2', 4, (0.533, -0.776, 1.203)],
    ],
}

def get_chains_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_file)

    chains_info = []
    
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_count = len(chain)
            chains_info.append((chain_id, residue_count))
    chain_list = []
    for chain_id, residue_count in chains_info:
        chain_list.append(chain_id)

    return chain_list

def get_chain( structure, model_id, chain_id):
        """Get the first chain matching the specified chain ID (could be None)."""

        chain = None
        for model in structure:
            if (model_id is not None) and (model.get_id() != model_id):
                continue
            for chain_curr in model:
                if (chain_id is None) or (chain_curr.get_id() == chain_id):
                    chain = chain_curr
                    break
            if chain is not None:
                break

        # check whether the specified chain has been found
        if chain is None:
            raise PdbParseError('CHAIN_NOT_FOUND')

        return chain

def get_rand_str():
    """Get a randomized string.

    Args: n/a

    Returns:
    * rand_str: randomized string
    """

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    rand_val = random.random()
    rand_str_raw = f'{timestamp}_{rand_val}'
    rand_str = hashlib.md5(rand_str_raw.encode('utf-8')).hexdigest()

    return rand_str

def get_structure( path):
        """Get the structure from the PDB file."""

        try:
            parser = PDBParser()
            if path.endswith('.pdb'):
                with open(path, 'r', encoding='UTF-8') as i_file:
                    structure = parser.get_structure(get_rand_str(), i_file)
            else:  # then <path> must end with '.gz'
                with gzip.open(path, 'rt') as i_file:
                    structure = parser.get_structure(get_rand_str(), i_file)
        except PDBConstructionException as error:
            raise PdbParseError('BIOPYTHON_FAILED_TO_PARSE') from error

        return structure

def parse_fas_file(path):
    """Parse the FASTA file.

    Args:
    * path: path to the FASTA file (could be GZIP-compressed)

    Returns:
    * prot_id: protein ID (as in the commentary line)
    * aa_seq: amino-acid sequence
    """

    # parse all the lines in the FASTA file
    assert os.path.exists(path), f'FASTA file does not exist: {path}'
    if not path.endswith('.gz'):
        with open(path, 'r', encoding='UTF-8') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]
    else:
        with gzip.open(path, 'rt') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]

    # determine the protein ID & amino-acid sequence
    prot_id = i_lines[0][1:]
    aa_seq = ''.join(i_lines[1:])

    return prot_id, aa_seq

def get_line_strs(path):
        """Get line strings from the PDB file."""

        # obtain line strings from the PDB file
        if path.endswith('.pdb'):
            with open(path, 'r', encoding='UTF-8') as i_file:
                i_lines = [i_line.strip() for i_line in i_file]
        else:  # then <path> must end with '.gz'
            with gzip.open(path, 'rt') as i_file:
                i_lines = [i_line.strip() for i_line in i_file]

        return i_lines

def get_aa_seq_from_seqres(path, chain_id):
        """Get the amino-acid sequence from SEQRES records."""

        # get residue names from SEQRES records
        resd_names = []
        line_strs = get_line_strs(path)
        for line_str in line_strs:
            if not line_str.startswith('SEQRES'):
                continue
            if (chain_id is not None) and (line_str[11] != chain_id):
                continue
            resd_names.extend(line_str[19:].split())

        # convert residue names into the amino-acid sequence
        if len(resd_names) == 0:  # no SEQRES records found
            aa_seq = None
        else:
            for resd_name in resd_names:
                if resd_name not in RESD_NAMES_3C:
                    raise PdbParseError('HAS_UNKNOWN_RESIDUE')
            aa_seq = ''.join([RESD_MAP_3TO1[x] for x in resd_names])

        return aa_seq

def get_segs(chain):
        """Get discontinous segments of amino-acid sequences."""

        seg_infos = []
        idx_resd_prev = -9999
        ins_code_prev = ' '  # no insertion
        n_resds_ins = 0  # number of inserted residues
        for residue in chain:
            # obtain the current residue's basic information
            resd_name = residue.get_resname()
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if resd_name not in RESD_NAMES_3C:
                raise PdbParseError('HAS_UNKNOWN_RESIDUE')
            if (idx_resd == idx_resd_prev) and (ins_code != ins_code_prev):
                n_resds_ins += 1
            idx_resd_prev = idx_resd
            ins_code_prev = ins_code

            # update the last segment, or add a new segment
            if (len(seg_infos) >= 1) and (seg_infos[-1]['ie'] == idx_resd + n_resds_ins):
                seg_infos[-1]['ie'] += 1
                seg_infos[-1]['seq'] += RESD_MAP_3TO1[resd_name]
            else:
                seg_infos.append({
                    'ib': idx_resd + n_resds_ins,  # inclusive
                    'ie': idx_resd + n_resds_ins + 1,  # exclusive
                    'seq': RESD_MAP_3TO1[resd_name],
                })

        return seg_infos

def build_seq_from_segs(seg_infos):
        """Build the full amino-acid sequence from discontinous segments."""

        seg_infos.sort(key=lambda x: x['ib'])
        aa_seq_list = []
        for idx_seg, seg_info in enumerate(seg_infos):
            if idx_seg != 0:
                gap = seg_info['ib'] - seg_infos[idx_seg - 1]['ie']
                aa_seq_list.append('X' * gap)
            aa_seq_list.append(seg_info['seq'])
        aa_seq = ''.join(aa_seq_list)

        return aa_seq

def align_segs_to_seq_impl(seg_infos, aa_seq, idx_seg, mask_vec):
        """Align discontinous segments to the full amino-acid sequence - core implementation."""

        # find all the matching sub-strings w/ overlapping segments allowed
        # NOTE we do not use re.finditer() here since it does not allow overlapping segments
        def _find_all_matches(seq_base, seq_qury):
            idx_base = 0
            idxs_beg = []  # list of starting indices
            while seq_qury in seq_base[idx_base:]:
                idx_beg = idx_base + seq_base[idx_base:].index(seq_qury)
                idx_base = idx_beg + 1
                idxs_beg.append(idx_beg)
            return idxs_beg

        if idx_seg == len(seg_infos):
            return True

        seg_info = seg_infos[idx_seg]
        seg_len = len(seg_info['seq'])
        idxs_resd_beg = _find_all_matches(aa_seq, seg_info['seq'])
        for idx_resd_beg in idxs_resd_beg:
            idx_resd_end = idx_resd_beg + seg_len
            if torch.max(mask_vec[idx_resd_beg:]) == 0:  # do not allow backward alignment
                offset = idx_resd_beg - seg_info['ib']
                mask_vec[idx_resd_beg:idx_resd_end] = 1  # mark as occupied
                is_valid = align_segs_to_seq_impl(seg_infos, aa_seq, idx_seg + 1, mask_vec)
                if is_valid:
                    seg_info['offset'] = offset
                    return True
                mask_vec[idx_resd_beg:idx_resd_end] = 0  # mark as unoccupied

        return False

def align_segs_to_seq(seg_infos, aa_seq):
        """Align discontinous segments to the full amino-acid sequence."""

        mask_vec = torch.zeros(len(aa_seq), dtype=torch.int8)
        is_valid = align_segs_to_seq_impl(seg_infos, aa_seq, 0, mask_vec)
        if not is_valid:
            raise PdbParseError('NO_VALID_OFFSET')
        
def get_atoms( chain, aa_seq):  # pylint: disable=too-many-locals
        """Get atom coordinates & masks from the specified chain."""

        # get discontinous segments and align them to the full amino-acid sequence
        seg_infos = get_segs(chain)
        if aa_seq is None:
            aa_seq = build_seq_from_segs(seg_infos)

        align_segs_to_seq(seg_infos, aa_seq)

        # obtain atom coordinates & masks
        seq_len = len(aa_seq)
        idx_resd_prev = -9999
        ins_code_prev = ' '  # no insertion
        n_resds_ins = 0  # number of inserted residues
        atom_cords = torch.zeros((seq_len, N_ATOMS_PER_RESD, 3), dtype=torch.float32)
        atom_masks = torch.zeros((seq_len, N_ATOMS_PER_RESD), dtype=torch.int8)
        for residue in chain:
            # skip hetero-residues, and obtain the residue's index
            het_flag, idx_resd, ins_code = residue.get_id()
            if het_flag.strip() != '':
                continue  # skip hetero-residues
            if (idx_resd == idx_resd_prev) and (ins_code != ins_code_prev):
                n_resds_ins += 1
            idx_resd_prev = idx_resd
            ins_code_prev = ins_code

            # determine the offset for the current segment
            seg_infos_sel = [x for x in seg_infos if x['ib'] <= idx_resd + n_resds_ins < x['ie']]
            if len(seg_infos_sel) != 1:
                raise PdbParseError('MULTIPLE_MATCHED_SEGMENTS')
            offset = seg_infos_sel[0]['offset']

            # update atom coordinates & masks
            atom_names = ATOM_NAMES_PER_RESD[residue.get_resname()]
            for idx_atom, atom_name in enumerate(atom_names):
                if residue.has_id(atom_name):
                    atom_cords[idx_resd + n_resds_ins + offset, idx_atom] = \
                        torch.from_numpy(residue[atom_name].get_coord())
                    atom_masks[idx_resd + n_resds_ins + offset, idx_atom] = 1

        return aa_seq, atom_cords, atom_masks

def get_plddt( path, aa_seq, chain_id):
        """Get pLDDT scores (per-residue & overall).

        Notes:
        * The overall pLDDT is computed all chains in the PDB file, rather than a single chain.
        """

        # initialization
        n_resds = len(aa_seq)

        # obtain line strings from the PDB file
        line_strs = get_line_strs(path)

        # get pLDDT scores (per-residue & overall)
        plddt_val_full = None
        plddt_vec_resd = torch.zeros((n_resds), dtype=torch.float32)
        for line_str in line_strs:
            if line_str.startswith('REMARK 250 Predicted lDDT-Ca score:'):
                plddt_val_full = torch.tensor([float(line_str.split()[-1])], dtype=torch.float32)
                continue
            if not line_str.startswith('ATOM'):
                continue
            if (chain_id is not None) and (line_str[21] != chain_id):
                continue
            if line_str[12:16].strip() == 'CA':
                idx_resd = int(line_str[22:26]) - 1
                if (idx_resd < 0) or (idx_resd >= n_resds):
                    raise PdbParseError('INVALID_RESIDUE_INDEX')
                plddt_vec_resd[idx_resd] = float(line_str[60:66])
        if plddt_val_full is None:
            plddt_val_full = torch.mean(plddt_vec_resd).reshape(-1)

        # re-scale pLDDT scores to [0, 1]
        if plddt_val_full.item() > 1.0:
            plddt_val_full /= 100.0
            plddt_vec_resd /= 100.0

        return plddt_vec_resd, plddt_val_full

def cdist(x1, x2=None):
    """Calculate the pairwise distance matrix.

    Args:
    * x1: input tensor of size N x D or B x N x D
    * x2: (optional) input tensor of size M x D or B x M x D

    Returns:
    * dist_tns: pairwise distance of size N x M or B x N x M

    Note:
    * If <x2> is not provided, then pairwise distance will be computed within <x1>.
    * The matrix multiplication approach will not be used to avoid the numerical stability issue.
    """

    # initialization
    x2 = x1 if x2 is None else x2

    # recursively call if the batch dimension is missing
    if (x1.ndim == 2) and (x2.ndim == 2):
        return cdist(x1.unsqueeze(dim=0), x2.unsqueeze(dim=0))[0]

    # validate inputs
    assert (x1.ndim == 3) and (x2.ndim == 3)
    assert (x1.shape[0] == x2.shape[0]) and (x1.shape[2] == x2.shape[2])

    # calculate the pairwise distance matrix
    cntr_tns = torch.mean(torch.cat([x1, x2], dim=1), dim=1, keepdim=True)  # B x 1 x D
    x1_cntr = x1 - cntr_tns  # B x N x D
    x2_cntr = x2 - cntr_tns  # B x M x D
    dist_tns = torch.cdist(x1_cntr, x2_cntr, compute_mode='donot_use_mm_for_euclid_dist')

    return dist_tns

def quat2rota_impl(qr, qx, qy, qz):  # pylint: disable=too-many-locals
    """Convert decomposed quaternion vectors into rotation matrices - core implementation.

    Args:
    * qr: 1st components in quaternion vectors of size N
    * qx: 2nd components in quaternion vectors of size N
    * qy: 3rd components in quaternion vectors of size N
    * qz: 4th components in quaternion vectors of size N

    Returns:
    * rota_tns: rotation matrices of size N x 3 x 3

    Reference:
    * J. Claraco, A tutorial on SE(3) transformation parameterizations and on-manifold optimization.
      Technical report, 2020. - Section 2.4.1
    """

    # calculate intermediate results
    qrr = torch.square(qr)
    qxx = torch.square(qx)
    qyy = torch.square(qy)
    qzz = torch.square(qz)
    qrx = 2 * qr * qx
    qry = 2 * qr * qy
    qrz = 2 * qr * qz
    qxy = 2 * qx * qy
    qxz = 2 * qx * qz
    qyz = 2 * qy * qz

    # calculate each entry in the rotation matrix
    r11 = qrr + qxx - qyy - qzz
    r12 = qxy - qrz
    r13 = qxz + qry
    r21 = qxy + qrz
    r22 = qrr - qxx + qyy - qzz
    r23 = qyz - qrx
    r31 = qxz - qry
    r32 = qyz + qrx
    r33 = qrr - qxx - qyy + qzz

    # stack all the entries into rotation matrices
    rota_tns = torch.stack([
        torch.stack([r11, r12, r13], dim=1),
        torch.stack([r21, r22, r23], dim=1),
        torch.stack([r31, r32, r33], dim=1),
    ], dim=1)

    return rota_tns

def quat2rota_part(quat_mat):
    """Convert partial quaternion vectors into rotation matrices.

    Args:
    * quat_mat: quaternion vectors of size N x 3

    Returns:
    * rota_tns: rotation matrices of size N x 3 x 3
    """

    # obtain normalized quaternion vectors
    norm_vec = torch.sqrt(1.0 + torch.sum(torch.square(quat_mat), dim=1))
    qr = 1.0 / norm_vec
    qx = quat_mat[:, 0] / norm_vec
    qy = quat_mat[:, 1] / norm_vec
    qz = quat_mat[:, 2] / norm_vec

    # convert decomposed quaternion vectors into rotation matrices
    rota_tns = quat2rota_impl(qr, qx, qy, qz)

    return rota_tns

def quat2rota_full(quat_mat):
    """Convert full quaternion vectors into rotation matrices.

    Args:
    * quat_mat: quaternion vectors of size N x 4

    Returns:
    * rota_tns: rotation matrices of size N x 3 x 3
    """

    # configurations
    eps = 1e-6

    # obtain normalized quaternion vectors
    quat_mat_norm = quat_mat / (torch.norm(quat_mat, dim=1, keepdim=True) + eps)
    quat_mat_flip = quat_mat_norm * torch.sign(quat_mat_norm[:, :1] + eps)  # qr: non-negative
    qr, qx, qy, qz = [x.squeeze(dim=1) for x in torch.split(quat_mat_flip, 1, dim=1)]

    # convert decomposed quaternion vectors into rotation matrices
    rota_tns = quat2rota_impl(qr, qx, qy, qz)

    return rota_tns

def quat2rota(quat_mat):
    """Convert full / partial quaternion vectors into rotation matrices.

    Args:
    * quat_mat: quaternion vectors of size N x 4 (full) or N x 3 (part)

    Returns:
    * rota_tns: rotation matrices of size N x 3 x 3
    """

    return quat2rota_full(quat_mat) if quat_mat.shape[1] == 4 else quat2rota_part(quat_mat)

def calc_plnr_angl(x1, x2, x3):
    """Calculate the planar angle.

    Args:
    * x1: 1st atom's 3D coordinate of size 3
    * x2: 2nd atom's 3D coordinate of size 3
    * x3: 3rd atom's 3D coordinate of size 3

    Returns:
    * rad: planar angle (in radian, ranging from 0 to pi)
    """

    eps = 1e-6
    a1 = x1 - x2
    a2 = x3 - x2
    rad = torch.arccos(torch.clip(
        torch.inner(a1, a2) / (torch.norm(a1) * torch.norm(a2) + eps), -1.0, 1.0))

    return rad


def calc_plnr_angl_batch(cord_tns):
    """Calculate planar angles in the batch mode.

    Args:
    * cord_tns: 3D coordinates of size N x 3 x 3

    Returns:
    * rad_vec: planar angles (in radian, ranging from 0 to pi) of size N
    """

    eps = 1e-6
    x1, x2, x3 = [x.squeeze(dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    a1 = x1 - x2
    a2 = x3 - x2
    n1 = torch.norm(a1, dim=1)
    n2 = torch.norm(a2, dim=1)
    rad_vec = torch.arccos(torch.clip(torch.sum(a1 * a2, dim=1) / (n1 * n2 + eps), -1.0, 1.0))

    return rad_vec


def calc_dihd_angl(x1, x2, x3, x4):
    """Calculate the dihedral angle.

    Args:
    * x1: 1st atom's 3D coordinate of size 3
    * x2: 2nd atom's 3D coordinate of size 3
    * x3: 3rd atom's 3D coordinate of size 3
    * x4: 4th atom's 3D coordinate of size 3

    Returns:
    * rad: dihedral angle (in radian, ranging from -pi to pi)
    """

    eps = 1e-6
    a1 = x2 - x1
    a2 = x3 - x2
    a3 = x4 - x3
    v1 = torch.cross(a1, a2)
    v1 = v1 / (torch.norm(v1) + eps)
    v2 = torch.cross(a2, a3)
    v2 = v2 / (torch.norm(v2) + eps)
    sign = torch.sign(torch.inner(v1, a3))
    rad = torch.arccos(torch.clip(
        torch.inner(v1, v2) / (torch.norm(v1) * torch.norm(v2) + eps), -1.0, 1.0))
    if sign != 0:
        rad *= sign

    return rad


def calc_dihd_angl_batch(cord_tns):
    """Calculate dihedral angles in the batch mode.

    Args:
    * cord_tns: 3D coordinates of size N x 4 x 3

    Returns:
    * rad_vec: dihedral angles (in radian, ranging from -pi to pi) of size N
    """

    eps = 1e-6
    x1, x2, x3, x4 = [x.squeeze(dim=1) for x in torch.split(cord_tns, 1, dim=1)]
    a1 = x2 - x1
    a2 = x3 - x2
    a3 = x4 - x3
    v1 = torch.cross(a1, a2, dim=1)
    v1 = v1 / (torch.norm(v1, dim=1, keepdim=True) + eps)  # is this necessary?
    v2 = torch.cross(a2, a3, dim=1)
    v2 = v2 / (torch.norm(v2, dim=1, keepdim=True) + eps)  # is this necessary?
    n1 = torch.norm(v1, dim=1)
    n2 = torch.norm(v2, dim=1)
    sign = torch.sign(torch.sum(v1 * a3, dim=1))
    sign[sign == 0.0] = 1.0  # to avoid multiplication with zero
    rad_vec = sign * \
        torch.arccos(torch.clip(torch.sum(v1 * v2, dim=1) / (n1 * n2 + eps), -1.0, 1.0))

    return rad_vec

def get_idx(data):
    """
    Determine the initial and final indices of the continuous segment containing the value 1.
    """
    data = torch.nonzero(data==1).squeeze()
    segments = []
    
    if data.numpy().tolist() == []:
        """
        In the absence of the value 1.
        """
        return []
    try:
        current_segment = [data[0].item()]
    except:
        """
        In the case where only a single instance of the value 1 is present.
        """
        current_segment = [data.item()]
        return [current_segment]

    for i in range(1, len(data)):
        # Verify whether the current element is contiguous with the preceding element.
        if data[i] == data[i - 1] + 1:
            current_segment.append(data[i].item())
        else:
            segments.append(current_segment)
            current_segment = [data[i].item()]

    segments.append(current_segment)  
    return segments

def Seq_idx(idx, seq):
    """
    Concatenate the sequence.
    """
    seq_com = ''
    for i in range(len(idx)):
        idx_beg = min(idx[i])
        idx_end = max(idx[i]) + 1
        seq_tem = seq[idx_beg:idx_end]
        seq_com += seq_tem
    return seq_com

class PdbParseError(Exception):
    """Exceptions raised when parsing a PDB file w/ BioPython."""

class PdbParser():
    """Parser for PDB files."""

    def init(self):
        """Constructor function."""

    def load(
            pdb_fpath, aa_seq=None, fas_fpath=None,
            model_id=None, chain_id=None, has_plddt=False,
        ):  # pylint: disable=too-many-arguments
        """Load a protein structure from the PDB file.

        Args:
        * pdb_fpath: path to the PDB file
        * aa_seq: (optional) reference amino-acid sequence
        * fas_fpath: (optional) path to the reference FASTA file
        * model_id: (optional) model ID
        * chain_id: (optional) chain ID
        * has_plddt: (optional) whether the PDB file contains per-residue & full-chain pLDDT scores

        Returns:
        * aa_seq: amino-acid sequence
        * atom_cords: per-atom 3D coordinates of size L x M x 3
        * atom_masks: per-atom 3D coordinates' validness masks of size L x M
        * meta_data: dict of meta-data stored in the PDB file
        * error_msg: error message raised when parsing the PDB file

        Note:
        * The GZ-compressed PDB file can be provided with a suffix of ".gz".
        * The amino-acid sequence is determined in the following order:
          a) parsed from the FASTA file
          b) parsed from SEQRES records in the PDB file
          c) parsed from ATOM records in the PDB file
        * If <chain_id> is not provided, then the first chain will be returned. The specific order
          is defined by the <BioPython> package. If <chain_id> is provided, then the first model
          with the specified chain ID will be returned.
        """

        # suppress all the warnings raised by <BioPython>
        warnings.simplefilter('ignore', BiopythonWarning)

        # show the greeting message
        logging.debug('parsing the PDB file: %s (chain ID: <%s>)', pdb_fpath, chain_id)
        if fas_fpath is not None:
            logging.debug('FASTA file provided: %s', fas_fpath)

        # attempt to parse the PDB file
        try:
            # check inputs
            if not os.path.exists(pdb_fpath):
                raise PdbParseError('PDB_FILE_NOT_FOUND')
            if not (pdb_fpath.endswith('.pdb') or pdb_fpath.endswith('.gz')):
                raise PdbParseError('PDB_FILE_FORMAT_NOT_SUPPORTED')
            if (fas_fpath is not None) and (not os.path.exists(fas_fpath)):
                raise PdbParseError('FASTA_FILE_NOT_FOUND')

            # obtain the amino-acid sequence (could be None)
            if aa_seq is None:
                if fas_fpath is not None:
                    _, aa_seq = parse_fas_file(fas_fpath)
                else:  # then the amino-acid sequence must be parsed from the PDB file
                    aa_seq = get_aa_seq_from_seqres(pdb_fpath, chain_id)

            # parse the PDB file w/ biopython
            structure = get_structure(pdb_fpath)

            # obtain meta-data from the structure
            meta_data = {
                'id': structure.header['idcode'],
                'date': structure.header['release_date'],
                'reso': structure.header['resolution'],
                'mthd': structure.header['structure_method'],
            }

            # find the first chain matching the model/chain ID
            chain = get_chain(structure, model_id, chain_id)
            
            # obtain atom coordinates & validness masks
            aa_seq, atom_cords, atom_masks = get_atoms(chain, aa_seq)

            # obtain pLDDT scores (per-residue & full-chain)
            if has_plddt:
                meta_data['plddt-r'], meta_data['plddt-c'] = \
                    get_plddt(pdb_fpath, aa_seq, chain_id)

            # set the error message to None
            error_msg = None
        except PdbParseError as error:
            aa_seq, atom_cords, atom_masks, meta_data, error_msg = None, None, None, None, error

        return aa_seq, atom_cords, atom_masks, meta_data, error_msg

class ProtStruct():  # pylint: disable=too-many-instance-attributes
    """Protein structures (3D coordinates, local frames, and torsion angles)."""

    def __init__(self):
        """Constructor function."""

        # basic information
        self.aa_seq = None
        self.cord_tns = None  # L x M x 3 (full-atom 3D coordinates)
        self.cmsk_mat = None  # L x M

        # additional informations
        self.fram_tns_bb = None  # L x 1 x 4 x 3 (backbone local frames)
        self.fmsk_mat_bb = None  # L x 1
        self.fram_tns_sc = None  # L x K x 4 x 3 (side-chain local frames)
        self.fmsk_mat_sc = None  # L x K
        self.angl_tns = None  # L x K x 2 (torsion angles)
        self.amsk_mat = None  # L x K
        self.cmsk_mat_vld = None  # L x M (only depends on the sequence)
        self.cmsk_mat_sym = None  # L x M (only depends on the sequence)
        self.amsk_mat_sym = None  # L x K (only depends on the sequence)

        # data availability indicators
        self.has_fram_bb = False  # whether backbone local frames are ready
        self.has_fram_sc = False  # whether side-chain local frames are ready
        self.has_angl = False  # whether torsion angles are ready
        self.has_mask = False  # whether valid/symmetric-or-not masks are ready

        # auxiliary constants
        self.eps = 1e-6


    def init_from_file(self, fas_fpath, pdb_fpath, chain_id=None):
        """Initialize the protein structure from FASTA & PDB files.

        Args:
        * fas_fpath: path to the FASTA file
        * pdb_fpath: path to the PDB file
        * chain_id: (optional) chain ID

        Returns: n/a
        """

        # initialization
        self.aa_seq, self.cord_tns, self.cmsk_mat, _, error_msg = \
            PdbParser.load(pdb_fpath, fas_fpath=fas_fpath, chain_id=chain_id)
        assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'

        # update data availability indicators
        self.has_fram_bb = False
        self.has_fram_sc = False
        self.has_angl = False
        self.has_mask = False


    def init_from_cord(self, aa_seq, cord_tns, cmsk_mat):
        """Initialize the protein structure from 3D coordinates.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M

        Returns: n/a
        """

        # initialization
        self.aa_seq = aa_seq
        self.cord_tns = cord_tns
        self.cmsk_mat = cmsk_mat

        # update data availability indicators
        self.has_fram_bb = False
        self.has_fram_sc = False
        self.has_angl = False
        self.has_mask = False


    def init_from_param(self, aa_seq, params, converter, atom_set='fa'):
        """Initialize the protein structure from QTA parameters.

        Args:
        * aa_seq: amino-acid sequence
        * params: dict of QTA parameters (must contain 'quat', 'trsl', and 'angl')
          > quat: per-residue quaternion vectors of size L x 4
          > trsl: per-residue translation vectors size size L x 3
          > angl: per-residue torsion angles of size L x K x 2
        * converter: <ProtConverter> object for coord-frame-angle conversions
        * atom_set: (optional) which atoms to be reconstructed (choices: 'ca' / 'b3' / 'b4' / 'fa')

        Returns: n/a

        Note:
        * We do not take any validness masks as inputs, since all the predicted QTA parameters are
            assumed to be valid.
        """

        # initialization
        n_resds = len(aa_seq)
        device = params['quat'].device

        # convert QTA parameters into backbone local frames & torsion angles
        self.aa_seq = aa_seq
        self.fram_tns_bb = torch.cat(
            [quat2rota(params['quat']), params['trsl'].unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
        self.fmsk_mat_bb = torch.ones((n_resds, 1), dtype=torch.int8, device=device)
        self.angl_tns = nn.functional.normalize(params['angl'], p=2.0, dim=2)
        self.amsk_mat = torch.ones((n_resds, N_ANGLS_PER_RESD), dtype=torch.int8, device=device)

        # reconstruct per-atom 3D coordinates
        assert atom_set in ['ca', 'b3', 'b4', 'fa'], f'unrecognized atom set: {atom_set}'
        self.cord_tns, self.cmsk_mat = converter.fa2cord(
            self.aa_seq, self.fram_tns_bb, self.fmsk_mat_bb, self.angl_tns, self.amsk_mat, atom_set)

        # update data availability indicators
        self.has_fram_bb = True
        self.has_fram_sc = False
        self.has_angl = True
        self.has_mask = False


    def build_fram_n_angl(self, converter, build_sc=False):
        """Build backbone and/or side-chain local frames and torison angles from 3D coordinates.

        Args:
        * converter: <ProtConverter> object for coord-frame-angle conversions
        * build_sc: (optional) whether to build side-chain local frames

        Returns: n/a
        """

        # build backbone local frames & torsion angles
        if not (self.has_fram_bb and self.has_angl):
            self.fram_tns_bb, self.fmsk_mat_bb, self.angl_tns, self.amsk_mat = \
                converter.cord2fa(self.aa_seq, self.cord_tns, self.cmsk_mat)
            self.has_fram_bb = True
            self.has_angl = True

        # (optional) build side-chain local frames
        if build_sc and not self.has_fram_sc:
            self.fram_tns_sc, self.fmsk_mat_sc = \
                converter.cord2fram(self.aa_seq, self.cord_tns, self.cmsk_mat, fram_set='sc')
            self.has_fram_sc = True


    def build_mask(self):
        """Build valid/symmetric-or-not masks for atoms & torsion angles."""

        assert (self.aa_seq is not None) and (self.cord_tns is not None)

        self.cmsk_mat_vld = self.get_cmsk_vld(self.aa_seq, self.cord_tns.device)
        self.cmsk_mat_sym = self.get_cmsk_sym(self.aa_seq, self.cord_tns.device)
        self.amsk_mat_sym = self.get_amsk_sym(self.aa_seq, self.cord_tns.device)
        self.has_mask = True


    def build_alt_pose(self, converter):
        """Build the alternative pose by flipping all the symmetric torsion angles.

        Args:
        * converter: <ProtConverter> object for coord-frame-angle conversions

        Returns:
        * cord_tns: alternative 3D coordinates of size L x M x 3
        * angl_tns: alternative torsion angles of size L x K x 2
        * fram_tns_sc: alternative side-chain local frames of size L x K x 4 x 3
        """

        # initialization
        device = self.cord_tns.device

        # flip all the symmetric torsion angles
        amsk_mat_sym = self.get_amsk_sym(self.aa_seq, device)
        angl_tns = (1 - 2 * amsk_mat_sym).unsqueeze(dim=2) * self.angl_tns

        # reconstruct per-atom 3D coordinates w/ flipped symmetric torsion angles
        cord_tns, _ = converter.fa2cord(
            self.aa_seq, self.fram_tns_bb, self.fmsk_mat_bb, angl_tns, self.amsk_mat, atom_set='fa')

        # build side-chain local frames
        fram_tns_sc, _ = converter.cord2fram(self.aa_seq, cord_tns, self.cmsk_mat, fram_set='sc')

        return cord_tns, angl_tns, fram_tns_sc


    def rename_sym_atoms(self, cord_tns_ref, cmsk_mat_ref, converter):  # pylint: disable=too-many-locals
        """Rename symmetric ground-truth atoms.

        Args:
        * cord_tns_ref: reference protein structure's 3D coordinates of size L x M x 3
        * cmsk_mat_ref: reference protein structure's 3D coordinates' validness masks of size L x M
        * converter: <ProtConverter> object for coord-frame-angle conversions

        Returns: n/a

        Note:
        * There is at most one symmetric rigid-body group in each residue. Hence, we only need to
            maintain a single swap-or-not indicator per residue.
        """

        # initialization
        device = self.cord_tns.device
        n_resds = self.cord_tns.shape[0]

        # build the alternative pose of current structure
        cord_tns_alt, angl_tns_alt, fram_tns_sc_alt = self.build_alt_pose(converter)

        # calculate pairwise distance matrices
        dist_mat_ref = cdist(cord_tns_ref.view(-1, 3))  # (L x M) x (L x M)
        dist_mat_bsc = cdist(self.cord_tns.view(-1, 3))
        dist_mat_alt = cdist(cord_tns_alt.view(-1, 3))

        # get per-atom symmetric-or-not masks
        cmsk_mat_sym = self.get_cmsk_sym(self.aa_seq, device)

        # calculate dRMSD for basic & alternative query structures
        cmsk_mat_cmb = self.cmsk_mat * cmsk_mat_ref
        dmsk_mat = (cmsk_mat_cmb * cmsk_mat_sym).view(-1, 1) * \
            (cmsk_mat_cmb * (1 - cmsk_mat_sym)).view(1, -1)
        drmsd_bsc = torch.sum(
            (dmsk_mat * torch.abs(dist_mat_bsc - dist_mat_ref)).view(n_resds, -1), dim=1)
        drmsd_alt = torch.sum(
            (dmsk_mat * torch.abs(dist_mat_alt - dist_mat_ref)).view(n_resds, -1), dim=1)

        # rename symmetric ground-truth atoms (and side-chain local frames, if provided)
        rmsk_vec = torch.less(drmsd_bsc, drmsd_alt)
        self.cord_tns = torch.where(rmsk_vec.view(-1, 1, 1), self.cord_tns, cord_tns_alt)
        self.angl_tns = torch.where(rmsk_vec.view(-1, 1, 1), self.angl_tns, angl_tns_alt)
        self.fram_tns_sc = torch.where(
            rmsk_vec.view(-1, 1, 1, 1), self.fram_tns_sc, fram_tns_sc_alt)


    def summarize(self):
        """Summarize the data availability of various aspect in the protein structure.

        Args: n/a

        Returns: n/a
        """

        # data availability
        logging.info('backbone local frames: %s', self.has_fram_bb)
        logging.info('side-chain local frames: %s', self.has_fram_sc)
        logging.info('torsion angles: %s', self.has_angl)
        logging.info('atom/angle masks: %s', self.has_mask)

        # 3D coordinates
        logging.info('aa_seq: %s', self.aa_seq)
        logging.info('cord_tns: %s / %s', self.cord_tns.shape, self.cord_tns.dtype)
        logging.info('cmsk_mat: %s / %s', self.cmsk_mat.shape, self.cmsk_mat.dtype)

        # backbone local frames
        if self.has_fram_bb:
            logging.info('fram_tns_bb: %s / %s', self.fram_tns_bb.shape, self.fram_tns_bb.dtype)
            logging.info('fmsk_mat_bb: %s / %s', self.fmsk_mat_bb.shape, self.fmsk_mat_bb.dtype)

        # side-chain local frames
        if self.has_fram_sc:
            logging.info('fram_tns_sc: %s / %s', self.fram_tns_sc.shape, self.fram_tns_sc.dtype)
            logging.info('fmsk_mat_sc: %s / %s', self.fmsk_mat_sc.shape, self.fmsk_mat_sc.dtype)

        # torsion angles
        if self.has_angl:
            logging.info('angl_tns: %s / %s', self.angl_tns.shape, self.angl_tns.dtype)
            logging.info('amsk_mat: %s / %s', self.amsk_mat.shape, self.amsk_mat.dtype)

        # atom/angle masks
        if self.has_mask:
            logging.info('cmsk_mat_vld: %s / %s', self.cmsk_mat_vld.shape, self.cmsk_mat_vld.dtype)
            logging.info('cmsk_mat_sym: %s / %s', self.cmsk_mat_sym.shape, self.cmsk_mat_sym.dtype)
            logging.info('amsk_mat_sym: %s / %s', self.amsk_mat_sym.shape, self.amsk_mat_sym.dtype)


    @classmethod
    def get_atoms(cls, aa_seq, atom_tns_all, atom_names_sel):  # pylint: disable=too-many-locals
        """Get 3D coordinates or validness masks for selected atom(s).

        Args:
        * aa_seq: amino-acid sequence
        * atom_tns_all: full-atom 3D coordinates (L x M x 3) or validness masks (L x M)
        * atom_names_sel: list of selected atom names of length M'

        Returns:
        * atom_tns_sel: selected 3D coordinates (L x M' x 3) or validness masks (L x M')

        Note:
        * If only one atom name if provided, then the per-atom dimension is squeezed.
        """

        # use the specifically optimized implementation if only CA atoms are needed
        if atom_names_sel == ['CA']:
            return atom_tns_all[:, 1]  # CA atom is always the 2nd atom, as defined in constants.py

        # initialization
        device = atom_tns_all.device
        n_atoms = len(atom_names_sel)

        # build the indexing tensor for selected atom(s)
        idxs_vec_dict = {}  # atom indices
        msks_vec_dict = {}  # atom indices' validness masks
        for resd_name in RESD_NAMES_3C:
            atom_names_all = ATOM_NAMES_PER_RESD[resd_name]
            idxs_vec_np = np.zeros((n_atoms), dtype=np.int64)
            msks_vec_np = np.zeros((n_atoms), dtype=np.int8)
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                if atom_name_sel in atom_names_all:  # otherwise, keep zeros unchanged
                    idxs_vec_np[idx_atom_sel] = atom_names_all.index(atom_name_sel)
                    msks_vec_np[idx_atom_sel] = 1
            idxs_vec_dict[resd_name] = idxs_vec_np
            msks_vec_dict[resd_name] = msks_vec_np

        # determine the overall indexing tensor based on the amino-acid sequence
        resd_names_3c = [RESD_MAP_1TO3[resd_name_1c] for resd_name_1c in aa_seq]
        idxs_mat_full_np = np.stack([idxs_vec_dict[x] for x in resd_names_3c], axis=0)
        msks_mat_full_np = np.stack([msks_vec_dict[x] for x in resd_names_3c], axis=0)
        idxs_mat_full = torch.tensor(idxs_mat_full_np, dtype=torch.int64, device=device)  # L x M'
        msks_mat_full = torch.tensor(msks_mat_full_np, dtype=torch.int64, device=device)  # L x M'

        # get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s)
        if atom_tns_all.ndim == 2:
            atom_tns_sel = msks_mat_full * torch.gather(atom_tns_all, 1, idxs_mat_full)
        else:
            n_dims_addi = atom_tns_all.shape[-1]
            atom_tns_sel = msks_mat_full.unsqueeze(dim=2) * torch.gather(
                atom_tns_all, 1, idxs_mat_full.unsqueeze(dim=2).repeat(1, 1, n_dims_addi))

        # squeeze the dimension if only one atom is selected
        if n_atoms == 1:
            atom_tns_sel.squeeze_(dim=1)

        return atom_tns_sel


    @classmethod
    def get_cmsk_vld(cls, aa_seq, device, bb_only=False, atom_set='fa'):
        """Get per-atom valid-or-not masks from amino-acid sequence.

        Args:
        * aa_seq: amino-acid sequence
        * device: computational device to place <cmsk_mat>
        * bb_only: (optional & deprecated) whether to only preserve backbone atoms (N - CA - C)
        * atom_set: (optional) valid atom set (choices: 'ca' / 'b3' / 'b4' / 'fa')

        Returns:
        * cmsk_mat: per-atom valid-or-not masks of size L x M
        """

        # initialization
        n_resds = len(aa_seq)

        # special case: CA atoms only
        if atom_set == 'ca':
            cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8, device=device)
            cmsk_mat[:, 1] = 1
            return cmsk_mat

        # build validness masks for each residue type, based on the specified atom set
        cmsk_vec_dict = {}
        for resd_name in RESD_NAMES_1C:
            # determine atom names
            atom_names_all = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            if bb_only or (atom_set == 'b3'):
                atom_names_vld = ['N', 'CA', 'C']
            elif atom_set == 'b4':
                atom_names_vld = ['N', 'CA', 'C', 'O']
            elif atom_set == 'fa':
                atom_names_vld = atom_names_all
            else:
                raise ValueError(f'unrecognized valid atom set: {atom_set}')

            # build validness masks
            cmsk_vec = torch.zeros(N_ATOMS_PER_RESD, dtype=torch.int8)
            for atom_name in atom_names_vld:
                idx_atom = atom_names_all.index(atom_name)
                cmsk_vec[idx_atom] = 1
            cmsk_vec_dict[resd_name] = cmsk_vec

        # generate per-atom valid-or-not masks
        if 'X' in aa_seq:
            aa_seq = aa_seq.replace('X', 'T')
        cmsk_mat = torch.stack([cmsk_vec_dict[x] for x in aa_seq], dim=0).to(device)

        return cmsk_mat


    @classmethod
    def get_cmsk_sym(cls, aa_seq, device):  # pylint: disable=too-many-locals
        """Get per-atom symmetric-or-not masks from amino-acid sequence.

        Args:
        * aa_seq: amino-acid sequence
        * device: computational device to place <cmsk_mat>

        Returns:
        * cmsk_mat: per-atom symmetric-or-not masks of size L x M
        """

        # initialization
        n_resds = len(aa_seq)

        # generate per-atom symmetric-or-not masks
        cmsk_mat = torch.zeros((n_resds, N_ATOMS_PER_RESD), dtype=torch.int8, device=device)
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            atom_names_all = ATOM_NAMES_PER_RESD[resd_name_3c]
            atom_infos = ATOM_INFOS_PER_RESD[resd_name_3c]
            angl_infos = ANGL_INFOS_PER_RESD[resd_name_3c]
            for idx_angl, (_, is_symm, _) in enumerate(angl_infos):
                if is_symm:
                    atom_names_sel = [x[0] for x in atom_infos if x[1] == idx_angl + 3]
                    for atom_name in atom_names_sel:
                        idx_atom = atom_names_all.index(atom_name)
                        cmsk_mat[idx_resd, idx_atom] = 1

        return cmsk_mat


    @classmethod
    def get_amsk_sym(cls, aa_seq, device):
        """Get per-angle symmetric-or-not masks from amino-acid sequence.

        Args:
        * aa_seq: amino-acid sequence
        * device: computational device to place <amsk_mat>

        Returns:
        * amsk_mat: per-angle symmetric-or-not masks of size L x K
        """

        # initialization
        n_resds = len(aa_seq)

        # generate per-atom symmetric-or-not masks
        amsk_mat = torch.zeros((n_resds, N_ANGLS_PER_RESD), dtype=torch.int8, device=device)
        for idx_resd, resd_name_1c in enumerate(aa_seq):
            resd_name_3c = RESD_MAP_1TO3[resd_name_1c]
            angl_infos = ANGL_INFOS_PER_RESD[resd_name_3c]
            for idx_angl, (_, is_symm, _) in enumerate(angl_infos):
                if is_symm:
                    amsk_mat[idx_resd, idx_angl + 2] = 1

        return amsk_mat

def calc_ppi_sites(pdb_path, dist_thres=10):
    """Calculate PPI sites"""
    ppi_data = {}
    chain_list = get_chains_from_pdb(pdb_path)
    seq_one, cord_one, mask_one, _, _ = PdbParser.load(pdb_path, chain_id=chain_list[0])
    seq_two, cord_two, mask_two, _, _ = PdbParser.load(pdb_path, chain_id=chain_list[1])
    
    seq_three, cord_three, mask_three, _, _ = PdbParser.load(pdb_path, chain_id=chain_list[2])
    cord_mat_one = ProtStruct.get_atoms(seq_one, cord_one, ['CA'])
    cmsk_vec_one = ProtStruct.get_atoms(seq_one, mask_one, ['CA'])
    cord_mat_two = ProtStruct.get_atoms(seq_two, cord_two, ['CA'])
    cmsk_vec_two = ProtStruct.get_atoms(seq_two, mask_two, ['CA'])
    cord_mat_three = ProtStruct.get_atoms(seq_three, cord_three, ['CA'])
    cmsk_vec_three = ProtStruct.get_atoms(seq_three, mask_three, ['CA'])
    
    dist_mat_13 = cdist(cord_mat_one, cord_mat_three)
    dist_max_13 = torch.max(dist_mat_13)
    dist_mat_13 += dist_max_13 * (1 - torch.outer(cmsk_vec_one, cmsk_vec_three))
    dist_vec_one = torch.min(dist_mat_13, dim=1)[0]  # minimal distance to the secondary chain
    dist_vec_13 = torch.min(dist_mat_13, dim=0)[0]  # minimal distance to the primary chain
    ppi_data[chain_list[0]] = torch.lt(dist_vec_one, dist_thres).to(torch.int8)
    ppi_data_13 = torch.lt(dist_vec_13, dist_thres).to(torch.int8)

    dist_mat_23 = cdist(cord_mat_two, cord_mat_three)
    dist_max_23 = torch.max(dist_mat_23)
    dist_mat_23 += dist_max_23 * (1 - torch.outer(cmsk_vec_two, cmsk_vec_three))
    dist_vec_two = torch.min(dist_mat_23, dim=1)[0]  # minimal distance to the secondary chain
    dist_vec_23 = torch.min(dist_mat_23, dim=0)[0]  # minimal distance to the primary chain
    ppi_data[chain_list[1]] = torch.lt(dist_vec_two, dist_thres).to(torch.int8)
    ppi_data_23 = torch.lt(dist_vec_23, dist_thres).to(torch.int8)

    ppi_data[chain_list[2]] = ppi_data_13 | ppi_data_23
    
    idx_one = get_idx(ppi_data[chain_list[0]])
    idx_two = get_idx(ppi_data[chain_list[1]])
    idx_three = get_idx(ppi_data[chain_list[2]])

    seq_idx_one = Seq_idx(idx_one, seq_one)
    seq_idx_two = Seq_idx(idx_two, seq_two)
    seq_idx_three = Seq_idx(idx_three, seq_three)
    
    idx_data = {}
    idx_data[chain_list[0]] = idx_one
    idx_data[chain_list[1]] = idx_two
    idx_data[chain_list[2]] = idx_three

    seq_data = {}
    seq_data[chain_list[0]] = seq_idx_one
    seq_data[chain_list[1]] = seq_idx_two
    seq_data[chain_list[2]] = seq_idx_three


    return ppi_data, idx_data, seq_data

def calc_ppi_sites_nano(pdb_path, dist_thres=10):
    """Calculate PPI sites for Nano antibody"""
    ppi_data = {}
    chain_list = get_chains_from_pdb(pdb_path)
    print(chain_list)
    if len(chain_list) == 2:
        seq_one, cord_one, mask_one, _, _ = PdbParser.load(pdb_path, chain_id=chain_list[0])
        
        seq_three, cord_three, mask_three, _, _ = PdbParser.load(pdb_path, chain_id=chain_list[1])
        cord_mat_one = ProtStruct.get_atoms(seq_one, cord_one, ['CA'])
        cmsk_vec_one = ProtStruct.get_atoms(seq_one, mask_one, ['CA'])
        cord_mat_three = ProtStruct.get_atoms(seq_three, cord_three, ['CA'])
        cmsk_vec_three = ProtStruct.get_atoms(seq_three, mask_three, ['CA'])
        
        dist_mat_13 = cdist(cord_mat_one, cord_mat_three)
        dist_max_13 = torch.max(dist_mat_13)
        dist_mat_13 += dist_max_13 * (1 - torch.outer(cmsk_vec_one, cmsk_vec_three))
        dist_vec_one = torch.min(dist_mat_13, dim=1)[0]  # minimal distance to the secondary chain
        dist_vec_13 = torch.min(dist_mat_13, dim=0)[0]  # minimal distance to the primary chain
        ppi_data[chain_list[0]] = torch.lt(dist_vec_one, dist_thres).to(torch.int8)
        ppi_data_13 = torch.lt(dist_vec_13, dist_thres).to(torch.int8)


        ppi_data[chain_list[1]] = ppi_data_13
        
        idx_one = get_idx(ppi_data[chain_list[0]])
        idx_three = get_idx(ppi_data[chain_list[1]])

        seq_idx_one = Seq_idx(idx_one, seq_one)
        seq_idx_three = Seq_idx(idx_three, seq_three)
        
        idx_data = {}
        idx_data[chain_list[0]] = idx_one
        idx_data[chain_list[1]] = idx_three

        seq_data = {}
        seq_data[chain_list[0]] = seq_idx_one
        seq_data[chain_list[1]] = seq_idx_three


        return ppi_data, idx_data, seq_data
    else:
        return None, None, None

def get_ppi_sites(PDB_Path):
    Data_ = {}
    problem_path = []
    num = 0
    for root, dirs, files in os.walk(PDB_Path):
        for file in tqdm(files):
                try:
                    if file[-3:] == 'pdb':
                        path = os.path.join(PDB_Path, file)
                        name = file.replace('.pdb', '')
                        try:
                            ppi_data, idx_data, seq_data = calc_ppi_sites(path)
                            num += 1
                        except:
                            ppi_data, idx_data, seq_data = calc_ppi_sites_nano(path)
                        
                        Data_[name] = {}
                        Data_[name]['idx_seq'] = ppi_data
                        Data_[name]['idx_ppi'] = idx_data
                        Data_[name]['seq_ppi'] = seq_data

                except Exception as e:
                    print(e)
                    problem_path.append(file)
                    print(file)
        
    pickle.dump(Data_, open('./data/atom/chai_ppi.pkl', 'wb'))
