"""Protein-related utility functions."""

import os
import gzip
from collections import OrderedDict
from collections import defaultdict

import torch


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


def parse_fas_file_mult(path, is_ordered=True):
    """Parse the FASTA file containing multiple chains.

    Args:
    * path: path to the FASTA file (can be GZIP-compressed)
    * is_ordered: (optional) whether the output dict is ordered as in the FASTA file

    Returns:
    * aa_seq_dict: (ordered) dict of (ID, sequence) pairs
    """

    # parse all the lines in the FASTA file
    assert os.path.exists(path), f'FASTA file does not exist: {path}'
    if not path.endswith('.gz'):
        with open(path, 'r', encoding='UTF-8') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]
    else:
        with gzip.open(path, 'rt') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]

    # build an ordered dict of (ID, sequence) pairs
    key_last = None
    aa_seq_dict = OrderedDict() if is_ordered else {}
    for i_line in i_lines:
        if i_line.startswith('>'):
            key_last = i_line[1:]
            aa_seq_dict[key_last] = ''
        else:
            assert key_last is not None, f'failed to get the protein ID in {path}'
            aa_seq_dict[key_last] += i_line

    return aa_seq_dict


def export_fas_file(prot_id, aa_seq, path):
    """Export the amino-acid sequence to a FASTA file.

    Args:
    * prot_id: protein ID (as in the commentary line)
    * aa_seq: amino-acid sequence
    * path: path to the FASTA file

    Returns: n/a
    """

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w', encoding='UTF-8') as o_file:
        o_file.write(f'>{prot_id}\n{aa_seq}\n')


def export_fas_file_mult(aa_seq_dict, path):
    """Export amino-acid sequences of multiple chains to a FASTA file.

    Args:
    * aa_seq_dict: ordered dict of (ID, sequence) pairs
    * path: path to the FASTA file

    Returns: n/a
    """

    os.makedirs(os.path.dirname(os.path.realpath(path)), exist_ok=True)
    with open(path, 'w', encoding='UTF-8') as o_file:
        for prot_id, aa_seq in aa_seq_dict.items():
            o_file.write(f'>{prot_id}\n{aa_seq}\n')


def parse_idx_file(path):
    """Parse the plain-text file of HDF5 index (HDF5 file name -> protein IDs).

    Args:
    * path: path to the HDF5 index file

    Returns:
    * prot_dict: dict of HDF5 index (HDF5 file name -> protein IDs)
    """

    prot_dict = {}  # key: protein ID => value: HDF5 file name
    with open(path, 'r', encoding='UTF-8') as i_file:
        for i_line in i_file:
            sub_strs = i_line.split()
            hdf_fname, prot_ids = sub_strs[0], sub_strs[1:]
            prot_dict.update({x: hdf_fname for x in prot_ids})

    return prot_dict


def get_asym_ids(aa_seqs):
    """Get aymmetric IDs that distinguish between chains.

    Args:
    * aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
    * asym_ids: asymmetric IDs of size L (L = \sum_i L_i)
    """

    asym_ids_list = []
    for idx, aa_seq in enumerate(aa_seqs):
        asym_ids_list.append(idx * torch.ones(len(aa_seq), dtype=torch.int64))
    asym_ids = torch.cat(asym_ids_list)

    return asym_ids


def get_enty_ids(aa_seqs):
    """Get entity IDs that distinguish between unique chain sequences.

    Args:
    * aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
    * enty_ids: entity IDs of size L (L = \sum_i L_i)
    """

    enty_ids_list = []
    aa_seqs_uniq = list(set(aa_seqs))
    for aa_seq in aa_seqs:
        enty_id = aa_seqs_uniq.index(aa_seq)
        enty_ids_list.append(enty_id * torch.ones(len(aa_seq), dtype=torch.int64))
    enty_ids = torch.cat(enty_ids_list)

    return enty_ids


def get_symm_ids(aa_seqs):
    """Get symmetric IDs that distinguish between chains of the same sequence.

    Args:
    * aa_seqs: list of amino-acid sequences, each of length L_i

    Returns:
    * symm_ids: symmetric IDs of size L (L = \sum_i L_i)
    """

    symm_ids_list = []
    symm_id_dict = defaultdict(int)
    for aa_seq in aa_seqs:
        symm_id = symm_id_dict[aa_seq]
        symm_ids_list.append(symm_id * torch.ones(len(aa_seq), dtype=torch.int64))
        symm_id_dict[aa_seq] += 1
    symm_ids = torch.cat(symm_ids_list)

    return symm_ids
