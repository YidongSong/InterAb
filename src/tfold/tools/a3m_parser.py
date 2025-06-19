"""Parser for A3M files."""

import os
import re
import string
import random
import subprocess

from Bio import SeqIO

from tfold.utils import get_rand_str
from tfold.utils import get_tmp_dpath


class A3mParser():  # pylint: disable=too-few-public-methods
    """Parser for A3M files."""

    def __init__(self, version='v2'):
        """Constructor function."""

        self.version = version
        self.tmp_dpath = get_tmp_dpath()
        self.curr_dir = os.path.dirname(os.path.realpath(__file__))
        self.bin_fpath = os.path.realpath(os.path.join(self.curr_dir, '../../cpp/bin/msa_sampler'))


    def run(self, path, msa_depth=-1, filt_mthd='unif'):
        """Run the A3M file parser.

        Args:
        * path: path to the A3M file
        * msa_depth: (optional) MSA depth (-1: unlimited)
        * filt_mthd: (optional) MSA filtering method (choices: 'unif' or 'topk' or 'neff')

        Returns:
        * records: list of (description, sequence) tuples
        """

        if self.version == 'v1':
            records = self.__parse_a3m_file_v1(path, msa_depth, filt_mthd)
        elif self.version == 'v2':
            records = self.__parse_a3m_file_v2(path, msa_depth, filt_mthd)
        else:
            raise ValueError('unrecognized version: ' + self.version)

        return records


    def __parse_a3m_file_v1(self, path, msa_depth, filt_mthd):
        """Parse the A3M file - v1."""

        # configurations
        deletekeys = dict.fromkeys(string.ascii_lowercase)
        deletekeys['.'] = None
        deletekeys['*'] = None
        translation = str.maketrans(deletekeys)

        # parse the A3M file w/ Bio-based APIs
        records_raw = list(SeqIO.parse(path, 'fasta'))
        records_raw = self.__filter_records(records_raw, msa_depth, filt_mthd)
        records = [(x.description, str(x.seq).translate(translation)) for x in records_raw]

        return records


    def __parse_a3m_file_v2(self, path, msa_depth, filt_mthd):
        """Parse the A3M file - v2."""

        regex = re.compile(r'[\.\*a-z]')
        with open(path, 'r', encoding='UTF-8') as i_file:
            i_lines = [i_line.strip() for i_line in i_file]
            records_raw = [(i_lines[x][1:], i_lines[x + 1]) for x in range(0, len(i_lines), 2)]
            records_raw = self.__filter_records(records_raw, msa_depth, filt_mthd)
            records = [(desc, re.sub(regex, '', seq)) for desc, seq in records_raw]

        return records


    def __filter_records(self, records_src, msa_depth, filt_mthd):
        """Filter MSA records to the specified depth."""

        if (msa_depth == -1) or (len(records_src) <= msa_depth):
            records_dst = records_src  # do nothing
        else:
            if filt_mthd == 'unif':
                records_dst = [records_src[0]] + random.sample(records_src[1:], msa_depth - 1)
            elif filt_mthd == 'topk':
                records_dst = records_src[:msa_depth]
            elif filt_mthd == 'neff':
                records_dst = self.__sample_records(records_src, msa_depth)
            else:
                raise ValueError('unrecognized MSA filtering method: ' + filt_mthd)

        return records_dst


    def __sample_records(self, records_src, msa_depth):
        """Sample MSA records w/ the Neff-based critertion."""

        # extract amino-acid sequences
        regex = re.compile(r'[\.\*a-z]')
        aa_seqs_full = [re.sub(regex, '', x[1]) for x in records_src]

        # setup file paths
        rand_str = get_rand_str()
        txt_fpath_in = os.path.join(self.tmp_dpath, f'{rand_str}.in')
        txt_fpath_out = os.path.join(self.tmp_dpath, f'{rand_str}.out')

        # prepare the input file
        with open(txt_fpath_in, 'w', encoding='UTF-8') as o_file:
            n_seqs = len(aa_seqs_full)
            n_resds = len(aa_seqs_full[0])
            o_file.write(f'{n_seqs} {n_resds} {msa_depth}\n')
            o_file.write('\n'.join(aa_seqs_full) + '\n')

        # call <msa_sampler> (source: cpp/msa_sampler.cc)
        subprocess.call([self.bin_fpath, txt_fpath_in, txt_fpath_out])

        # parse the output file
        with open(txt_fpath_out, 'r', encoding='UTF-8') as i_file:
            idxs_smpl = [int(i_line.split()[0]) for i_line in i_file]
            records_dst = [records_src[x] for x in idxs_smpl]

        # clean-up
        os.remove(txt_fpath_in)
        os.remove(txt_fpath_out)

        return records_dst
