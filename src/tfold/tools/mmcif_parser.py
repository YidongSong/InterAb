"""Parser for mmCIF files."""

import gzip
from collections import OrderedDict

import torch
from openfold.data import mmcif_parsing

from tfold.utils import get_md5sum
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_converter import ProtConverter


class mmCIFParseError(Exception):
    """Exceptions raised when parsing a mmCIF file."""


class mmCIFParser():
    """Parser for mmCIF files."""

    def __init__(self):
        """Constructor function."""

        pass


    @classmethod
    def load(cls, path, chain_id=None):
        """Parse the protein data from mmCIF file.

        Args:
        * path: path to the mmCIF file (could be GZIP-compressed)
        * chain_id: (optional) chain ID (if set to None, then all the chains will be returned)

        Returns:
        * prot_data: dict of amino-acid sequences, atom coordinates & validness masks
        * meta_data: dict of meta information from the header (method, release date, and resolution)
        * error_msg: error message (None: no error)
        """

        try:
            # read the whole mmCIF file
            if path.endswith('.cif'):
                with open(path, 'r', encoding='UTF-8') as i_file:
                    mmcif_string = i_file.read()
            elif path.endswith('.cif.gz'):
                with gzip.open(path, 'rt') as i_file:
                    mmcif_string = i_file.read()
            else:
                raise mmCIFParseError('UNRECOGNIZED_FILE_FORMAT')

            # parse the mmCIF file
            file_id = get_md5sum(path)
            outputs = mmcif_parsing.parse(file_id=file_id, mmcif_string=mmcif_string)
            mmcif_object = outputs.mmcif_object
            if mmcif_object is None:
                raise mmCIFParseError(list(outputs.errors.values())[0])
            meta_data = mmcif_object.header
            chain_ids = sorted(list(mmcif_object.chain_to_seqres.keys()))

            # parse detailed data for each chain (or only for the specified chain)
            prot_data = OrderedDict()
            if chain_id is not None:
                if chain_id not in chain_ids:
                    raise mmCIFParseError('CHAIN_ID_NOT_FOUND')
                prot_data[chain_id] = cls.__get_chain_data(mmcif_object, chain_id)
            else:
                for chain_id in chain_ids:
                    prot_data[chain_id] = cls.__get_chain_data(mmcif_object, chain_id)

            # set the error message to None
            error_msg = None
        except mmCIFParseError as error:
            prot_data, meta_data, error_msg = None, None, error

        return prot_data, meta_data, error_msg


    @classmethod
    def __get_chain_data(cls, mmcif_object, chain_id):
        """Get data for the specified chain ID."""

        # check whether the amino-acid sequence contains unexpected token(s)
        aa_seq = mmcif_object.chain_to_seqres[chain_id]
        if len(set(aa_seq) - set(RESD_NAMES_1C + ['X'])) != 0:
            raise mmCIFParseError('UNEXPECTED_AA_TYPE')

        # parse atom coordinates and convert them into the <atom14> format
        cord_tns_np, cmsk_mat_np = mmcif_parsing.get_atom_coords(mmcif_object, chain_id)
        cord_tns = torch.tensor(cord_tns_np, dtype=torch.float32)
        cmsk_mat = torch.tensor(cmsk_mat_np, dtype=torch.int8)
        cord_tns, cmsk_mat = ProtConverter.atom37_to_atom14(aa_seq, cord_tns, cmsk_mat)
        chain_data = {'seq': aa_seq, 'cord': cord_tns, 'cmsk': cmsk_mat}

        return chain_data
