"""The template feature extractor."""

import os
import torch
import pickle
import bz2
import logging
import numpy as np

from Bio.PDB.Polypeptide import is_aa
from Bio import pairwise2
from Bio.Align import substitution_matrices
from Bio.PDB import PDBParser, MMCIFParser

from alphafold.common import residue_constants
from alphafold.data.templates import _check_residue_distances

from openfold.data.templates import TEMPLATE_FEATURES
from openfold.data.data_transforms import (
    fix_templates_aatype,
    make_template_mask,
    make_pseudo_beta,
    atom37_to_torsion_angles
)
from openfold.data import parsers
from openfold.data.templates import TemplateHitFeaturizer
from openfold.utils.feats import build_template_pair_feat, build_template_angle_feat

from tfold.utils import all_logging_disabled
from tfold.tools import ProtStruct
from tfold.tools import AtomMapper
from tfold.tools.prot_constants import RESD_NAMES_3C, RESD_MAP_3TO1


class TemplateFeaturizer():
    """Template feature extractor"""

    def __init__(self):
        """Constructor function."""

        # base information

    def parpare_templ_feat(self, aa_seq, tpl_fpath, idx_resd_beg=None, idx_resd_end=None):

        assert os.path.exists(tpl_fpath), f'{tpl_fpath} is not exists'

        if tpl_fpath.endswith('.pdb'):  # use native structure
            template_info = self.prepare_pdb_feat(aa_seq, tpl_fpath)
        elif tpl_fpath.endswith('.hhr'):  # use antigen template info searching by hhsearch using MSA
            with all_logging_disabled(highest_level=logging.WARNING):
                template_info = self.prepare_hhr_feat(aa_seq, tpl_fpath)
        elif tpl_fpath.endswith('.pkl.bz2'):  # use preprocessed template feature
            template_info = self.prepare_pkl_feat(aa_seq, tpl_fpath)
        else:
            raise ValueError(f'Not support the format of tpl_path: {tpl_fpath}')

        if len(template_info['template_aatype']) > 0 and idx_resd_beg is not None and idx_resd_end is not None:
            for feat_type in ['template_aatype', 'template_all_atom_positions', 'template_all_atom_mask']:
                template_info[feat_type] = template_info[feat_type][:, idx_resd_beg:idx_resd_end]
        return template_info

    @classmethod
    def prepare_pdb_feat(cls, aa_seq, pdb_fpath):
        """Generate structure information for template module using pdb
        Args:
        * aa_seq: amino-acid sequence
        * pdb_fpath: path to the PDB file

        Returns:
        * structure_info: dict of structure feature
            template_all_atom_positions: [1, n_res, 37, 3]
            template_all_atom_mask: [1, n_res, 37]
            template_aatype: [1, n_res, 22]
            template_domain_names: [1]
        """
        seq2templateMapping = [list(range(len(aa_seq))), list(range(len(aa_seq)))]

        pdbseqs, residueList, _ = cls.extract_seq_from_pdb(pdb_fpath)
        seq2pdb_mapping = cls.map_seq2residue_list(aa_seq, pdbseqs[0], residueList[0])

        mapping = [-1] * len(aa_seq)
        for tgt_pos, tpl_pos in zip(
                seq2templateMapping[0], seq2templateMapping[1]):
            mapping[tgt_pos] = seq2pdb_mapping[tpl_pos]

        struc_positions, struc_positions_mask = cls.extract_coordinates_by_mapping(
            aa_seq, mapping, residueList[0]
        )

        struc_aatype = residue_constants.sequence_to_onehot(
            aa_seq, residue_constants.HHBLITS_AA_TO_ID
        )

        template_info = {
            "template_aatype": [np.array(struc_aatype)],
            "template_all_atom_positions": [np.array(struc_positions)],
            "template_all_atom_mask": [np.array(struc_positions_mask)],
            "template_domain_names": ['A'],
        }

        for name in template_info:
            template_info[name] = np.stack(template_info[name], axis=0).astype(TEMPLATE_FEATURES[name])

        return template_info

    @classmethod
    def prepare_hhr_feat(cls, aa_seq, hhr_fpath):
        """Generate structure information for template module using hhr"""

        af2_data_path = os.environ['af2_data_path']

        hhsearch_hits = parsers.parse_hhr(open(hhr_fpath).read())

        template_featurizer = TemplateHitFeaturizer(
            mmcif_dir=f'{af2_data_path}/pdb_mmcif/mmcif_files',
            max_template_date='2022-01-01',
            max_hits=20,
            kalign_binary_path='kalign',
            release_dates_path=None,
            obsolete_pdbs_path=f'{af2_data_path}/pdb_mmcif/obsolete.dat'
        )

        template_result = template_featurizer.get_templates(
            query_sequence=aa_seq,
            query_pdb_code=None,
            query_release_date=None,
            hits=hhsearch_hits
        )

        template_info = template_result.features

        return template_info

    @classmethod
    def prepare_pkl_feat(cls, aa_seq, pkl_fpath):
        template_info = pickle.load(bz2.BZ2File(pkl_fpath, 'rb'))

        return template_info

    @classmethod
    def prepare_multi_templ_feat(cls, aa_seq_1, cord_tns_1, aa_seq_2, cord_tns_2):
        """Generate multimer template information using two initial monomer structure

        Args:
        * aa_seq_1: amino-acid sequence of size L1
        * cord_tns_1:
        * aa_seq_2: amino-acid sequence of size L2
        * cord_tns_2:

        Returns:
        * template_info: dict of template feature
            template_all_atom_positions: [2, L1+L2, 37, 3]
            template_all_atom_mask: [2, L1+L2, 37]
            template_aatype: [2, L1+L2, 22]
            template_sum_probs: [2]
            template_domain_names: [2]
        """
        atom_mapper = AtomMapper()
        template_info = {}

        for feature_name in TEMPLATE_FEATURES:
            template_info[feature_name] = []

        cmsk_mat_1 = ProtStruct.get_cmsk_vld(aa_seq_1, cord_tns_1.device)
        cord_tns_1 = atom_mapper.run(aa_seq_1, cord_tns_1, frmt_src='n14-tf', frmt_dst='n37')
        cmsk_mat_1 = atom_mapper.run(aa_seq_1, cmsk_mat_1, frmt_src='n14-tf', frmt_dst='n37')

        positions = torch.cat([cord_tns_1.to('cpu'), torch.zeros(len(aa_seq_2), 37, 3)], dim=0)
        mask = torch.cat([cmsk_mat_1.to('cpu'), torch.zeros(len(aa_seq_2), 37)], dim=0)
        sequence = aa_seq_1 + '-' * len(aa_seq_2)
        aatype = residue_constants.sequence_to_onehot(
            sequence, residue_constants.HHBLITS_AA_TO_ID
        )
        result = {
            "template_all_atom_positions": np.array(positions),
            "template_all_atom_mask": np.array(mask),
            "template_sequence": sequence.encode(),
            "template_aatype": np.array(aatype),
            "template_domain_names": 'A',
            "template_sum_probs": 1,
        }
        for k in template_info:
            template_info[k].append(result[k])

        cmsk_mat_2 = ProtStruct.get_cmsk_vld(aa_seq_2, cord_tns_2.device)
        cord_tns_2 = atom_mapper.run(aa_seq_2, cord_tns_2, frmt_src='n14-tf', frmt_dst='n37')
        cmsk_mat_2 = atom_mapper.run(aa_seq_2, cmsk_mat_2, frmt_src='n14-tf', frmt_dst='n37')
        positions = torch.cat([torch.zeros(len(aa_seq_1), 37, 3), cord_tns_2.to('cpu')], dim=0)
        mask = torch.cat([torch.zeros(len(aa_seq_1), 37), cmsk_mat_2.to('cpu')], dim=0)
        sequence = '-' * len(aa_seq_1) + aa_seq_2
        aatype = residue_constants.sequence_to_onehot(
            sequence, residue_constants.HHBLITS_AA_TO_ID
        )
        result = {
            "template_all_atom_positions": np.array(positions),
            "template_all_atom_mask": np.array(mask),
            "template_sequence": sequence.encode(),
            "template_aatype": np.array(aatype),
            "template_domain_names": 'B',
            "template_sum_probs": 1,
        }
        for k in template_info:
            template_info[k].append(result[k])

        for name in template_info:
            template_info[name] = np.stack(template_info[name], axis=0).astype(TEMPLATE_FEATURES[name])

        template_info["aatype"] = residue_constants.sequence_to_onehot(
            sequence=aa_seq_1+aa_seq_2,
            mapping=residue_constants.restype_order_with_x,
        )

        tensor_features = {
            "template_all_atom_positions",
            "template_all_atom_mask",
            "template_sum_probs",
            "template_aatype",
            "aatype"
        }

        for k, v in template_info.items():
            if k in tensor_features:
                template_info[k] = torch.tensor(v)

        template_info = fix_templates_aatype(template_info)
        template_info = make_template_mask(template_info)
        template_info = make_pseudo_beta("template_")(template_info)
        template_info = atom37_to_torsion_angles("template_")(template_info)
        template_pair_feat, _ = cls.extract_template_feat(template_info)

        return template_pair_feat

    @classmethod
    def extract_template_feat(cls, template_info):
        """Prepare template module input from template information
        Args:
        * template_info: dict of template feature
            template_all_atom_positions: [n_template, n_res, 37, 3]
            template_all_atom_mask: [n_template, n_res, 37]
            template_aatype: [n_template, n_res, 22]
            template_sum_probs: [n_template]
            template_domain_names: [n_template]
            template_pseudo_beta
            template_pseudo_beta_mask
        Returns:
        *
        """

        n_templ = template_info["template_aatype"].shape[0]
        template_pair_feat = []
        template_angle_feat = []

        for i in range(n_templ):
            single_template_feat = {k: template_info[k][i] for k in template_info if k.startswith("template_")}
            template_pair_feat.append(
                build_template_pair_feat(
                    single_template_feat,
                    min_bin=3.25,
                    max_bin=50.75,
                    no_bins=39,
                    use_unit_vector=True
                )
            )
            template_angle_feat.append(
                build_template_angle_feat(
                    single_template_feat
                )
            )

        template_pair_feat = torch.stack(template_pair_feat)
        template_angle_feat = torch.stack(template_angle_feat)

        return template_pair_feat, template_angle_feat

    @classmethod
    def extract_seq_from_pdb(cls, pdbfile):
        """Extract sequence and residue coordinate from pdbfile.
        Args:
        * pdbfile: string of pdbfile

        Returns:
        * pdbseqs: list of pdbseqs
        * residueList: list of coordinates
        * chains: list of chain name
        """
        if pdbfile.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
        elif pdbfile.endswith(".cif"):
            parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("NoName", pdbfile)

        model = structure[0]
        pdbseqs = []
        residueLists = []
        chains = []
        for chain in model:
            residues = chain.get_residues()
            residueList = [r for r in residues if is_aa(r, standard=True) and
                           (r.get_resname().upper() in RESD_NAMES_3C)]
            pdbseq = "".join([RESD_MAP_3TO1[r.get_resname()] for r in residueList])
            pdbseqs.append(pdbseq)
            residueLists.append(residueList)
            chains.append(chain)

        return pdbseqs, residueLists, chains

    @classmethod
    def extract_coordinates_by_mapping(cls, sequence, seq2pdb_mapping, residueList):
        """Extract coordinates information from template by mapping

        Args:
        * sequence: string of target sequence
        * seq2pdb_mapping: mapping of sequence to template
        * residueList: coordinates of template

        Returns:
        * all_positions: coordinates of target sequence extracted from template according to mapping
        * all_positions_mask: mask of coordinates
        """
        num_res = len(sequence)
        all_positions = np.zeros([num_res, residue_constants.atom_type_num, 3])
        all_positions_mask = np.zeros([num_res, residue_constants.atom_type_num], dtype=np.int64)

        for res_index, mapping_index in enumerate(seq2pdb_mapping):
            pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
            mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
            # missing
            if mapping_index != -1:
                for atom in residueList[mapping_index].get_atoms():
                    atom_name = atom.get_name()
                    x, y, z = atom.get_coord()
                    if atom_name in residue_constants.atom_order.keys():
                        pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                        mask[residue_constants.atom_order[atom_name]] = 1.0
                    elif atom_name.upper() == "SE" and residueList[mapping_index].get_resname() == "MSE":
                        pos[residue_constants.atom_order["SD"]] = [x, y, z]
                        mask[residue_constants.atom_order["SD"]] = 1.0
            all_positions[res_index] = pos
            all_positions_mask[res_index] = mask
        _check_residue_distances(all_positions, all_positions_mask, 150.0)

        return all_positions, all_positions_mask

    @classmethod
    def map_seq2residue_list(cls, sequence, pdbseq, residueList):
        """map one query sequence to a list of PDB residues by sequence alignment

        Args:
        * sequence: string of sequence
        * pdbseq: string of sequence with standard amino acids
        * residueList: a list of residue with standard amino acids

        Returns:
        * seq2pdb_mapping: map fomr the query to pdb
        """

        # here we align PDB residues to query sequence instead of query to PDB residues
        blosum80 = substitution_matrices.load("BLOSUM80")
        alignments = pairwise2.align.localds(pdbseq, sequence, blosum80, -5, -0.2)
        if not bool(alignments):
            return None, None, None

        # find the alignment with the minimum difference
        diffs = []
        for alignment in alignments:
            mapping_pdb2seq = alignment2mapping(alignment)
            diff = 0
            for current_map, prev_map, current_residue, prev_residue in zip(
                mapping_pdb2seq[1:],
                mapping_pdb2seq[:-1],
                residueList[1:],
                residueList[:-1],
            ):
                # in principle, every PDB residue with valid 3D coordinates shall appear in the query sequence.
                # otherwise, apply a big penalty
                if current_map < 0:
                    diff += 10
                    continue

                if prev_map < 0:
                    continue

                # calculate the difference of sequence separation in both the PDB seq and the query seq
                # the smaller, the better
                current_id = current_residue.get_id()[1]
                prev_id = prev_residue.get_id()[1]
                id_diff = max(1, current_id - prev_id)
                map_diff = current_map - prev_map
                diff += abs(id_diff - map_diff)

            numMisMatches, numMatches = calc_num_mis_matches(alignment)
            diffs.append(diff - numMatches)

        diffs = np.array(diffs)
        alignment = alignments[diffs.argmin()]
        numMisMatches, numMatches = calc_num_mis_matches(alignment)

        # map from the query seq to pdb
        mapping_seq2pdb = alignment2mapping((alignment[1], alignment[0]))

        return mapping_seq2pdb


def alignment2mapping(alignment):
    """Calculate the index mapping from first seq to the second one using the provided alignment.

    Args:
    * alignment: list contains two sequence string with gap

    Returns:
    * mapping: list with the same length as the first sequence
    """
    S1, S2 = alignment[:2]

    # convert an aligned seq to a binary vector with 1 indicates aligned and 0 gap
    y = np.array([1 if a != "-" else 0 for a in S2])

    # get the position of each residue in the original sequence, starting from 0.
    ycs = np.cumsum(y) - 1
    np.putmask(ycs, y == 0, -1)

    # map from the 1st seq to the 2nd one. set -1 for an unaligned residue in the 1st sequence
    mapping = [y0 for a, y0 in zip(S1, ycs) if a != "-"]

    return mapping


def calc_num_mis_matches(alignment):
    """calculate the number of mismatches and matches excluding residues denoted as 'X'

    Args:
    * alignment: list contains two sequence string with gap

    Returns:
    * numMisMatches: number of alignment which do not match
    * numMatches: number of alignment which match
    """
    S1, S2 = alignment[:2]
    numMisMatches = np.sum([a != b for a, b in zip(S1, S2) if a != "-" and b != "-" and a != "X" and b != "X"])
    numMatches = np.sum([a == b for a, b in zip(S1, S2) if a != "-" and a != "X"])

    return numMisMatches, numMatches
