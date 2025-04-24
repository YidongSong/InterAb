import os
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import pdb
from Bio import PDB 
import pickle
import sys
from tfold.tools import ProtStruct
from tfold.tools import PdbParser
from tfold.tools import prot_constants as constants
from concurrent.futures import ThreadPoolExecutor


def get_chains_from_pdb(pdb_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("PDB_structure", pdb_file)

    chains_info = []

    # 遍历模型、链和残基
    for model in structure:
        for chain in model:
            chain_id = chain.id
            residue_count = len(chain)
            chains_info.append((chain_id, residue_count))
    chain_list = []
    for chain_id, residue_count in chains_info:
        chain_list.append(chain_id)

    return chain_list

def Mean_xyz(pdb_file):
    parser = PDB.PDBParser()
    structure = parser.get_structure("PDB", pdb_file)

    # 存储坐标
    coordinates = []

    # 遍历结构并提取坐标
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    coordinates.append(atom.get_coord())
    
    return np.mean(coordinates, axis=0)

def Mean_cord(atom_tns, mean_xyz):
    """
    The protein coordinates were normalized by mean subtraction.
    """
    for i in range(atom_tns.shape[0]):
        atom_tns[i] -= mean_xyz

    return atom_tns


def prepare_atom_inputs(seq_list, idx_list, pdb_fpath, sfeat=None, pfeat=None, dtype=torch.float32, device='cpu'):

    seq_full = ''.join(seq_list)
    seq_len = len(seq_full)

    cmsk_vld = ProtStruct.get_cmsk_vld(seq_full, device)

    # atom_ref_pos: [N, 3]
    atom_ref_pos = []
    chain_uid = []
    chain_list = get_chains_from_pdb(pdb_fpath)
    if len(chain_list) == 3:
        chain_dict = {chain_list[0]: 1, chain_list[1]: 2, chain_list[2]: 3}
    if len(chain_list) == 2:
        chain_dict = {chain_list[0]: 1, chain_list[1]: 2}
    chain_uid = []
    mean_coordinates = Mean_xyz(pdb_fpath)
    x_mean, y_mean, z_mean = mean_coordinates[0], mean_coordinates[1], mean_coordinates[2]

    
    for chain_id in chain_list:
        
        for i in range(len(idx_list[chain_id])):
            aa_seq, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath, chain_id=chain_id)
            idx_beg = idx_list[chain_id][i][0]
            idx_end = idx_list[chain_id][i][-1] + 1
            seq_test = aa_seq[idx_beg:idx_end]
            cord_tns = cord_tns[idx_beg:idx_end]
            cmsk_mat = cmsk_mat[idx_beg:idx_end]
            atom_tns = cord_tns[cmsk_mat.bool()]
            atom_tns = Mean_cord(atom_tns, torch.tensor([x_mean, y_mean, z_mean]))
            atom_ref_pos.append(atom_tns)
            chain_tns = torch.full((atom_tns.shape[0], 1), chain_dict[chain_id])
            chain_uid.append(chain_tns)
    
    atom_ref_pos = torch.cat(atom_ref_pos, dim=0).to(dtype)  # [N, 3]
    chain_uid = torch.cat(chain_uid, dim=0).to(dtype) # [N, 1]

    molecule_atom_lens = torch.sum(cmsk_vld, dim=-1)
    N_ATOMS = torch.sum(cmsk_vld)

    residue_indices = torch.arange(seq_len).to(device)
    atom_ref_space_uid = residue_indices.flatten().repeat_interleave(molecule_atom_lens.flatten(), dim=-1)

    # ref_mask [N, 1]
    ref_mask = torch.ones((N_ATOMS, 1), dtype=dtype)

    # ref_element [N, 37]
    np_aatype = np.array([constants.RESD_ORDER_WITH_X[resd] for resd in list(seq_full)])
    per_res_idx = constants.restype_atom37_to_atom14[np_aatype]
    res_idx = np.tile(np.arange(per_res_idx.shape[0])[..., None], (1, per_res_idx.shape[1]))
    atom37_pos_mask = constants.restype_atom37_mask[np_aatype]
    atom37_pos_mask = torch.from_numpy(atom37_pos_mask).type_as(cmsk_vld)

    cmsk37_mat = cmsk_vld[res_idx, per_res_idx] * atom37_pos_mask
    cmsk37_mat = cmsk37_mat.reshape(seq_len, 37)
    nonzero_indices = torch.nonzero(cmsk37_mat, as_tuple=False)

    ref_element = torch.zeros((N_ATOMS, constants.ATOM_TYPE_NUM), dtype=dtype, device=device)
    ref_element[torch.arange(N_ATOMS), nonzero_indices[:, 1]] = 1
    
    # prepare atom features
    atom_inputs = {
        'atom_feats': torch.cat([atom_ref_pos, ref_mask, ref_element, chain_uid], dim=-1),  # [N, 42]
        # 'atom_feats': torch.cat([ref_element, chain_uid], dim=-1),  # [N, 38]
        'atom_ref_pos': atom_ref_pos,  # [N, 3]
        'atom_ref_space_uid': atom_ref_space_uid,  # [N]
        'molecule_atom_lens': molecule_atom_lens,
    }

    if sfeat is not None:
        atom_inputs['sfeat'] = sfeat.detach().to(dtype)
    if pfeat is not None:
        atom_inputs['pfeat'] = pfeat.detach().to(dtype)

    # insert an additional batch dimension
    def _add_batch_dim(obj):
        if isinstance(obj, dict):
            return {k: _add_batch_dim(v) for k, v in obj.items()}
        return obj.unsqueeze(dim=0) if isinstance(obj, torch.Tensor) else obj

    atom_inputs = _add_batch_dim(atom_inputs)

    return atom_inputs  # [1, N, dim]


"""
Calculate the atom_inputs
"""

chai_stru_ppi = pickle.load(open('../data/atom/chai_ppi.pkl', 'rb'))
atom_inputs_dict = {}
for key in tqdm(chai_stru_ppi.keys()):
    pdb_path = '../data/atom/complex_pdb/' + key + '.pdb'
    seq_list = []
    try:
        idx_list = chai_stru_ppi[key]['idx_ppi']
        for key1 in chai_stru_ppi[key]['seq_ppi'].keys():
            seq_list.append(chai_stru_ppi[key]['seq_ppi'][key1])
        atom_inputs_1 = prepare_atom_inputs(seq_list, idx_list, pdb_path)
        atom_inputs_dict[key] = atom_inputs_1
    except Exception as e:
        print(e)

pickle.dump(atom_inputs_dict, open('../data/atom/atom_inputs.pkl', 'wb'))  
