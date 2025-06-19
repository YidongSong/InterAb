import string

import torch
from openfold.np.residue_constants import *

from tfold.third_parties.esm.data import Alphabet

# This mapping is used when we need to store atom data in a format that requires
# fixed atom data size for every residue (e.g. a numpy array).
atom_types = [
    'N', 'CA', 'C', 'CB', 'O', 'CG', 'CG1', 'CG2', 'OG', 'OG1', 'SG', 'CD',
    'CD1', 'CD2', 'ND1', 'ND2', 'OD1', 'OD2', 'SD', 'CE', 'CE1', 'CE2', 'CE3',
    'NE', 'NE1', 'NE2', 'OE1', 'OE2', 'CH2', 'NH1', 'NH2', 'OH', 'CZ', 'CZ2',
    'CZ3', 'NZ', 'OXT'
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}
atom_type_num = len(atom_types)  # := 37.
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

tpl_fea_dict = {
        'v1':   [23, 1,  '.pkl', 0],
        'v2':   [50, 1,  '.pkl', 0],
        'v2.1': [51, 1,  '.pkl', 0],
        'v3':   [57, 88, '.npz', 1],
        # 'v3.1': [58, 88,  None, None],
}

def gen_idx2res():
    idx2resname = {}
    for key in resname_to_idx.keys():
        if key != 'UNK':
            res = restype_3to1[key]
            idx = resname_to_idx[key]
            idx2resname[idx] = res
    idx2resname[20] = 'X'
    idx2resname[21] = '-'
    return idx2resname

def get_af2_msa_trans_idx():
    idx2resname[22] = '<mask>'
    msa_trans_resname2idx = Alphabet.from_architecture('msa_transformer').tok_to_idx
    af2idx_to_msatrans_idx = {}
    for id in idx2resname.keys():
        msa_trans_id = msa_trans_resname2idx[idx2resname[id]]
        af2idx_to_msatrans_idx[id] = msa_trans_id
    return af2idx_to_msatrans_idx

def getval_array(d):
    v = np.array(list(d.values()))
    k = np.array(list(d.keys()))
    maxv = k.max()
    minv = k.min()
    n = maxv - minv + 1
    val = np.empty(n, dtype=v.dtype)
    val[k] = v
    return val

# For alphafold2 -> MSA Trans token conversion
idx2resname = gen_idx2res()
af2idx_to_msatrans_idx = get_af2_msa_trans_idx()
af2idx_to_msatrans_val_arr = getval_array(af2idx_to_msatrans_idx)

type_dict = {
    'aatype' :torch.int64,
    'residue_index' :torch.int64,
    'seq_length' :torch.int64,
    'template_aatype' :torch.int64,
    'template_all_atom_positions' :torch.float32,
    'template_sum_probs' :torch.float32,
    'template_all_atom_mask' :torch.float32,
    'seq_mask' :torch.float32,
    'msa_mask' :torch.float32,
    'msa_row_mask' :torch.float32,
    'template_mask' :torch.float32,
    'template_pseudo_beta' :torch.float32,
    'template_pseudo_beta_mask' :torch.float32,
    'template_torsion_angles_sin_cos' :torch.float32,
    'template_alt_torsion_angles_sin_cos' :torch.float32,
    'template_torsion_angles_mask' :torch.float32,
    'atom14_atom_exists' :torch.float32,
    'residx_atom14_to_atom37' :torch.int64,
    'residx_atom37_to_atom14' :torch.int64,
    'atom37_atom_exists' :torch.float32,
    'extra_msa' :torch.int64,
    'extra_msa_mask' :torch.float32,
    'extra_msa_row_mask' :torch.float32,
    'bert_mask' :torch.float32,
    'true_msa' : torch.int64,
    'true_msa_esm_tokens': torch.int64,
    'masked_msa_esm_tokens': torch.int64,
    'extra_msa_esm_tokens': torch.int64,
    'msa_esm_tokens':  torch.int64,
    'extra_has_deletion' :torch.float32,
    'extra_deletion_value' :torch.float32,
    'msa_feat' :torch.float32,
    'target_feat' :torch.float32,
    'lbl' :torch.int64,
}

fp_tensor_dict = {
    'template_all_atom_positions' :torch.float32,
    'template_sum_probs' :torch.float32,
    'template_all_atom_mask' :torch.float32,
    'seq_mask' :torch.float32,
    'msa_mask' :torch.float32,
    'msa_row_mask' :torch.float32,
    'template_mask' :torch.float32,
    'template_pseudo_beta' :torch.float32,
    'template_pseudo_beta_mask' :torch.float32,
    'template_torsion_angles_sin_cos' :torch.float32,
    'template_alt_torsion_angles_sin_cos' :torch.float32,
    'template_torsion_angles_mask' :torch.float32,
    'atom14_atom_exists' :torch.float32,
    'atom37_atom_exists' :torch.float32,
    'extra_msa_mask' :torch.float32,
    'extra_msa_row_mask' :torch.float32,
    'bert_mask' :torch.float32,
    'extra_has_deletion' :torch.float32,
    'extra_deletion_value' :torch.float32,
    'msa_feat' :torch.float32,
    'target_feat' :torch.float32,
    't1ds':torch.float32,
    't2ds':torch.float32,
}

fp_tensor_set = set(fp_tensor_dict.keys())

crop_index_dict = {
    'aatype': [0],
    'residue_index': [0],
    'template_aatype': [1],
    'template_all_atom_positions': [1],
    'template_all_atom_mask': [1],
    'seq_mask': [0],
    'msa_mask': [1],
    'template_pseudo_beta': [1],
    'template_pseudo_beta_mask': [1],
    'template_torsion_angles_sin_cos': [1],
    'template_alt_torsion_angles_sin_cos': [1],
    'template_torsion_angles_mask': [1],
    'atom14_atom_exists': [0],
    'residx_atom14_to_atom37': [0],
    'residx_atom37_to_atom14': [0],
    'atom37_atom_exists': [0],
    'extra_msa': [1],
    'extra_msa_mask': [1],
    'bert_mask': [1],
    'true_msa': [1],
    'true_msa_esm_tokens': [1],
    'masked_msa_esm_tokens': [1],
    'extra_msa_esm_tokens': [1],
    'extra_has_deletion': [1],
    'extra_deletion_value': [1],
    'msa_feat': [1],
    'msa_esm_tokens': [1],
    'target_feat': [0],
    'lbl': [1, 2],
    't1ds':[1],
    't2ds':[1, 2],
    'lbls':[1, 2],
}
