"""Unit-tests for the <AtomAttentionEncoder> module."""


import logging

import torch
from einops import repeat
from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.utils import get_peak_memory
from tfold.utils import send_to_device
from tfold.modules.af3_smod.atom_attn_encoder import AtomAttentionEncoder


def main():
    """Main entry."""

    # configurations
    n_smpls = 2
    n_resds = 400
    device = torch.device('cuda')

    def repeat_tensor(tensor):
        return repeat(tensor, 'b ... -> (b a) ...', a=n_smpls)

    # config
    config = {
        'n_dims_atom_inputs': 128,
        'n_dims_atom': 64,
        'n_dims_atompair': 16,
        'atoms_per_window': 27,
        'n_dims_token': 128,
        'n_dims_sfea': 128,
        'n_dims_pfea': 64,
        'atom_transformer_blocks': 2,
        'atom_transformer_heads': 4,
    }

    molecule_atom_lens = torch.randint(4, 8, (1, n_resds))
    atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()

    # atom features
    atom_ref_pos = torch.randn((atom_seq_len, 3), dtype=torch.float32).unsqueeze(0)
    atom_feats = torch.randn((atom_seq_len, 128), dtype=torch.float32).unsqueeze(0)

    residue_indices = torch.arange(n_resds)
    atom_ref_space_uid = residue_indices.flatten().repeat_interleave(
        molecule_atom_lens.flatten(), dim=0).unsqueeze(0)

    atom_inputs = {
        'atom_feats': repeat_tensor(atom_feats),
        'atom_ref_pos': repeat_tensor(atom_ref_pos),
        'atom_ref_space_uid': repeat_tensor(atom_ref_space_uid),
        'molecule_atom_lens': repeat_tensor(molecule_atom_lens),
    }

    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    nfea_tns = torch.randn((n_smpls, atom_seq_len, 3), dtype=torch.float32, device=device)
    sfea_tns_trunk = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32, device=device)
    pfea_tns = torch.randn((n_smpls, n_resds, n_resds, config['n_dims_pfea']), dtype=torch.float32, device=device)
    atom_inputs = send_to_device(atom_inputs, device)

    inspect_data(atom_inputs, name='atom_inputs')
    logging.info('[inputs] nfea_tns: %s', nfea_tns.shape)
    logging.info('[inputs] sfea_tns_trunk: %s', sfea_tns_trunk.shape)
    logging.info('[inputs] pfea_tns: %s', pfea_tns.shape)

    # test the AtomAttentionEncoder module
    atom_attn_encoder = AtomAttentionEncoder(**config).to(device)
    sfea_tns, afea_tns, atom_feat_cond, atompair_feat_cond = atom_attn_encoder(
        atom_inputs, nfea_tns, sfea_tns_trunk, pfea_tns)
    print('peak GPU memory after atom_attn_encoder: %.2f (MB)' % get_peak_memory())
    logging.info('[outputs] pfea_tns: %s', pfea_tns.shape)


if __name__ == '__main__':
    main()
