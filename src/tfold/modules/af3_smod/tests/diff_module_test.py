"""Unit-tests for the <AtomAttentionEncoder> module."""


import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.utils import get_peak_memory
from tfold.utils import send_to_device
from tfold.modules.af3_smod.diff_module import DiffusionModule


def main():
    """Main entry."""

    # configurations
    n_smpls = 1
    n_resds = 300
    device = torch.device('cuda')

    # config
    config = {
        'n_dims_atom_inputs': 100,
        'n_dims_pfea_trunk': 128,
        'n_dims_penc': 64,
        'atoms_per_window': 27,
        'n_dims_sfea': 192,
        'n_dims_pfea': 96,
        'n_dims_atom': 64,
        'n_dims_atompair': 16,
        'n_dims_fourier': 256,
        'n_dims_token': 384,
        'sigma_data': 16,
    }

    # atom features
    molecule_atom_lens = torch.randint(4, 8, (n_smpls, n_resds)).to(device)
    atom_seq_len = molecule_atom_lens.sum(dim=-1).amax()
    atom_ref_pos = torch.randn((n_smpls, atom_seq_len, 3), dtype=torch.float32)
    atom_feats = torch.randn((n_smpls, atom_seq_len, config['n_dims_atom_inputs']), dtype=torch.float32)

    residue_indices = torch.arange(n_resds).expand(n_smpls, -1).to(device)
    atom_ref_space_uid = residue_indices.flatten().repeat_interleave(
        molecule_atom_lens.flatten(), dim=-1).view(n_smpls, -1)
    atom_inputs = {
        'atom_feats': atom_feats,
        'atom_ref_pos': atom_ref_pos,
        'atom_ref_space_uid': atom_ref_space_uid,
        'molecule_atom_lens': molecule_atom_lens,
    }
    # initialization
    tfold_init(verb_levl='DEBUG')

    # randomly initialize input tensors
    times = torch.randn(1, ).to(device)
    nfea_tns = torch.randn((n_smpls, atom_seq_len, 3), dtype=torch.float32, device=device)
    sfea_tns = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32, device=device)
    sfea_tns_trunk = torch.randn((n_smpls, n_resds, config['n_dims_sfea']), dtype=torch.float32, device=device)
    pfea_tns_trunk = torch.randn(
        (n_smpls, n_resds, n_resds, config['n_dims_pfea_trunk']), dtype=torch.float32, device=device)
    penc_tns = torch.randn(
        (n_smpls, n_resds, n_resds, config['n_dims_penc']), dtype=torch.float32, device=device)
    atom_inputs = send_to_device(atom_inputs, device)

    inspect_data(atom_feats, name='atom_feats')
    logging.info('[inputs] nfea_tns: %s', nfea_tns.shape)
    logging.info('[inputs] times: %s', times.shape)
    logging.info('[inputs] sfea_tns: %s', sfea_tns.shape)
    logging.info('[inputs] sfea_tns_trunk: %s', sfea_tns_trunk.shape)
    logging.info('[inputs] pfea_tns_trunk: %s', pfea_tns_trunk.shape)
    logging.info('[inputs] penc_tns: %s', penc_tns.shape)

    # test the DiffusionModule
    diff_module = DiffusionModule(**config).to(device)
    atom_pos_upd = diff_module(
        nfea_tns,
        times,
        atom_inputs,
        sfea_tns,
        sfea_tns_trunk,
        pfea_tns_trunk,
        penc_tns,
        molecule_atom_lens
    )
    print('peak GPU memory after diffusion module: %.2f (MB)' % get_peak_memory())
    logging.info('[outputs] atom_pos_upd: %s', atom_pos_upd.shape)


if __name__ == '__main__':
    main()
