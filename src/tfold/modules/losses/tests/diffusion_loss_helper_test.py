"""Unit-tests for the <ConfidenceLossHelper_> module."""

import torch

from tfold.tools import PdbParser
from tfold.tools import ProtStruct
from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.utils import parse_fas_file
from tfold.modules.losses.diffusion_loss_helper import DiffusionLossHelper
from tfold.modules.losses.utils import disp_loss_n_metrics


def main():
    """Main entry."""

    # configurations
    config = {
        'sigma_data': 16,
        'wc_mse': 1.0,
        'wc_bond': 1.0,
        'wc_smooth_lddt': 1.0,
        'skip_loss': False,
    }

    device = torch.device('cuda:0')
    loss_helper = DiffusionLossHelper(**config)

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse the native PDB file
    prot_id = 'T1024-D1'
    data_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Datasets/USM-Project/CASP14'
    fas_fpath = f'{data_dir}/fasta.files/{prot_id}.fasta'
    pdb_fpath = f'{data_dir}/pdb.files.native/{prot_id}.pdb'
    _, aa_seq = parse_fas_file(fas_fpath)
    _, cord_tns, cmsk_mat, _, error_msg = PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath} {error_msg}'

    n_smpls = 20
    n_resds = len(aa_seq)
    cord_tns = cord_tns.to(device)
    cmsk_mat = cmsk_mat.to(device)
    cmsk_vld = ProtStruct.get_cmsk_vld(aa_seq, device)

    # prepare atom-level input
    """n_atom = torch.sum(cmsk_vld)
    molecule_atom_lens = torch.sum(cmsk_vld, dim=-1)"""
    atom_tns = cord_tns[cmsk_vld.bool()].unsqueeze(0)
    amsk_mat = cmsk_mat[cmsk_vld.bool()].unsqueeze(0)

    noise_sigmas = torch.randint(0, 16, (n_smpls,), device=device).float()

    inputs = {
        'base': {
            'seq': aa_seq,
            'atom': atom_tns,
            'amsk': amsk_mat,
        }
    }

    cord_pred = torch.randn(n_smpls, n_resds, 14, 3).to(device)
    atom_pred = cord_pred[:, cmsk_vld.bool()]

    outputs = {
        '3d': {
            'atom': atom_pred,
        },
        'sigmas': noise_sigmas,
    }

    inspect_data(inputs, name='[inputs]')
    inspect_data(outputs, name='[inputs] pred_dict')

    diffusion_loss, metrics = loss_helper.run(inputs, outputs)
    disp_loss_n_metrics(diffusion_loss, metrics, name='CONF')


if __name__ == '__main__':
    main()
