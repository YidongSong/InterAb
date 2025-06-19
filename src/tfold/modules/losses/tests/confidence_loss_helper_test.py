"""Unit-tests for the <ConfidenceLossHelper_> module."""

import torch

from tfold.tools import PdbParser
from tfold.utils import tfold_init
from tfold.utils import inspect_data
from tfold.utils import parse_fas_file
from tfold.modules.losses.confidence_loss_helper import ConfidenceLossHelper
from tfold.modules.losses.utils import disp_loss_n_metrics


def main():
    """Main entry."""

    # configurations
    config = {
        'wc_lddt': 1.0,
        'wc_pae': 1.0,
        'wc_pde': 1.0,
        'skip_loss': False,
    }

    device = torch.device('cuda:0')
    loss_helper = ConfidenceLossHelper(**config)

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse the native PDB file
    prot_id = 'T1024-D1'
    data_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Datasets/USM-Project/CASP14'
    fas_fpath = f'{data_dir}/fasta.files/{prot_id}.fasta'
    pdb_fpath = f'{data_dir}/pdb.files.native/{prot_id}.pdb'
    _, aa_seq = parse_fas_file(fas_fpath)
    _, cord_tns, cmsk_mat, _, error_msg = \
        PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath} {error_msg}'

    n_resds = len(aa_seq)

    cord_tns = cord_tns.unsqueeze(0).to(device)
    cmsk_mat = cmsk_mat.unsqueeze(0).to(device)

    inputs = {
        'base': {
            'seq': aa_seq,
            'cord': cord_tns,
            'cmsk': cmsk_mat,
        }
    }

    outputs = {
        'conf_logts': {
            'plddt': torch.randn((n_resds, 50), device=device),
            'pae': torch.randn((n_resds, n_resds, 64), device=device),
            'pde': torch.randn((n_resds, n_resds, 64), device=device),
        },
        '3d': {
            'cord': torch.randn((n_resds, 14, 3), device=device),
        },
    }

    inspect_data(inputs, name='[inputs]')
    inspect_data(outputs, name='[inputs] pred_dict')

    conf_loss, metrics = loss_helper.run(inputs, outputs)
    disp_loss_n_metrics(conf_loss, metrics, name='CONF')


if __name__ == '__main__':
    main()
