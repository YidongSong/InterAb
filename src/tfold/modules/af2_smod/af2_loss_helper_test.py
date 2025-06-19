"""Unit-tests for the <AF2LossHelper> module."""

import os
import logging

import torch
from torch import nn

from tfold.utils import tfold_init
from tfold.utils import rot2quat
from tfold.tools import compute_ptmscore
from tfold.tools import PdbParser
from tfold.tools import ProtStruct
from tfold.tools import ProtConverter
from tfold.modules.af2_smod.af2_loss_helper import AF2LossHelper
from tfold.modules.af2_smod.utils import init_qta_params


def centralize(aa_seq, cord_tns, cmsk_mat):
    """Centralize 3D coordinates based on all the CA atoms."""

    cord_mat_ca = ProtStruct.get_atoms(aa_seq, cord_tns, ['CA'])
    cmsk_vec_ca = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])
    cord_vec = torch.sum(cmsk_vec_ca.unsqueeze(dim=1) * cord_mat_ca, dim=0) / torch.sum(cmsk_vec_ca)
    cord_tns -= cord_vec.view(1, 1, 3)

    return cord_tns


def get_rand_preds(aa_seq, device):
    """Get randomized raw predictions."""

    # initialization
    n_resds = len(aa_seq)

    # get randomized raw predictions
    quat_tns, trsl_tns, angl_tns = init_qta_params(1, n_resds, mode='random')
    quat_tns = quat_tns[0].to(device)  # remove the batch dimension
    trsl_tns = trsl_tns[0].to(device)
    angl_tns = angl_tns[0].to(device)

    return quat_tns, trsl_tns, angl_tns


def get_optim_preds(aa_seq, cord_tns, cmsk_mat):
    """Get optimal raw predictions from ground-truth 3D coordinates."""

    # calculate optimal QTA parameters
    prot_struct = ProtStruct()
    prot_converter = ProtConverter()
    prot_struct.init_from_cord(aa_seq, cord_tns, cmsk_mat)
    prot_struct.build_fram_n_angl(prot_converter)
    quat_tns = rot2quat(prot_struct.fram_tns_bb[:, 0, :3], quat_type='full').detach()
    trsl_tns = prot_struct.fram_tns_bb[:, 0, 3].detach()
    angl_tns = prot_struct.angl_tns.detach()

    return quat_tns, trsl_tns, angl_tns


def get_plddt_n_ptm(aa_seq, device):
    """Get dicts of pLDDT & pTM predictions."""

    # initialization
    n_resds = len(aa_seq)
    n_bins_plddt = 50
    n_bins_ptm = 64
    bin_vals = (torch.arange(n_bins_plddt, device=device) + 0.5) / n_bins_plddt

    # get the dict of pLDDT predictions
    logt_tns = torch.randn((n_resds, n_bins_plddt), device=device)
    plddt_res = torch.sum(bin_vals.view(1, -1) * nn.functional.softmax(logt_tns, dim=1), dim=1)
    plddt_chn = torch.mean(plddt_res)
    plddt_dict = {'logit': logt_tns, 'plddt-r': plddt_res, 'plddt-c': plddt_chn}

    # get the dict of pTM predictions
    logt_tns = torch.randn((n_resds, n_resds, n_bins_ptm), device=device)
    tmsc_dict = {'ptm_logt': logt_tns, 'ptm': compute_ptmscore(logt_tns)}

    return plddt_dict, tmsc_dict


def build_af2smod_outputs(aa_seq, quat_tns, trsl_tns, angl_tns):  # pylint: disable=too-many-locals
    """Build <AF2SMod>-like outputs from raw predictions."""

    # configurations
    n_lyrs = 4
    device = quat_tns.device

    # initialization
    prot_struct = ProtStruct()
    prot_converter = ProtConverter()
    plddt_dict, tmsc_dict = get_plddt_n_ptm(aa_seq, device)

    # build <AF2SMod>-like predictions
    params_list = []
    plddt_list = []
    cord_list = []
    fram_tns_sc = None
    for idx_lyr in range(n_lyrs):
        # record QTA parameters & pLDDT predictions
        params = {
            'quat': quat_tns,
            'trsl': trsl_tns,
            'angl': angl_tns,
            'quat-u': quat_tns,
        }
        params_list.append(params)
        plddt_list.append(plddt_dict)

        # reconstruct per-atom 3D coordinates
        atom_set = 'ca' if idx_lyr != n_lyrs - 1 else 'fa'
        prot_struct.init_from_param(aa_seq, params, prot_converter, atom_set)
        cord_list.append(prot_struct.cord_tns)

        # obtain side-chain local frames
        if idx_lyr == n_lyrs - 1:
            prot_struct.build_fram_n_angl(prot_converter, build_sc=True)
            fram_tns_sc = prot_struct.fram_tns_sc

    return params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict


def main():  # pylint: disable=too-many-locals,too-many-statements
    """Main entry."""

    # configurations
    config = {
        'wc_fape': 1.0,
        'wc_angl': 1.0,
        'wc_lddt': 0.1,
        'wc_qnrm': 0.02,
        'wc_clsh': 0.01,
        'wc_tmsc': 0.1,
        'wc_ifape': 0.0,
        'loss_nb_max': 1.0,
        'debug': False,
    }
    device = torch.device('cuda:0')
    prot_id = 'T1024-D1'
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(curr_dir, '../../tools/examples')
    fas_fpath = os.path.join(data_dir, f'{prot_id}.fasta')
    pdb_fpath = os.path.join(data_dir, f'{prot_id}_native.pdb')

    # initialization
    tfold_init(verb_levl='DEBUG')

    # parse the native PDB file
    aa_seq, cord_tns, cmsk_mat, meta_data, error_msg = \
        PdbParser.load(pdb_fpath, fas_fpath=fas_fpath)
    assert error_msg is None, f'failed to parse the PDB file: {pdb_fpath}'
    cord_tns = cord_tns.to(device)
    cmsk_mat = cmsk_mat.to(device)
    cord_tns = centralize(aa_seq, cord_tns, cmsk_mat)

    # initialize <AF2LossHelper> for loss evaluation
    loss_helper = AF2LossHelper(**config)
    loss_helper.init(aa_seq, cord_tns, cmsk_mat, meta_data['reso'], meta_data['mthd'])

    # evaluate over randomly initialized predictions
    logging.info('=== evaluating over randomly initialized predictions ===')
    quat_tns, trsl_tns, angl_tns = get_rand_preds(aa_seq, device)
    af2smod_outputs = build_af2smod_outputs(aa_seq, quat_tns, trsl_tns, angl_tns)
    loss, metrics = loss_helper.calc_loss(*af2smod_outputs)
    logging.info('loss: %.4f', loss.item())
    for key, val in metrics.items():
        logging.info('%s: %.4f', key, val)

    # evaluate over optimally initialized predictions
    logging.info('=== evaluating over optimally initialized predictions ===')
    quat_tns, trsl_tns, angl_tns = get_optim_preds(aa_seq, cord_tns, cmsk_mat)
    af2smod_outputs = build_af2smod_outputs(aa_seq, quat_tns, trsl_tns, angl_tns)
    loss, metrics = loss_helper.calc_loss(*af2smod_outputs)
    logging.info('loss: %.4f', loss.item())
    for key, val in metrics.items():
        logging.info('%s: %.4f', key, val)


if __name__ == '__main__':
    main()
