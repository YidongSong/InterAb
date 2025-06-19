"""Unit-tests for the <PspFeaturizer> class."""

import os
import logging
import time
import random

import torch

from tfold.utils import tfold_init
from tfold.tools import PdbParser
from tfold.tools import ProtStruct
from collections import OrderedDict
from tfold.tools import PspFeaturizer
from tfold.third_parties.tfold_released.protein.parser import parse_a3m


def read_name_list(filename):
    with open(filename, 'r') as F:
        content = [item.strip() for item in F.readlines()]
    return content


def main():
    """Main entry."""

    # configurations
    psp_path = '/apdcephfs_qy3/share_301997302/fandiwu/Pre-trained.Models/tFold-release/checkpoints/alphafold_4_ptm.pth'
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # psp_feat_extractor = PspFeaturizer(psp_path, 3, device)
    psp_feat_extractor = PspFeaturizer.restore(psp_path)
    psp_feat_extractor.model.to(device)

    root_dir = '/mnt/ai4x_ceph/fandiwu/buddy1/Datasets/USM-Project/SAbDab-22-Ag'
    pid_fpath = f'{root_dir}/prot_ids.txt'
    msa_dpath = '/mnt/ai4x_ceph/fandiwu/rodatasets/SAbDab/msa.files'
    out_dpath = f'{root_dir}/results/test'

    with open(pid_fpath, 'r') as F:
        prot_ids = [item.strip() for item in F.readlines()]
        random.shuffle(prot_ids)

    os.makedirs(out_dpath, exist_ok=True)

    for prot_id in prot_ids:
        out_fpath = f'{out_dpath}/{prot_id}.pdb'
        logging.info('prot_id: %s', prot_id)
        if os.path.exists(out_fpath):
            continue
        msa_fpath = os.path.join(msa_dpath, f'{prot_id}.a3m')

        with open(msa_fpath) as f:
            msa, deletion_matrix = parse_a3m(f.read())
            aa_seq = msa[0]

        start = time.time()
        result_dict, ptm = psp_feat_extractor(msa, deletion_matrix, num_recycles=0)
        logging.info('sequence: %s (%d AAs)', aa_seq, len(aa_seq))
        logging.info('Finish %s psp inference in %.3f' % (prot_id, time.time() - start))
        logging.info('mfea_mat: %s', result_dict['mfea'][-1].shape)
        logging.info('pfea_tns: %s', result_dict['pfea'][-1].shape)
        logging.info('pTMscore: %.3f', ptm)

        cord = result_dict['cord'][-1]
        prot_data = OrderedDict({
            'A': {
                'seq': aa_seq,
                'cord': cord,
                'cmsk': ProtStruct.get_cmsk_vld(aa_seq, device),
            },
        })
        PdbParser.save_multimer(prot_data, out_fpath)


if __name__ == '__main__':
    main()
