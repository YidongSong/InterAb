"""Unit-tests for the <MsaSampler> class."""

import logging

import torch

from tfold.utils import tfold_init
from tfold.utils import make_config_list
from tfold.tools.a3m_parser import A3mParser
from tfold.tools.msa_sampler import MsaSampler
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.third_parties.esm.data import Alphabet


def compare_tokens(tokens_true, tokens_pert, tokens_mask):  # pylint: disable=too-many-locals
    """Compare ground-truth & perturbed MSA tokens."""

    # initialization
    n_rows, n_cols = tokens_true.shape
    alphabet = Alphabet.from_architecture('MSA Transformer')
    idx_mask = alphabet.mask()
    idx_skip = alphabet.get_idx('-')
    idxs_amin = {alphabet.get_idx(x) for x in RESD_NAMES_1C}

    # compare all the MSA tokens
    n_hits_orig = 0
    n_hits_mask = 0
    n_hits_skip = 0
    n_hits_amin = 0
    n_hits_othr = 0
    for ir in range(n_rows):  # pylint: disable=invalid-name
        for ic in range(n_cols):  # pylint: disable=invalid-name
            token_true = tokens_true[ir][ic].item()
            token_pert = tokens_pert[ir][ic].item()
            if token_true == token_pert:
                n_hits_orig += 1
            else:
                assert tokens_mask[ir][ic] == 1
                if token_pert == idx_mask:
                    n_hits_mask += 1
                elif token_pert == idx_skip:
                    n_hits_skip += 1
                elif token_pert in idxs_amin:
                    n_hits_amin += 1
                else:
                    logging.warning('unexpected token: %s', token_pert)
                    n_hits_othr += 1

    # display summarized results
    logging.info('# of original tokens: %d', n_hits_orig)
    logging.info('# of masked tokens: %d', n_hits_mask)
    logging.info('# of skipping tokens: %d', n_hits_skip)
    logging.info('# of standard AA tokens: %d', n_hits_amin)
    logging.info('# of other tokens: %d', n_hits_othr)


def main():  # pylint: disable=too-many-locals
    """Main entry."""

    # configurations
    config_dict = {
        'msa_depth_base': [128],
        'msa_depth_addi': [0, 384],
        'blk_del': [False, True],
        'smpl_mthd': ['unif', 'topk', 'hybrid'],
        'pert_mthd': ['af2', 'legacy'],
        'use_tokens_feat': [False, True],
        'is_train': [False, True],
    }
    config_list = make_config_list(**config_dict)
    a3m_fpaths = [
        '/apdcephfs/share_1594716/jonathanwu/Datasets/RCSB-PDB-343k/a3m.files.bfd/4WHT_M.a3m',
        '/apdcephfs/share_1594716/jonathanwu/Datasets/RCSB-PDB-343k/a3m.files.uniref/4WHT_M.a3m',
    ]
    device = torch.device('cuda:0')

    # initialization
    tfold_init()

    # parse A3M files to obtain MSA tokens
    parser = A3mParser()
    alphabet = Alphabet.from_architecture('MSA Transformer')
    converter = alphabet.get_batch_converter()
    msa_data = []
    for a3m_fpath in a3m_fpaths:
        if len(msa_data) == 0:
            msa_data = parser.run(a3m_fpath)
        else:
            msa_data_addi = parser.run(a3m_fpath)
            msa_data.extend(msa_data_addi[1:])  # skip the first sequence
    _, _, msa_tokens_raw = converter([msa_data])
    tokens_full = msa_tokens_raw[:, :, 1:].to(device)  # 1 x K x L

    # test w/ <MsaSampler>
    for config in config_list:
        logging.info('=== configurations ===')
        for key, val in config.items():
            logging.info('> %s: %s', key, val)

        sampler = MsaSampler(**config)
        tokens_true, tokens_pert, tokens_mask, tokens_feat, tokens_addi = sampler.run(tokens_full)
        logging.info('tokens_true: %s / %s', tokens_true.shape, tokens_true.dtype)
        logging.info('tokens_pert: %s / %s', tokens_pert.shape, tokens_pert.dtype)
        logging.info('tokens_mask: %s / %s', tokens_mask.shape, tokens_mask.dtype)
        if tokens_feat is not None:
            logging.info('tokens_feat: %s / %s', tokens_feat.shape, tokens_feat.dtype)
        if tokens_addi is not None:
            logging.info('tokens_addi: %s / %s', tokens_addi.shape, tokens_addi.dtype)

        compare_tokens(tokens_true[0].cpu(), tokens_pert[0].cpu(), tokens_mask[0].cpu())


if __name__ == '__main__':
    main()
