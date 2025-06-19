"""The pre-trained language model (PLM) featurizer.

List of available PLMs:
* ProtTrans-XLNet
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ProtTrans-models/prot_xlnet
* ProtTrans-T5-XL
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ProtTrans-models/prot_t5_xl_uniref50
* AntiBERTy
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/AntiBERTy-models/antiberty
* ESM-1b
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ESM-models/esm1b_t33_650M_UR50S.pt
* ESM-1v
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ESM-models/esm1v_t33_650M_UR90S_1.pt
* ESM-2 (650M)
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ESM-models/esm2_t33_650M_UR50D.pt
* ESM-2 (3B)
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ESM-models/esm2_t36_3B_UR50D.pt
* ESM-2 (15B)
/apdcephfs/share_1594716/jonathanwu/Pre-trained.Models/ESM-models/esm2_t48_15B_UR50D.pt
"""

import os
import re

import torch
from transformers import BertModel, BertTokenizer
from transformers import XLNetModel, XLNetTokenizer
from transformers import T5EncoderModel, T5Tokenizer
from esm.pretrained import load_model_and_alphabet_core


def restore_model(path):
    """Restore the pre-trained language model."""

    name = os.path.basename(path).split('.')[0]
    if name == 'prot_xlnet':
        model = XLNetModel.from_pretrained(path)
        tokenizer = XLNetTokenizer.from_pretrained(path, do_lower_case=False)
    elif name == 'prot_t5_xl_uniref50':
        model = T5EncoderModel.from_pretrained(path)
        tokenizer = T5Tokenizer.from_pretrained(path, do_lower_case=False)
    elif name == 'antiberty':
        model = BertModel.from_pretrained(path)
        tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
    elif name.startswith('esm'):
        model_data = torch.load(path, map_location='cpu')
        model, alphabet = load_model_and_alphabet_core(name, model_data, regression_data=None)
        tokenizer = alphabet.get_batch_converter()
    else:
        raise ValueError(f'unrecognized pre-trained language model name: {name}')

    return model, tokenizer


def get_config(name, tokenizer):
    """Get model-specific configurations."""

    # get the beginning index, ending index, and mask token index
    if name == 'prot_xlnet':
        pad_mode = 'head'  # add padding tokens before the main sequence
        idx_beg, idx_end = 0, -2  # no prepended token & 2 appended tokens
        mask_token = '<mask>'  # mask token used by XLNet
        mask_idx = tokenizer._convert_token_to_id(mask_token)  # pylint: disable=protected-access
    elif name == 'prot_t5_xl_uniref50':
        pad_mode = 'tail'  # add padding tokens after the main sequence
        idx_beg, idx_end = 0, -1  # no prepended token & 1 appended token
        mask_token = '<extra_id_0>'  # mask token used by T5
        mask_idx = tokenizer._convert_token_to_id(mask_token)  # pylint: disable=protected-access
    elif name == 'antiberty':
        pad_mode = 'tail'  # add padding tokens after the main sequence
        idx_beg, idx_end = 1, -1  # 1 prepended token & 1 appended token
        mask_token = '[MASK]'
        mask_idx = tokenizer._convert_token_to_id(mask_token)  # pylint: disable=protected-access
    elif name.startswith('esm'):
        pad_mode = 'tail'  # add padding tokens after the main sequence
        idx_beg, idx_end = 1, -1  # 1 prepended token & 1 appended token
        mask_idx = tokenizer.alphabet.mask_idx
    else:
        raise ValueError(f'unrecognized pre-trained language model name: {name}')

    return pad_mode, idx_beg, idx_end, mask_idx


class PlmFeaturizer():
    """The pre-trained language model (PLM) featurizer."""

    def __init__(self, path, device, plm_n_attn_lyrs=-1, debug=False):
        """Constructor function."""

        # setup configurations
        self.path = path
        self.device = device
        self.plm_n_attn_lyrs = plm_n_attn_lyrs
        self.debug = debug

        # restore the pre-trained language model
        self.name = os.path.basename(self.path).split('.')[0]
        self.model, self.tokenizer = restore_model(self.path)
        self.model.eval()  # put into the evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False  # do not update model parameters
        self.model = self.model.to(self.device)

        # additional configurations
        self.pad_mode, self.idx_beg, self.idx_end, self.mask_idx = \
            get_config(self.name, self.tokenizer)
        assert self.pad_mode in ['head', 'tail'], f'unrecognized padding mode: {self.pad_mode}'


    @property
    def n_dims_sfea(self):
        """Get the number of dimensions in single features."""

        if self.name in ['prot_xlnet', 'prot_t5_xl_uniref50']:
            n_dims = self.model.config.d_model
        elif self.name == 'antiberty':
            n_dims = self.model.config.hidden_size
        elif self.name.startswith('esm1'):
            n_dims = self.model.args.embed_dim
        elif self.name.startswith('esm2'):
            n_dims = self.model.embed_dim
        else:
            raise ValueError(f'unrecognized pre-trained language model name: {self.name}')

        return n_dims


    @property
    def n_lyrs_attn(self):
        """Get the number of layers for attention extraction."""

        if self.plm_n_attn_lyrs != -1:
            n_lyrs_attn = self.plm_n_attn_lyrs
        else:  # otherwise, determine <n_lyrs_attn> by the PLM being used
            if self.name == 'prot_xlnet':
                n_lyrs_attn = self.model.config.n_layer
            elif self.name == 'prot_t5_xl_uniref50':
                n_lyrs_attn = self.model.config.num_layers
            elif self.name.startswith('esm1'):
                n_lyrs_attn = self.model.args.layers
            elif self.name.startswith('esm2'):
                n_lyrs_attn = self.model.num_layers
            elif self.name.startswith('antiberty'):
                n_lyrs_attn = self.model.config.num_hidden_layers
            else:
                raise ValueError(f'unrecognized pre-trained language model name: {self.name}')

        return n_lyrs_attn


    @property
    def n_dims_pfea(self):
        """Get the number of dimensions in pair features."""

        # determine the number of dimensions in each layer's attention weights
        if self.name == 'prot_xlnet':
            n_dims_per_lyr = self.model.config.n_head
        elif self.name == 'prot_t5_xl_uniref50':
            n_dims_per_lyr = self.model.config.num_heads
        elif self.name == 'antiberty':
            n_dims_per_lyr = self.model.config.num_attention_heads
        elif self.name.startswith('esm1'):
            n_dims_per_lyr = self.model.args.attention_heads
        elif self.name.startswith('esm2'):
            n_dims_per_lyr = self.model.attention_heads
        else:
            raise ValueError(f'unrecognized pre-trained language model name: {self.name}')

        # multiply by the number of layers for extracting attention weights
        n_dims = n_dims_per_lyr * self.n_lyrs_attn

        return n_dims


    def run_single(self, aa_seq, mask_prob=0.0):
        """Extract PLM embeddings as single & pair features - single sequence input.

        Args:
        * aa_seq: amino-acid sequence of length L
        * mask_prob: (optional) how likely amino-acid tokens are randomly masked out

        Returns:
        * sfea_mat: single features of size L x D_s
        * pfea_tns: pair features of size L x L x D_p
        * mask_vec: (optional) masked-or-not indicators of size L
        """

        # determine whether random masks should be applied
        apply_mask = (mask_prob > 0.0)

        # extract PLM embeddings w/ batch-mode implementation
        if not apply_mask:
            sfea_mat_list, pfea_tns_list = self.run([aa_seq])
            mask_vec_list = [None]
        else:
            sfea_mat_list, pfea_tns_list, mask_vec_list = self.run([aa_seq], mask_prob)

        # extract PLM embeddings (and mask indicators)
        sfea_mat = sfea_mat_list[0]
        pfea_tns = pfea_tns_list[0]
        mask_vec = mask_vec_list[0]

        return (sfea_mat, pfea_tns, mask_vec) if apply_mask else (sfea_mat, pfea_tns)


    def run(self, aa_seqs, n_resds_list=None, mask_prob=0.0):
        """Extract PLM embeddings as single & pair features.

        Args:
        * aa_seqs: list of amino-acid sequences, each of length L_i
        * n_resd_list
        * mask_prob: (optional) how likely amino-acid tokens are randomly masked out

        Returns:
        * sfea_mat_list: list of single features, each of size L_i x D_s
        * pfea_tns_list: list of pair features, each of size L_i x L_i x D_p
        * mask_vec_list: (optional) list of masked-or-not indicators, each of size L_i
        """

        # put the model into the evaluation mode
        self.model.eval()

        # replace non-standard amino-acids w/ the unknown token
        if n_resds_list is None:
            n_resds_list = [len(x) for x in aa_seqs]
        tok_unk = '[UNK]' if self.name == 'antiberty' else '<unk>'
        aa_seqs = [re.sub(r'[BJOUXZ]', tok_unk, ' '.join(x)) for x in aa_seqs]

        # extract PLM embeddings
        if self.name in ['prot_xlnet', 'prot_t5_xl_uniref50', 'antiberty']:
            # prepare inputs
            inputs = self.tokenizer.batch_encode_plus(
                aa_seqs, padding=True, return_tensors='pt', return_special_tokens_mask=True)
            tokn_mat = inputs.input_ids.to(self.device)
            n_tokns = tokn_mat.shape[1]

            # get special token masks (1: special token)
            if self.debug:
                smsk_mat = inputs.special_tokens_mask.to(self.device)

            # add random masks to amino-acid tokens
            tokn_mat, mask_mat = self.__add_masks(tokn_mat, n_resds_list, mask_prob)

            # perform the forward pass w/ pre-trained language model
            with torch.no_grad():
                outputs = self.model(tokn_mat, output_attentions=True)

            # get initial single & pair features
            sfea_tns = outputs.last_hidden_state  # N x L' x D_s (L' != L)
            pfea_tns = torch.cat(outputs.attentions[-self.n_lyrs_attn:], dim=1).permute(0, 2, 3, 1)  # N x L' x L' x D_p
        elif self.name.startswith('esm'):  # esm-1b / esm-1v / esm-2 (650m / 3b / 15b)
            # prepare inputs
            _, _, tokn_mat = self.tokenizer([('', x) for x in aa_seqs])
            tokn_mat = tokn_mat.to(self.device)
            n_tokns = tokn_mat.shape[1]

            # manually build special token masks (1: special token)
            if self.debug:
                smsk_mat = torch.zeros_like(tokn_mat, dtype=torch.int8)
                for tok in self.tokenizer.alphabet.all_special_tokens:
                    if tok == tok_unk:
                        continue
                    smsk_mat += torch.eq(tokn_mat, self.tokenizer.alphabet.get_idx(tok)).to(torch.int8)

            # add random masks to amino-acid tokens
            tokn_mat, mask_mat = self.__add_masks(tokn_mat, n_resds_list, mask_prob)

            # perform the forward pass w/ pre-trained language model
            n_layer = self.model.num_layers
            with torch.no_grad():
                outputs = self.model(tokn_mat, repr_layers=[n_layer], need_head_weights=True)

            # get initial single & pair features
            sfea_tns = outputs['representations'][n_layer]  # N x L' x D_s (L' != L)
            pfea_tns = outputs['attentions'][:, -self.n_lyrs_attn:].permute(0, 3, 4, 1, 2)
            pfea_tns = pfea_tns.view(*pfea_tns.shape[:3], -1)  # N x L' x L' x D_p
        else:
            raise ValueError(f'unrecognized pre-trained language model name: {self.name}')

        # validate <pad_mode>, <idx_beg>, and <idx_end>
        if self.debug:
            for idx, n_resds in enumerate(n_resds_list):
                if self.pad_mode == 'head':
                    idx_end = n_tokns + self.idx_end
                    idx_beg = idx_end - n_resds
                else:
                    idx_beg = self.idx_beg
                    idx_end = idx_beg + n_resds
                assert torch.all(torch.eq(smsk_mat[idx, :idx_beg], 1))
                assert torch.all(torch.eq(smsk_mat[idx, idx_beg:idx_end], 0))
                assert torch.all(torch.eq(smsk_mat[idx, idx_end:], 1))

        # remove prepended & appended special tokens
        sfea_mat_list = []  # L_i x D_s
        pfea_tns_list = []  # L_i x L_i x D_p
        mask_vec_list = []  # L_i
        for idx, n_resds in enumerate(n_resds_list):
            # determine starting & ending indices for the current sequence
            if self.pad_mode == 'head':
                idx_end = n_tokns + self.idx_end
                idx_beg = idx_end - n_resds
            else:
                idx_beg = self.idx_beg
                idx_end = idx_beg + n_resds

            # extract single & pair features
            sfea_mat_list.append(sfea_tns[idx, idx_beg:idx_end])
            pfea_tns_list.append(pfea_tns[idx, idx_beg:idx_end, idx_beg:idx_end])
            if mask_mat is not None:
                mask_vec_list.append(mask_mat[idx, idx_beg:idx_end])

        if mask_mat is None:
            return sfea_mat_list, pfea_tns_list

        return sfea_mat_list, pfea_tns_list, mask_vec_list


    def __add_masks(self, tokn_mat, n_resds_list, mask_prob):
        """Add random masks to amino-acid tokens."""

        if mask_prob == 0.0:
            return tokn_mat, None

        # determine which amino-acid tokens to be perturbed
        mask_mat = (torch.rand_like(tokn_mat, dtype=torch.float32) < mask_prob)
        for idx, n_resds in enumerate(n_resds_list):
            mask_mat[idx, :self.idx_beg] = 0  # do no perturb prepended special tokens
            mask_mat[idx, self.idx_beg + n_resds:] = 0  # do no perturb appended special tokens

        # add random masks to amino-acid tokens
        tokn_mat = torch.where(mask_mat, self.mask_idx, tokn_mat)

        return tokn_mat, mask_mat
