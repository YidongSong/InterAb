"""The multiple sequence alignments (MSA) sampler.

Alternative configurations
    1. Block deletion
        a. Disabled
        b. Enabled
    2. MSA sampling
        a. Uniform sampling (training) + top-K selection (inference)
        b. Uniform sampling (training + inference)
    3. Random perturbation
        a. 15% masks
        b. 10.5% masks + 1.5% uniform + 1.5% profile + 1.5% original
    4. MSA tokens' additional features
        a. Disabled
        b. Enabled
    5. Additional MSA tokens
        a. Disabled
        b. Enabled
"""

import random

import torch
from torch import nn

from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.third_parties.esm.data import Alphabet


class MsaSampler():  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """The multiple sequence alignments (MSA) sampler."""

    def __init__(
            self,
            msa_depth_base=128,  # base MSA depth
            msa_depth_addi=0,  # additional MSA depth
            blk_del=True,  # whether to enable MSA block deletion (train-only)
            smpl_mthd='unif',  # MSA sampling method (choices: 'unif' / 'topk' / 'hybrid')
            pert_mthd='af2',  # MSA perturbation method (choices: 'af2' / 'legacy')
            pert_infer=False,  # whether to perturb MSA tokens during inference
            use_tokens_feat=True,  # whether to use MSA tokens' additional features
            is_train=True,  # whether the current MSA sampler is used for model training
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        # setup configurations
        self.msa_depth_base = msa_depth_base
        self.msa_depth_addi = msa_depth_addi
        self.blk_del = blk_del
        self.smpl_mthd = smpl_mthd
        self.pert_mthd = pert_mthd
        self.pert_infer = pert_infer
        self.use_tokens_feat = use_tokens_feat
        self.is_train = is_train

        # additional configurations
        self.alphabet = Alphabet.from_architecture('MSA Transformer')
        self.alphabet_size = len(self.alphabet.all_toks)  # alphabet_size = 33
        self.n_seqs_max = 4096  # (train-only) maximal number of sequences in the MSA data
        self.n_seqs_per_chk = 1024  # number of sequences per chunk for assigning sequences
        self.mask_prob = 0.15


    @torch.no_grad()
    def run(self, tokens_full):
        """Run the MSA sampling process.

        Args:
        * tokens_full: full MSA tokens of size 1 x K x L

        Returns:
        * tokens_true: ground-truth MSA tokens of cluster centers of size 1 x K_c x L
        * tokens_pert: perturbed MSA tokens of cluster centers of size 1 x K_c x L
        * tokens_mask: perturb-or-not masks of MSA tokens of cluster centers of size 1 x K_c x L
        * tokens_feat: MSA tokens' additional features of size 1 x K_c x L x D
        * tokens_addi: additional MSA tokens of size 1 x K_a x L

        Note:
        * The batch dimension can be omitted, since BS must be 1 during model training & inference.
        * If <use_tokens_feat> is False, then <tokens_feat> will be None.
        * If <msa_depth_addi> equals 0, then <tokens_addi> will be None.
        """

        # recursively run the MSA sampling process w/ batch dimension removed for simplicity
        if tokens_full.ndim == 3:
            assert tokens_full.shape[0] == 1
            rtn_vals = self.run(tokens_full[0])
            rtn_vals = [x.unsqueeze(dim=0) if x is not None else x for x in rtn_vals]
            return tuple(rtn_vals)

        # (train-only) MSA block deletion
        if self.is_train and self.blk_del:
            tokens_full = self.__apply_blk_del(tokens_full)

        # randomly sample cluster centers (and additional MSA tokens)
        tokens_true, tokens_addi = self.__smpl_tokens(tokens_full)

        # randomly perturb MSA tokens of cluster centers
        if self.is_train or self.pert_infer:
            if self.pert_mthd == 'af2':
                tokens_pert, tokens_mask = self.__pert_tokens_af2(tokens_full, tokens_true)
            elif self.pert_mthd == 'legacy':
                tokens_pert, tokens_mask = self.__pert_tokens_legacy(tokens_true)
            else:
                raise ValueError(f'unrecognized MSA perturbation method: {self.pert_mthd}')
        else:  # inference mode w/o random perturbation
            tokens_pert = tokens_true.detach().clone()
            tokens_mask = torch.ones_like(tokens_true, dtype=torch.int8)

        # build MSA tokens' additional features
        tokens_feat = None
        if self.use_tokens_feat:
            tokens_feat = self.__get_tokens_feat(tokens_full, tokens_true)

        return tokens_true, tokens_pert, tokens_mask, tokens_feat, tokens_addi


    def __apply_blk_del(self, tokens):
        """Apply MSA block deletion."""

        # configurations
        n_blk_del_ops = 5  # number of MSA block deletion operations
        blk_del_ratio = 0.3

        # apply MSA block deletion
        n_seqs = tokens.shape[0]
        if n_seqs > 1:  # no MSA block deletion for orphan sequences
            blk_size = int(blk_del_ratio * n_seqs + 0.5)
            mask_vec = torch.ones((n_seqs), dtype=torch.int8)  # 1: kept / 0: deleted
            for _ in range(n_blk_del_ops):
                idx_beg = random.randrange(1, n_seqs)
                idx_end = min(n_seqs, idx_beg + blk_size)
                mask_vec[idx_beg:idx_end] = 0  # mark as deleted
            idxs_nnz = torch.nonzero(mask_vec)[:, 0].tolist()
            tokens = torch.stack([tokens[x] for x in idxs_nnz], dim=0)

        return tokens


    def __smpl_tokens(self, tokens_full):
        """Randomly sample cluster centers (and additional MSA tokens)."""

        # initialization
        n_seqs_full = tokens_full.shape[0]

        # randomly sample MSA tokens as cluster centers
        if n_seqs_full <= self.msa_depth_base:
            idxs_seq_clst = list(range(n_seqs_full))
        else:
            if self.smpl_mthd == 'unif':
                idxs_seq_clst = [0] + random.sample(range(1, n_seqs_full), self.msa_depth_base - 1)
            elif self.smpl_mthd == 'topk':
                idxs_seq_clst = list(range(self.msa_depth_base))
            elif self.smpl_mthd == 'hybrid':
                n_seqs_clst_1st = self.msa_depth_base // 2
                n_seqs_clst_2nd = self.msa_depth_base - n_seqs_clst_1st
                idxs_seq_clst = list(range(n_seqs_clst_1st)) + \
                    random.sample(range(n_seqs_clst_1st, n_seqs_full), n_seqs_clst_2nd)
            else:
                raise ValueError(f'unrecognized MSA sampling method: {self.smpl_mthd}')

        # randomly sample MSA tokens as additional inputs
        idxs_seq_uncl = list(set(range(n_seqs_full)) - set(idxs_seq_clst))
        if len(idxs_seq_uncl) <= self.msa_depth_addi:
            idxs_seq_addi = idxs_seq_uncl
        else:
            idxs_seq_addi = random.sample(idxs_seq_uncl, self.msa_depth_addi)

        # gather MSA tokens based on selected indices
        tokens_clst = torch.stack([tokens_full[x] for x in idxs_seq_clst], dim=0)
        tokens_addi = None if len(idxs_seq_addi) == 0 else \
            torch.stack([tokens_full[x] for x in idxs_seq_addi], dim=0)

        return tokens_clst, tokens_addi


    def __pert_tokens_af2(self, tokens_full, tokens_true):  # pylint: disable=too-many-locals
        """Randomly perturb MSA tokens in the AF2 mode."""

        # configurations
        n_seqs_spec = 7
        n_seqs_unif = 1
        n_seqs_prof = 1
        n_seqs_orig = 1
        n_seqs_cand = n_seqs_spec + n_seqs_unif + n_seqs_prof + n_seqs_orig

        # initialization
        device = tokens_true.device
        n_clsts, n_resds = tokens_true.shape  # ground-truth tokens for MSA cluster centers

        # build a categorical distribution to generate amino-acids sampled uniformly
        prob_mat = torch.zeros((n_resds, self.alphabet_size), dtype=torch.float32)
        for resd_name in RESD_NAMES_1C:
            idx = self.alphabet.get_idx(resd_name)
            prob_mat[:, idx] = 1.0 / len(RESD_NAMES_1C)
        distr = torch.distributions.categorical.Categorical(probs=prob_mat)
        tokens_unif = distr.sample(torch.Size([n_clsts])).to(device)

        # build a categorical distribution to generate amino-acids sampled from the MSA profile
        onht_tns = nn.functional.one_hot(tokens_full, num_classes=self.alphabet_size)
        prob_mat = torch.mean(onht_tns.to(torch.float32), dim=0)
        distr = torch.distributions.categorical.Categorical(probs=prob_mat)
        tokens_prof = distr.sample(torch.Size([n_clsts]))

        # build candidate MSA tokens for random perturbation
        idx_mat = torch.randint(n_seqs_cand, (n_clsts, n_resds), device=device)
        tokens_spec = self.alphabet.mask_idx * torch.ones_like(tokens_true)
        tokens_cand = torch.where(
            torch.lt(idx_mat, n_seqs_spec), tokens_spec, torch.where(
            torch.lt(idx_mat, n_seqs_spec + n_seqs_unif), tokens_unif, torch.where(
            torch.lt(idx_mat, n_seqs_spec + n_seqs_unif + n_seqs_prof), tokens_prof, tokens_true,
        )))

        # randomly perturb amino-acids
        tokens_mask = (torch.rand_like(tokens_true, dtype=torch.float32) < self.mask_prob)
        tokens_pert = torch.where(tokens_mask, tokens_cand, tokens_true)
        tokens_mask = tokens_mask.to(torch.int8)

        return tokens_pert, tokens_mask


    def __pert_tokens_legacy(self, tokens_true):
        """Randomly perturb MSA tokens in the legacy mode."""

        # apply random masks on MSA tokens
        tokens_mask = (torch.rand_like(tokens_true, dtype=torch.float32) < self.mask_prob)
        tokens_pert = torch.where(tokens_mask, self.alphabet.mask_idx, tokens_true)
        tokens_mask = tokens_mask.to(torch.int8)

        return tokens_pert, tokens_mask


    def __get_tokens_feat(self, tokens_full, tokens_clst):
        """Get MSA tokens' additional features."""

        def _tvec2prob(token_vecs):
            token_mat = torch.stack(token_vecs, dim=0)
            onht_tns = nn.functional.one_hot(token_mat, num_classes=self.alphabet_size)
            prob_mat = torch.mean(onht_tns.to(torch.float32), dim=0)
            return prob_mat

        # limit the maximal number of MSA records for efficiency
        n_seqs = tokens_full.shape[0]
        if self.is_train and (n_seqs > self.n_seqs_max):
            idxs = [0] + random.sample(range(1, n_seqs), self.n_seqs_max - 1)
            tokens_full = torch.stack([tokens_full[x] for x in idxs], dim=0)

        # get MSA tokens' additional features
        n_seqs_clst = tokens_clst.shape[0]
        cidx_vec, dist_vec = self.__assign_tokens(tokens_full, tokens_clst)
        token_vecs_list = [[tokens_clst[x]] for x in range(n_seqs_clst)]
        idxs_seq_full_nnz = torch.nonzero(dist_vec)[:, 0].tolist()  # skip duplicated sequences
        for idx_seq_full in idxs_seq_full_nnz:
            idx_seq_clst = cidx_vec[idx_seq_full]
            token_vecs_list[idx_seq_clst].append(tokens_full[idx_seq_full])
        prob_mat_list = [_tvec2prob(x) for x in token_vecs_list]
        tokens_feat = torch.stack(prob_mat_list, dim=0)

        return tokens_feat


    def __assign_tokens(self, tokens_full, tokens_clst):
        """Assign MSA tokens to their nearest clusters."""

        cidx_vec_list = []
        dist_vec_list = []
        n_seqs = tokens_full.shape[0]
        for idx_seq_beg in range(0, n_seqs, self.n_seqs_per_chk):
            idx_seq_end = min(idx_seq_beg + self.n_seqs_per_chk, n_seqs)
            tokens_sel = tokens_full[idx_seq_beg:idx_seq_end]
            dist_mat = torch.sum(tokens_sel.unsqueeze(dim=1) != tokens_clst.unsqueeze(dim=0), dim=2)
            cidx_vec = torch.argmin(dist_mat, dim=1)
            dist_vec = torch.gather(dist_mat, 1, cidx_vec.unsqueeze(dim=1))
            cidx_vec_list.append(cidx_vec)
            dist_vec_list.append(dist_vec)
        cidx_vec = torch.cat(cidx_vec_list, dim=0).cpu()
        dist_vec = torch.cat(dist_vec_list, dim=0).cpu()

        return cidx_vec, dist_vec
