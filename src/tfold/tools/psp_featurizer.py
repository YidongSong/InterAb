# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import random

import numpy as np
import torch
import torch.nn as nn

from tfold.tools import AtomMapper
from tfold.tools import compute_ptmscore
from tfold.utils import tensor_tree_map
from tfold.third_parties.tfold_released.model.arch.alphafold import AlphaFold
from tfold.third_parties.tfold_released.config import CfgNode
from tfold.third_parties.tfold_released.protein import data_transform


def template_feats_placeholder():
    return {
        'template_aatype': np.array([], np.int64),
        'template_all_atom_masks': np.zeros([], np.float32),
        'template_all_atom_positions': np.zeros([], np.float32),
        'template_domain_names': np.zeros([], np.object),
    }


class PspFeaturizer(nn.Module):
    """The pre-trained protein structure prediction (PSP) feature extractor."""

    def __init__(
            self,
            config
    ):
        super().__init__()
        self.config = config
        self.num_recycles = config.data.common.max_recycling_iters
        self.atom_mapper = AtomMapper()
        self.model = AlphaFold(self.config)
        self.model.eval()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def n_dims_sfea(self):
        """Get the number of dimension in single features."""
        return self.config.model.input_embedder.c_m

    @property
    def n_dims_pfea(self):
        """Get the number of dimension in pair features."""
        return self.config.model.input_embedder.c_z

    @classmethod
    def restore(cls, path):
        state = torch.load(path, map_location='cpu')
        config = CfgNode(state['config'])
        model = cls(config).eval()
        model.model.load_state_dict(state['model'])
        logging.info('model weights restored from %s', path)
        return model

    def forward(self, msa, deletion_matrix, num_recycles=None):
        num_recycles = self.num_recycles if num_recycles is None else num_recycles
        aaseq = msa[0]
        feats = data_transform.np_make_sequence_features(aaseq)
        feats.update(data_transform.np_make_msa_features([msa], [deletion_matrix]))
        feats.update(template_feats_placeholder())
        process_feature_dict = data_transform.np_example_to_predict_features(feats, config=self.config.data)

        process_feature_dict = {
            k: torch.as_tensor(v, device=self.device)
            for k, v in process_feature_dict.items()
        }

        if self.config.data.common.max_recycling_iters > 0:
            for item in process_feature_dict:
                num_iters = random.sample(range(num_recycles + 1), num_recycles + 1)
                process_feature_dict[item] = process_feature_dict[item][..., num_iters]

        with torch.no_grad():
            result, ptm = self._forward_impl(process_feature_dict)

        for i in range(num_recycles + 1):
            result[i][2] = self.atom_mapper.run(aaseq, result[i][2], frmt_src='n37', frmt_dst='n14-tf')
            result[i][3] = self.atom_mapper.run(aaseq, result[i][3], frmt_src='n37', frmt_dst='n14-tf')

        result_dict = {
            'mfea': torch.stack([result[i][0] for i in range(num_recycles + 1)], dim=0),
            'pfea': torch.stack([result[i][1] for i in range(num_recycles + 1)], dim=0),
            'cord': torch.stack([result[i][2] for i in range(num_recycles + 1)], dim=0),
            'cmsk': torch.stack([result[i][3] for i in range(num_recycles + 1)], dim=0),
        }
        torch.cuda.empty_cache()
        return result_dict, ptm

    def _forward_impl(self, batch):
        # Initialize recycling embeddings
        m_1_prev, z_prev, x_prev = None, None, None
        prevs = [m_1_prev, z_prev, x_prev]
        result = []
        # Main recycling loop
        num_iters = batch['aatype'].shape[-1]
        for cycle_no in range(num_iters):
            # Select the features for the current recycling cycle
            fetch_cur_batch = lambda t: t[..., cycle_no]  # noqa
            feats = tensor_tree_map(fetch_cur_batch, batch)
            # Enable grad iff we're training and it's the final recycling layer
            is_final_iter = cycle_no == (num_iters - 1)
            with torch.no_grad():
                # Run the next iteration of the model
                outputs, m_1_prev, z_prev, x_prev = self.model.iteration(feats, prevs)
                feat_tns = [
                    outputs['msa'].clone(),
                    outputs['pair'].clone(),
                    outputs['final_atom_positions'].clone(),
                    outputs['final_atom_mask'].clone(),
                ]
                result.append(feat_tns)
                if not is_final_iter:
                    del outputs
                    prevs = [m_1_prev, z_prev, x_prev]
                    del m_1_prev, z_prev, x_prev

        outputs.update(self.model.aux_heads(outputs))
        ptm = None
        if 'tm_logits' in outputs:
            ptm = compute_ptmscore(outputs['tm_logits']).item()

        return result, ptm
