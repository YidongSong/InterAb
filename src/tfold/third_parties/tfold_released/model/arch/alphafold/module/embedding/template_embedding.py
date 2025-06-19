# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Data: 2023/12/22 11:14
# Author: chenchenqin
from functools import partial
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from tfold.third_parties.tfold_released.model.layer import LayerNorm, DropoutRowwise, DropoutColumnwise, Linear
from tfold.third_parties.tfold_released.model.module.attention import (
    GatedMultiheadAttention as Attention,
    TriangleAttentionStartingNode, TriangleAttentionEndingNode,
    TriangleMultiplicationOutgoing, TriangleMultiplicationIncoming
)
from tfold.third_parties.tfold_released.model.module.evoformer.evoformer_msa import Transition
from tfold.third_parties.tfold_released.model.utils import checkpoint_blocks, chunk_layer
from tfold.third_parties.tfold_released.protein import residue_constants as rc
from tfold.third_parties.tfold_released.protein.data_transform import make_pseudo_beta
from tfold.third_parties.tfold_released.transform import compute_dihedral_angle
from tfold.third_parties.tfold_released.transform.affine import Rigid
from tfold.third_parties.tfold_released.utils.tensor import batched_gather, permute_final_dims, tensor_tree_map, dict_multimap


class TemplatePointwiseAttention(nn.Module):
    """
    Implements Algorithm 17.
    """

    def __init__(self,
                 c_t=64,
                 c_z=128,
                 num_heads=4,
                 inf=1e5):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_z:
                Pair embedding channel dimension
        """
        super(TemplatePointwiseAttention, self).__init__()
        self.c_t = c_t
        self.c_z = c_z
        self.num_heads = num_heads
        self.inf = inf
        self.mha = Attention(
            c_q=self.c_z,
            c_k=self.c_t,
            c_v=self.c_t,
            num_heads=self.num_heads,
            gating=False
        )

    def _chunk(self,
               z: torch.Tensor,
               t: torch.Tensor,
               biases: List[torch.Tensor],
               chunk_size: int,
               ) -> torch.Tensor:
        mha_inputs = {
            "q_x": z,
            "kv_x": t,
            "biases": biases,
        }
        return chunk_layer(
            self.mha,
            mha_inputs,
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self,
                t: torch.Tensor,
                z: torch.Tensor,
                template_mask: Optional[torch.Tensor] = None,
                chunk_size: Optional[int] = 256
                ) -> torch.Tensor:
        """
        Note that this module suffers greatly from a small chunk size

        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            z:
                [*, N_res, N_res, C_t] pair embedding
            template_mask: [*, N_templ] template mask
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """

        # [*, N_res, N_res, 1, C_z]
        z = z.unsqueeze(-2)
        # [*, N_res, N_res, N_temp, C_t]
        t = permute_final_dims(t, (1, 2, 0, 3))
        # [*, N_res, N_res, 1, C_z]
        biases = []
        if template_mask is not None:
            bias = self.inf * (template_mask[..., None, None, None, None, :] - 1)
            biases.append(bias)

        if chunk_size is not None and not self.training:
            z = self._chunk(z, t, biases, chunk_size)
        else:
            z = self.mha(q_x=z, kv_x=t, biases=biases)

        # [*, N_res, N_res, C_z]
        z = z.squeeze(-2)

        return z


class TemplatePairBlock(nn.Module):
    def __init__(
            self,
            c_t: int,
            c_hidden_tri_mul: int,
            num_heads: int = 4,
            pair_transition_n: int = 2,
            dropout_rate: float = 0.0,
            inf: float = 1e5
    ):
        super(TemplatePairBlock, self).__init__()
        self.c_t = c_t
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.num_heads = num_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf
        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)
        self.tri_att_start = TriangleAttentionStartingNode(
            self.c_t,
            self.num_heads,
            inf=inf,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            self.c_t,
            self.num_heads,
            inf=inf,
        )
        self.tri_mul_out = TriangleMultiplicationOutgoing(
            self.c_t,
            self.c_hidden_tri_mul
        )
        self.tri_mul_in = TriangleMultiplicationIncoming(
            self.c_t,
            self.c_hidden_tri_mul
        )
        self.pair_transition = Transition(self.c_t, self.pair_transition_n)

    def forward(self,
                z: torch.Tensor,
                mask: torch.Tensor = None,
                chunk_size: Optional[int] = None
                ):
        single_templates = [
            t.unsqueeze(-4) for t in torch.unbind(z, dim=-4)
        ]

        if mask is not None:
            single_templates_masks = [
                m.unsqueeze(-3) for m in torch.unbind(mask, dim=-3)
            ]
        else:
            single_templates_masks = [None, ] * len(single_templates)

        for i in range(len(single_templates)):
            single = single_templates[i]
            single_mask = single_templates_masks[i]
            single = single + self.dropout_row(
                self.tri_att_start(single, mask=single_mask)
            )
            single = single + self.dropout_col(self.tri_att_end(single, mask=single_mask))
            tmu_update = self.tri_mul_out(
                single,
                mask=single_mask
            )
            single = single + self.dropout_row(tmu_update)
            tmu_update = self.tri_mul_in(
                single,
                mask=single_mask
            )
            single = single + self.dropout_row(tmu_update)
            single = single + self.pair_transition(single, chunk_size=chunk_size)
            single_templates[i] = single

        z = torch.cat(single_templates, dim=-4)

        return z


class TemplatePairStack(nn.Module):
    """Implements Algorithm 16.

    Args:
        c_t:
            Template embedding channel dimension
        c_hidden_tri_att:
            Hidden dimension for triangular multiplication
        num_blocks:
            Number of blocks in the stack
        pair_transition_n:
            Scale of pair transition (Alg. 15) hidden dimension
        dropout_rate:
            Dropout rate used throughout the stack
        blocks_per_ckpt:
            Number of blocks per activation checkpoint. None disables
            activation checkpointing
    """

    def __init__(
            self,
            c_t,
            c_hidden_tri_mul=64,
            num_blocks=2,
            num_heads=4,
            pair_transition_n=2,
            dropout_rate=0.25,
            inf=1e9
    ):
        super(TemplatePairStack, self).__init__()
        self.blocks_per_ckpt = False
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = TemplatePairBlock(
                c_t=c_t,
                c_hidden_tri_mul=c_hidden_tri_mul,
                num_heads=num_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                inf=inf,
            )
            self.blocks.append(block)
        self.layer_norm = LayerNorm(c_t)
        self.activation_checkpoint = False

    def forward(
            self,
            t: torch.tensor,
            mask: torch.tensor = None,
            chunk_size: int = None
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """
        if mask is not None and mask.shape[-3] == 1:
            expand_idx = list(mask.shape)
            expand_idx[-3] = t.shape[-4]
            mask = mask.expand(*expand_idx)

        blocks = [
            partial(
                b,
                mask=mask,
                chunk_size=chunk_size
            )
            for b in self.blocks
        ]
        if chunk_size is not None:
            blocks = [partial(b, chunk_size=chunk_size) for b in blocks]

        t, = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            interval=self.activation_checkpoint if self.training else None,
        )

        t = self.layer_norm(t)

        return t


def get_chi_atom_indices(device: torch.device):
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
        A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
        in the order specified in rc.restypes + unknown residue type
        at the end. For chi angles which are not defined on the residue, the
        positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append(
                [rc.atom_order[atom] for atom in chi_angle]
            )
        for _ in range(4 - len(atom_indices)):
            atom_indices.append([0, 0, 0, 0])  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.
    return torch.tensor(chi_atom_indices, device=device)


def compute_chi_angles(
        positions: torch.Tensor,
        mask: torch.Tensor,
        aatype: torch.Tensor
):
    """Computes the chi angles given all atom positions and the amino acid type.

    Args:
        positions: [num_res, num_atoms, 3], with positions of atoms needed to calculate chi angles.
            Supports up to 1 batch dimension.
        mask:  [num_res, rc.atom_type_num] that masks which atom
            positions are set for each residue. If given, then the chi mask will be
            set to 1 for a chi angle only if the amino acid has that chi angle and all
            the chi atoms needed to calculate that chi angle are set. If not given
            (set to None), the chi mask will be set to 1 for a chi angle if the amino
            acid has that chi angle and whether the actual atoms needed to calculate
            it were set will be ignored.
        aatype: A tensor of shape [num_res] with amino acid type integer
            code (0 to 21). Supports up to 1 batch dimension.

    Returns:
        A tuple of tensors (chi_angles, mask), where both have shape
        [num_res, 4]. The mask masks out unused chi angles for amino acid
        types that have less than 4 chi angles. If atom_positions_mask is set, the
        chi mask will also mask out uncomputable chi angles.
    """

    # Don't assert on the num_res and batch dimensions as they might be unknown.
    assert positions.shape[-1] == rc.num_atom_types
    assert mask.shape[-1] == rc.num_atom_types
    no_batch_dims = len(aatype.shape) - 1

    # Compute the table of chi angle indices. Shape: [restypes, chis=4, atoms=4].
    chi_atom_indices = get_chi_atom_indices(aatype.device)

    # DISCREPANCY: DeepMind doesn't remove the gaps here. I don't know why
    # theirs works.
    aatype_gapless = torch.clamp(aatype, max=20)

    # Select atoms to compute chis. Shape: [*, num_res, chis=4, atoms=4].
    atom_indices = chi_atom_indices[aatype_gapless]
    # Gather atom positions. Shape: [num_res, chis=4, atoms=4, xyz=3].
    chi_angle_atoms = positions.map_tensor_fn(
        partial(
            batched_gather,
            inds=atom_indices,
            dim=-1,
            no_batch_dims=no_batch_dims + 1
        )
    )

    a, b, c, d = [chi_angle_atoms[..., i] for i in range(4)]

    chi_angles = compute_dihedral_angle(a, b, c, d)

    # Copy the chi angle mask, add the UNKNOWN residue. Shape: [restypes, 4].
    chi_angles_mask = list(rc.chi_angles_mask)
    chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
    chi_angles_mask = torch.tensor(chi_angles_mask, device=aatype.device)
    # Compute the chi angle mask. Shape [num_res, chis=4].
    chi_mask = chi_angles_mask[aatype_gapless]

    # The chi_mask is set to 1 only when all necessary chi angle atoms were set.
    # Gather the chi angle atoms mask. Shape: [num_res, chis=4, atoms=4].
    chi_angle_atoms_mask = batched_gather(
        mask,
        atom_indices,
        dim=-1,
        no_batch_dims=no_batch_dims + 1
    )
    # Check if all 4 chi angle atoms were set. Shape: [num_res, chis=4].
    chi_angle_atoms_mask = torch.prod(chi_angle_atoms_mask, dim=-1)
    chi_mask = chi_mask * chi_angle_atoms_mask.to(chi_angles.dtype)

    return chi_angles, chi_mask


class MultimerTemplateAngleEmbedding(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super(MultimerTemplateAngleEmbedding, self).__init__()
        self.linear_1 = Linear(c_in, c_out)
        self.linear_2 = Linear(c_out, c_out)
        self.relu = nn.ReLU()

    def forward(self,
                batch,
                atom_pos,
                aatype_one_hot,
                ):
        dtype = batch["template_all_atom_positions"].dtype
        template_chi_angles, template_chi_mask = (
            compute_chi_angles(
                atom_pos,
                batch["template_all_atom_mask"],
                batch["template_aatype"],
            )
        )
        template_features = torch.cat(
            [
                aatype_one_hot,
                torch.sin(template_chi_angles) * template_chi_mask,
                torch.cos(template_chi_angles) * template_chi_mask,
                template_chi_mask,
            ],
            dim=-1,
        ).to(dtype=dtype)

        template_mask = template_chi_mask[..., 0].to(dtype=dtype)

        template_activations = self.linear_1(template_features)
        template_activations = self.relu(template_activations)
        template_activations = self.linear_2(template_activations)

        out = {}
        out["template_single_embedding"] = template_activations
        out["template_mask"] = template_mask

        return out


class TemplateEmbedding(nn.Module):

    def __init__(self, config):
        self.config = config
        super(TemplateEmbedding, self).__init__()

        self.template_pair_stack = TemplatePairStack(
            **self.config["template_pair_stack"]
        )
        self.template_pointwise_att = TemplatePointwiseAttention(
            **self.config["template_pointwise_attention"]
        )

    def forward(self,
                batch,
                z,
                pair_mask=None,
                chunk_size=None):
        template_mask = batch.get("template_mask", None)
        if template_mask is not None:
            template_mask.to(dtype=z.dtype)

        t_pair_feats = batch["template_pair_feats"]  # [*, N, N, 88]
        t_pair = self.template_pair_embedder(t_pair_feats)
        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            t_pair,
            mask=pair_mask.unsqueeze(-3) if pair_mask is not None else None,
            chunk_size=chunk_size
        )
        # [*, N, N, C_z]
        t = self.template_pointwise_att(t, z, template_mask=template_mask)
        if template_mask is not None:
            t_mask = (torch.sum(template_mask, dim=-1) > 0).to(t.dtype)
            # Append singletons
            t_mask = t_mask.reshape(*t_mask.shape, *([1] * (len(t.shape) - len(t_mask.shape))))
            t = t * t_mask

        ret = {"template_pair_embedding": t}
        if self.template_angle_enabled:
            # [*, S_t, N, C_m]
            ret["template_angle_embedding"] = self.template_angle_embedder(batch["template_angle_feat"])

        return ret


class MultimerTemplatePairEmbedding(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_dgram: int,
                 c_aatype: int,
                 ):
        super(MultimerTemplatePairEmbedding, self).__init__()
        self.dgram_linear = Linear(c_dgram, c_out, init='relu')
        self.aatype_linear_1 = Linear(c_aatype, c_out, init='relu')
        self.aatype_linear_2 = Linear(c_aatype, c_out, init='relu')
        self.query_embedding_layer_norm = LayerNorm(c_in)
        self.query_embedding_linear = Linear(c_in, c_out, init='relu')
        self.pseudo_beta_mask_linear = Linear(1, c_out, init='relu')
        self.x_linear = Linear(1, c_out, init='relu')
        self.y_linear = Linear(1, c_out, init='relu')
        self.z_linear = Linear(1, c_out, init='relu')
        self.backbone_mask_linear = Linear(1, c_out, init='relu')

    def forward(self,
                template_dgram: torch.Tensor,
                aatype_one_hot: torch.Tensor,
                query_embedding: torch.Tensor,
                pseudo_beta_mask: torch.Tensor,
                backbone_mask: torch.Tensor,
                multichain_mask_2d: torch.Tensor,
                unit_vector: torch.Tensor,
                ) -> torch.Tensor:
        act = 0.
        pseudo_beta_mask_2d = (
                pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
        )
        pseudo_beta_mask_2d *= multichain_mask_2d
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act += self.dgram_linear(template_dgram)
        act += self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None])

        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act += self.aatype_linear_1(aatype_one_hot[..., None, :, :])
        act += self.aatype_linear_2(aatype_one_hot[..., None, :])

        backbone_mask_2d = (
                backbone_mask[..., None] * backbone_mask[..., None, :]
        )
        backbone_mask_2d *= multichain_mask_2d
        x, y, z = [(coord * backbone_mask_2d).to(dtype=query_embedding.dtype) for coord in unit_vector]
        act += self.x_linear(x[..., None])
        act += self.y_linear(y[..., None])
        act += self.z_linear(z[..., None])

        act += self.backbone_mask_linear(backbone_mask_2d[..., None].to(dtype=query_embedding.dtype))

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act += self.query_embedding_linear(query_embedding)

        return act


def dgram_from_positions(
        pos: torch.Tensor,
        min_bin: float = 3.25,
        max_bin: float = 50.75,
        no_bins: float = 39,
        inf: float = 1e8,
):
    dgram = torch.sum(
        (pos[..., None, :] - pos[..., None, :, :]) ** 2, dim=-1, keepdim=True
    )
    lower = torch.linspace(min_bin, max_bin, no_bins, device=pos.device) ** 2
    upper = torch.cat([lower[1:], lower.new_tensor([inf])], dim=-1)
    dgram = ((dgram > lower) * (dgram < upper)).type(dgram.dtype)

    return dgram


def make_backbone_affine(positions, mask: torch.Tensor):
    a = rc.atom_order['N']
    b = rc.atom_order['CA']
    c = rc.atom_order['C']
    rigid_mask = (mask[..., a] * mask[..., b] * mask[..., c])
    rigid = Rigid.make_transform_from_reference(positions[..., a], positions[..., b], positions[..., c])
    return rigid, rigid_mask


class MultimerTemplateEmbedding(nn.Module):
    def __init__(self, config):
        super(MultimerTemplateEmbedding, self).__init__()
        self.config = config
        self.template_angle_embedder = MultimerTemplateAngleEmbedding(
            **config["template_angle_embedder"],
        )
        self.template_pair_embedder = MultimerTemplatePairEmbedding(
            **config["template_pair_embedder"],
        )
        self.template_pair_stack = TemplatePairStack(
            **config["template_pair_stack"],
        )
        self.linear_t = Linear(config.c_t, config.c_z)
        self.relu = nn.ReLU()

    def forward(self,
                batch,
                z,
                padding_mask_2d,
                templ_dim,
                chunk_size=None
                ):
        asym_id = batch["asym_id"]
        multichain_mask_2d = (asym_id[..., None] == asym_id[..., None, :])
        template_embeds = []
        n_templ = batch["template_aatype"].shape[templ_dim]
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )

            single_template_embeds = {}
            template_positions, pseudo_beta_mask = make_pseudo_beta(
                single_template_feats["template_aatype"],
                single_template_feats["template_all_atom_positions"],
                single_template_feats["template_all_atom_mask"])

            template_dgram = dgram_from_positions(
                template_positions,
                inf=self.config.inf,
                **self.config.distogram,
            )
            aatype_one_hot = F.one_hot(
                single_template_feats["template_aatype"], 22,
            )
            raw_atom_pos = single_template_feats["template_all_atom_positions"]
            atom_pos = raw_atom_pos.to(dtype=torch.float32)
            rigid, backbone_mask = make_backbone_affine(
                atom_pos,
                single_template_feats["template_all_atom_mask"]
            )
            points = rigid.translation
            rigid_vec = rigid[..., None].inverse().apply_to_point(points)
            unit_vector = rigid_vec.normalized()

            pair_act = self.template_pair_embedder(
                template_dgram,
                aatype_one_hot,
                z,
                pseudo_beta_mask,
                backbone_mask,
                multichain_mask_2d,
                unit_vector,
            )

            single_template_embeds["template_pair_embedding"] = pair_act
            single_template_embeds.update(
                self.template_angle_embedder(
                    single_template_feats,
                    atom_pos,
                    aatype_one_hot,
                )
            )
            template_embeds.append(single_template_embeds)

        template_embeds = dict_multimap(
            partial(torch.cat, dim=templ_dim),
            template_embeds,
        )

        # [*, S_t, N, N, C_z]
        t = self.template_pair_stack(
            template_embeds["template_pair_embedding"],
            padding_mask_2d.unsqueeze(-3).to(dtype=z.dtype),
            chunk_size=chunk_size
        )
        # [*, N, N, C_z]
        t = torch.sum(t, dim=-4) / n_templ
        t = self.relu(t)
        t = self.linear_t(t)
        template_embeds["template_pair_embedding"] = t

        return template_embeds
