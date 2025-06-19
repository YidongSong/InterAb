"""The atom attention encoder module. Algorithm 5 of AF3"""


import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from tfold.utils import pad_or_slice_to
from tfold.modules.af3_smod.diff_trans import DiffusionTransformer
from tfold.modules.af3_smod.utils import repeat_consecutive_with_lens
from tfold.modules.af3_smod.utils import pack_one


class AtomAttentionEncoder(nn.Module):
    """Algorithm 5 """

    def __init__(
        self,
        n_dims_atom_inputs,
        n_dims_atom=128,
        n_dims_atompair=16,
        atoms_per_window=27,
        n_dims_token=384,
        n_dims_sfea=384,
        n_dims_pfea=128,
        atom_transformer_blocks=3,
        atom_transformer_heads=4,
        atom_transformer_kwargs: dict = dict(),
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window

        self.atom_single_conditioning = nn.Linear(n_dims_atom_inputs, n_dims_atom, bias=False)

        self.valid_mask_embedding = nn.Linear(1, n_dims_atompair, bias=False)

        self.offset_embedding = nn.Linear(3, n_dims_atompair, bias=False)
        self.inverse_squared_distances_embedding = nn.Linear(1, n_dims_atompair, bias=False)

        self.single_to_atom_feat_cond = nn.Sequential(
            nn.LayerNorm(n_dims_sfea),
            nn.Linear(n_dims_sfea, n_dims_atom, bias=False)
        )

        self.pairwise_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(n_dims_pfea),
            nn.Linear(n_dims_pfea, n_dims_atompair, bias=False)
        )

        self.atom_pos_to_atom_feat = nn.Linear(3, n_dims_atom, bias=False)

        self.atom_repr_to_atompair_feat_cond = nn.Sequential(
            nn.LayerNorm(n_dims_atom),
            nn.ReLU(),
            nn.Linear(n_dims_atom, n_dims_atompair * 2, bias=False),
        )

        self.atompair_feats_mlp = nn.Sequential(
            nn.Linear(n_dims_atompair, n_dims_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(n_dims_atompair, n_dims_atompair, bias=False),
            nn.ReLU(),
            nn.Linear(n_dims_atompair, n_dims_atompair, bias=False),
        )

        self.atom_transformer = DiffusionTransformer(
            n_lyrs=atom_transformer_blocks,
            n_heads=atom_transformer_heads,
            dim=n_dims_atom,
            n_dims_cond=n_dims_atom,
            n_dims_pfea=n_dims_atompair,
            attn_window_size=atoms_per_window,
            **atom_transformer_kwargs
        )

        self.atom_feats_to_pooled_token = AtomToTokenPooler(
            dim=n_dims_atom,
            dim_out=n_dims_token
        )

    def enable_activation_checkpoint(self, enabled=True):
        """Enable the activation_checkpoint."""

        self.atom_transformer.enable_activation_checkpoint(enabled)

    def forward(self, atom_inputs, nfea_tns, sfea_tns, pfea_tns):
        """Perform the forward pass.

        Args:
            * atom_inputs:
              - atom_feats  B x M x c
              - atom_ref_pos B x M x 3 (not always provided)
              - atom_ref_space_uid B x M
              - molecule_atom_lens B x N
            * nfea_tns: noise coordinate, B x M x 3
            * sfea_tns: sequence representation: B x N x c_s
            * pfea_tns: pairwise representation: B x N x N x c_z

        Returns:
            tfea_tns: token representation B x N x c_t
            afea_tns: atom representation B x M x c_a
            atom_feat_cond: B x M x c_a
            atompair_feat_cond: B x M x M x c_p
        """

        # Initialise
        device = sfea_tns.device
        dtype = nfea_tns.dtype
        batch_size, seq_len = sfea_tns.shape[:2]
        atom_seq_len = atom_inputs['atom_feats'].shape[1]

        # Algorithm 5 Line 1, create the atom single conditioning: embed per-atom meta data
        atom_feat_cond = self.atom_single_conditioning(atom_inputs['atom_feats'].to(dtype))

        # Algorithm 5 Line 2 to Line 6
        atom_ref_space_uid = atom_inputs['atom_ref_space_uid']
        same_ref_space_mask = (atom_ref_space_uid.unsqueeze(1) == atom_ref_space_uid.unsqueeze(2)).to(dtype)
        atom_ref_pos = atom_inputs['atom_ref_pos'].to(atom_feat_cond.dtype)
        pairwise_rel_pos = atom_ref_pos.unsqueeze(1) - atom_ref_pos.unsqueeze(2)
        atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim=-1, p=2) ** 2) ** -1

        # Embed pairwise inverse squared distances, and the valid mask
        atompair_feat_cond = self.offset_embedding(pairwise_rel_pos) * same_ref_space_mask.unsqueeze(-1)
        atompair_feat_cond += self.inverse_squared_distances_embedding(
            atom_inv_square_dist.unsqueeze(-1)) * same_ref_space_mask.unsqueeze(-1)
        atompair_feat_cond += self.valid_mask_embedding(same_ref_space_mask.unsqueeze(-1))

        # Initialise the atom single representation as the single conditioning
        afea_tns = atom_feat_cond

        # Broadcast the single and pair embedding from the trunk
        # Line 9, first transformer token_sfea to atom_sfea
        sfea_tns = self.single_to_atom_feat_cond(sfea_tns)
        sfea_tns = repeat_consecutive_with_lens(sfea_tns, atom_inputs['molecule_atom_lens'])
        sfea_tns = pad_or_slice_to(sfea_tns, length=atom_feat_cond.shape[1], dim=1)
        atom_feat_cond += sfea_tns

        # Line 10
        pfea_tns = self.pairwise_to_atompair_feat_cond(pfea_tns)
        indices = torch.arange(seq_len, device=device)
        indices = repeat(indices, 'n -> b n', b=batch_size)
        indices = repeat_consecutive_with_lens(indices, atom_inputs['molecule_atom_lens'])
        indices = pad_or_slice_to(indices, atom_seq_len, dim=-1)

        row_indices = col_indices = indices
        row_indices = rearrange(row_indices, 'b n -> b n 1')
        col_indices = rearrange(col_indices, 'b n -> b 1 n')
        row_indices, col_indices = torch.broadcast_tensors(row_indices, col_indices)

        row_indices, unpack_one = pack_one(row_indices, 'b *')
        col_indices, _ = pack_one(col_indices, 'b *')
        rowcol_indices = col_indices + row_indices * pfea_tns.shape[2]
        rowcol_indices = repeat(rowcol_indices, 'b rc -> b rc dap', dap=pfea_tns.shape[-1])
        pfea_tns, _ = pack_one(pfea_tns, 'b * dap')
        pfea_tns = pfea_tns.gather(1, rowcol_indices)
        pfea_tns = unpack_one(pfea_tns, 'b * dap')
        atompair_feat_cond += pfea_tns

        # Line 11, add the noisy position
        afea_tns += self.atom_pos_to_atom_feat(nfea_tns)

        # Line 13, Add the combined single conditioning to the pair representation.
        atom_repr_cond = self.atom_repr_to_atompair_feat_cond(atom_feat_cond)
        atom_repr_cond_row, atom_repr_cond_col = atom_repr_cond.chunk(2, dim=-1)
        atompair_feat_cond += atom_repr_cond_row.unsqueeze(-2)
        atompair_feat_cond += atom_repr_cond_col.unsqueeze(-3)

        # Line14, Run a small MLP on the pair activations.
        atompair_feat_cond = self.atompair_feats_mlp(atompair_feat_cond) + atompair_feat_cond

        # Cross attention transformer.
        afea_tns = self.atom_transformer(afea_tns, atom_feat_cond, atompair_feat_cond)

        # Aggregate per-atom representation to per-token representation
        sfea_tns = self.atom_feats_to_pooled_token(
            atom_feats=afea_tns,
            molecule_atom_lens=atom_inputs['molecule_atom_lens']
        )

        return sfea_tns, afea_tns, atom_feat_cond, atompair_feat_cond


class AtomToTokenPooler(nn.Module):
    """Aggregate per-atom representation to per-token representation"""

    def __init__(self, dim, dim_out=None):
        super().__init__()
        dim_out = dim if dim_out is None else dim_out

        self.proj = nn.Sequential(
            nn.Linear(dim, dim_out, bias=False),
            nn.ReLU()
        )

    def forward(self, atom_feats, molecule_atom_lens):
        """Perform the forward pass.

        Args:
        atom_feats: B x M x c
        molecule_atom_lens: B x N

        Returns:
        token_feats: B x N x c
        """

        atom_feats = self.proj(atom_feats)

        atom_len = atom_feats.shape[1]
        token_mask = molecule_atom_lens > 0

        assert (molecule_atom_lens <= atom_len).all(), \
            'one of the lengths given exceeds the total sequence length of the features passed in'

        cumsum_feats = atom_feats.cumsum(dim=1)
        cumsum_feats = F.pad(cumsum_feats, (0, 0, 1, 0), value=0)

        cumsum_indices = molecule_atom_lens.cumsum(dim=1)
        cumsum_indices = F.pad(cumsum_indices, (1, 0), value=0)

        cumsum_indices = repeat(cumsum_indices, 'b n -> b n d', d=cumsum_feats.shape[-1])
        sel_cumsum = cumsum_feats.gather(-2, cumsum_indices)

        # subtract cumsum at one index from the previous one
        summed = sel_cumsum[:, 1:] - sel_cumsum[:, :-1]

        token_feats = summed / molecule_atom_lens.clamp(min=1).unsqueeze(-1)
        token_feats = torch.where(token_mask.unsqueeze(-1), token_feats, torch.zeros_like(token_feats))

        return token_feats
