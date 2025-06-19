"""The AlphaFold3 structure module."""

from torch import nn

from tfold.modules.af3_smod.diff_cond import SingleConditioning, PairwiseConditioning
from tfold.modules.af3_smod.atom_attn_encoder import AtomAttentionEncoder
from tfold.modules.af3_smod.diff_trans import DiffusionTransformer
from tfold.modules.af3_smod.utils import repeat_consecutive_with_lens
from tfold.utils import pad_or_slice_to


class DiffusionModule(nn.Module):
    """ Algorithm 20 """

    def __init__(
        self,
        n_dims_atom_inputs,
        n_dims_pfea_trunk,
        n_dims_penc=128,
        atoms_per_window=27,  # for atom sequence, take the approach of (batch, seq, atoms, ..), where atom dimension is set to the molecule or molecule with greatest number of atoms, the rest padded. atom_mask must be passed in - default to 27 for proteins, with tryptophan having 27 atoms # noqa
        n_dims_sfea=384,
        n_dims_pfea=128,
        n_dims_atom=128,
        n_dims_atompair=16,
        n_dims_fourier=256,
        n_dims_token=768,
        sigma_data=16,
        atom_encoder_depth=3,   # AtomAttentionEncoder
        atom_encoder_heads=4,
        token_transformer_depth=24,   # token level DiffusionTransformer
        token_transformer_heads=16,
        atom_decoder_depth=3,   # Decoder
        atom_decoder_heads=4,
        atom_encoder_kwargs: dict = dict(),
        atom_decoder_kwargs: dict = dict(),
        token_transformer_kwargs: dict = dict(),
    ):
        super().__init__()

        self.atoms_per_window = atoms_per_window
        self.sigma_data = sigma_data

        # conditioning
        self.single_conditioner = SingleConditioning(
            sigma_data=sigma_data,
            n_dims_sfea=n_dims_sfea,
            n_dims_fourier=n_dims_fourier,
        )
        self.pairwise_conditioner = PairwiseConditioning(
            n_dims_pfea_trunk=n_dims_pfea_trunk,
            n_dims_pfea=n_dims_pfea,
            n_dims_penc=n_dims_penc,
        )

        # atom attention encoder
        self.atom_attn_encoder = AtomAttentionEncoder(
            n_dims_atom_inputs=n_dims_atom_inputs,
            n_dims_atom=n_dims_atom,
            n_dims_atompair=n_dims_atompair,
            atoms_per_window=atoms_per_window,
            n_dims_token=n_dims_token,
            n_dims_sfea=n_dims_sfea,
            n_dims_pfea=n_dims_pfea,
            atom_transformer_blocks=atom_encoder_depth,
            atom_transformer_heads=atom_encoder_heads,
            **atom_encoder_kwargs,
        )

        # full self-attom on token level
        self.cond_tokens_with_cond_single = nn.Sequential(
            nn.LayerNorm(n_dims_sfea),
            nn.Linear(n_dims_sfea, n_dims_token, bias=False)
        )
        self.token_transformer = DiffusionTransformer(
            n_lyrs=token_transformer_depth,
            n_heads=token_transformer_heads,
            dim=n_dims_token,
            n_dims_cond=n_dims_sfea,
            n_dims_pfea=n_dims_pfea,
            **token_transformer_kwargs
        )
        self.attended_token_norm = nn.LayerNorm(n_dims_token)

        # atom attention decoder
        self.tokens_to_atom_decoder_input_cond = nn.Linear(n_dims_token, n_dims_atom, bias=False)
        self.atom_decoder = DiffusionTransformer(
            n_lyrs=atom_decoder_depth,
            n_heads=atom_decoder_heads,
            dim=n_dims_atom,
            n_dims_cond=n_dims_atom,
            n_dims_pfea=n_dims_atompair,
            attn_window_size=atoms_per_window,
            **atom_decoder_kwargs
        )
        self.atom_feat_to_atom_pos_update = nn.Sequential(
            nn.LayerNorm(n_dims_atom),
            nn.Linear(n_dims_atom, 3, bias=False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def enable_activation_checkpoint(self, enabled=True):
        """Enable the activation_checkpoint."""

        self.single_conditioner.enable_activation_checkpoint(enabled)
        self.pairwise_conditioner.enable_activation_checkpoint(enabled)
        self.atom_attn_encoder.enable_activation_checkpoint(enabled)
        self.token_transformer.enable_activation_checkpoint(enabled)
        self.atom_decoder.enable_activation_checkpoint(enabled)

    def forward(
        self,
        atom_tns,        # noised_atom_pos,
        times,
        atom_inputs,
        sfea_tns,
        sfea_tns_trunk,
        pfea_tns_trunk,
        penc_tns,
        molecule_atom_lens
    ):

        batch_size, seq_len = sfea_tns_trunk.shape[:2]

        # conditioning
        cond_sfea_tns = self.single_conditioner(
            times=times,
            inpt=sfea_tns,
            sfea_tns_trunk=sfea_tns_trunk
        )

        cond_pfea_tns = self.pairwise_conditioner(
            pfea_tns_trunk=pfea_tns_trunk,
            penc_tns=penc_tns
        )

        # Sequence-local Atom Attention and aggregation to coarse-grained tokens
        sfea_tns, afea_tns, atom_feat_cond, atompair_feat_cond = self.atom_attn_encoder(
            atom_inputs,
            atom_tns,
            sfea_tns_trunk,
            cond_pfea_tns
        )

        # Full self-attention on token level
        sfea_tns += self.cond_tokens_with_cond_single(cond_sfea_tns)

        sfea_tns = self.token_transformer(
            sfea_tns,
            sfea_tns=cond_sfea_tns,
            pfea_tns=cond_pfea_tns,
        )

        sfea_tns = self.attended_token_norm(sfea_tns)

        # Atom Decoder
        # Broadcast per-token activations to per-atom activations and add the skip connection
        sfea_tns = self.tokens_to_atom_decoder_input_cond(sfea_tns)
        sfea_tns = repeat_consecutive_with_lens(sfea_tns, molecule_atom_lens)
        sfea_tns = pad_or_slice_to(sfea_tns, length=afea_tns.shape[1], dim=1)
        sfea_tns += afea_tns

        # Cross attention transformer
        afea_tns = self.atom_decoder(
            sfea_tns,
            sfea_tns=atom_feat_cond,
            pfea_tns=atompair_feat_cond
        )

        # Map to positions update
        atom_tns = self.atom_feat_to_atom_pos_update(afea_tns)

        return atom_tns
