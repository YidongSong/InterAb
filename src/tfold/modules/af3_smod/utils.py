import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange, repeat, reduce, einsum, pack, unpack

from tfold.modules.common import LayerNorm
from tfold.utils import pad_at_dim
from tfold.utils import slice_at_dim


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def swish(x):
    return x * torch.sigmoid(x)


def concat_previous_window(t, dim_seq, dim_window):
    t = pad_at_dim(t, (1, 0), dim=dim_seq, value=0.)
    t = torch.cat((
        slice_at_dim(t, slice(None, -1), dim=dim_seq),
        slice_at_dim(t, slice(1, None), dim=dim_seq),
    ), dim=dim_window)

    return t


def pad_to_multiple(t, multiple, dim=-1, value=0.):
    seq_len = t.shape[dim]
    padding_needed = (multiple - (seq_len % multiple)) % multiple

    if padding_needed == 0:
        return t

    return pad_at_dim(t, (0, padding_needed), dim=dim, value=value)


def pad_and_window(t, window_size):
    t = pad_to_multiple(t, window_size, dim=1)
    t = rearrange(t, 'b (n w) ... -> b n w ...', w=window_size)
    return t


def full_pairwise_repr_to_windowed(pfea_tns, window_size):

    seq_len, device = pfea_tns.shape[-2], pfea_tns.device

    padding_needed = (window_size - (seq_len % window_size)) % window_size
    pfea_tns = F.pad(pfea_tns, (0, 0, 0, padding_needed, 0, padding_needed), value=0.)
    pfea_tns = rearrange(pfea_tns, '... (i w1) (j w2) d -> ... i j w1 w2 d', w1=window_size, w2=window_size)

    pfea_tns = concat_previous_window(pfea_tns, dim_seq=-4, dim_window=-2)

    # get the diagonal
    n = torch.arange(pfea_tns.shape[-4], device=device)
    pfea_tns = pfea_tns[..., n, n, :, :, :]

    return pfea_tns


def full_attn_bias_to_windowed(attn_bias, window_size):

    attn_bias = rearrange(attn_bias, '... -> ... 1')
    attn_bias = full_pairwise_repr_to_windowed(attn_bias, window_size=window_size)
    return rearrange(attn_bias, '... 1 -> ...')


def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def unpack_one(to_unpack, unpack_pattern=None):
        unpacked, = unpack(to_unpack, ps, pattern if unpack_pattern is None else unpack_pattern)
        return unpacked

    return packed, unpack_one


# SwiGLU
class SwiGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        x = F.silu(gates) * x

        return x


class AdaptiveLayerNorm(nn.Module):
    """ Algorithm 26 """

    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()

        self.norm = LayerNorm(dim, elementwise_affine=False, bias=False)
        self.norm_cond = LayerNorm(dim_cond, bias=False)

        self.to_gamma = nn.Sequential(
            nn.Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = nn.Linear(dim_cond, dim, bias=False)

    def forward(
        self,
        x,
        cond
    ):
        x = self.norm(x)
        cond = self.norm_cond(cond)
        x = x * self.to_gamma(cond) + self.to_beta(cond)

        return x


class CentreRandomAugmentation(nn.Module):
    """ Algorithm 19 """

    def __init__(self, trans_scale=1.0):
        super().__init__()
        self.trans_scale = trans_scale
        self.register_buffer('dummy', torch.tensor(0), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def forward(self, cord_tns, mask=None):
        """
        cord_tns: coordinates to be augmented
        """
        batch_size = cord_tns.shape[0]

        if mask is not None:
            cord_tns = torch.where(mask.unsqueeze(-1) == 1, cord_tns, 0.)
            num = reduce(cord_tns, 'b n c -> b c', 'sum')
            den = reduce(mask.to(cord_tns.dtype), 'b n -> b', 'sum')
            cord_tns_mean = (num / den.unsqueeze(-1).clamp(min=1.)).unsqueeze(1)  # b, 1, 3
        else:
            cord_tns_mean = cord_tns.mean(dim=1, keepdim=True)

        # Center the coordinates
        centered_cord = cord_tns - cord_tns_mean

        # Generate random rotation matrix
        rotation_matrix = self._random_rotation_matrix(batch_size)

        # Generate random translation vector
        translation_vector = self._random_translation_vector(batch_size)
        translation_vector = rearrange(translation_vector, 'b c -> b 1 c')

        # Apply rotation and translation
        augmented_coords = einsum(centered_cord, rotation_matrix, 'b n i, b j i -> b n j') + translation_vector

        return augmented_coords

    def _random_rotation_matrix(self, batch_size):

        # Generate random rotation angles
        angles = torch.rand((batch_size, 3), device=self.device) * 2 * torch.pi

        # Compute sine and cosine of angles
        sin_angles = torch.sin(angles)
        cos_angles = torch.cos(angles)

        # Construct rotation matrix
        eye = torch.eye(3, device=self.device)
        rotation_matrix = repeat(eye, 'i j -> b i j', b=batch_size).clone()

        rotation_matrix[:, 0, 0] = cos_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 0, 1] = cos_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] - \
            sin_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 0, 2] = cos_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] + \
            sin_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 1, 0] = sin_angles[:, 0] * cos_angles[:, 1]
        rotation_matrix[:, 1, 1] = sin_angles[:, 0] * sin_angles[:, 1] * sin_angles[:, 2] + \
            cos_angles[:, 0] * cos_angles[:, 2]
        rotation_matrix[:, 1, 2] = sin_angles[:, 0] * sin_angles[:, 1] * cos_angles[:, 2] - \
            cos_angles[:, 0] * sin_angles[:, 2]
        rotation_matrix[:, 2, 0] = -sin_angles[:, 1]
        rotation_matrix[:, 2, 1] = cos_angles[:, 1] * sin_angles[:, 2]
        rotation_matrix[:, 2, 2] = cos_angles[:, 1] * cos_angles[:, 2]

        return rotation_matrix

    def _random_translation_vector(self, batch_size):
        # Generate random translation vector
        translation_vector = torch.randn((batch_size, 3), device=self.device) * self.trans_scale
        return translation_vector


def atom_ref_pos_to_atompair_inputs(atom_ref_pos, atom_ref_space_uid):

    assert atom_ref_pos.shape[0] == atom_ref_space_uid.shape[0]

    # line 2
    pairwise_rel_pos = atom_ref_pos.unsqueeze(1) - atom_ref_pos.unsqueeze(2)

    # line 3
    same_ref_space_mask = atom_ref_space_uid.unsqueeze(1) == atom_ref_space_uid.unsqueeze(2)

    # line 5 - pairwise inverse squared distance
    atom_inv_square_dist = (1 + pairwise_rel_pos.norm(dim=-1, p=2) ** 2) ** -1

    # concat all into atompair_inputs for projection into atompair_feats within Alphafold3
    atompair_inputs = torch.cat(
        (pairwise_rel_pos, atom_inv_square_dist.unsqueeze(-1), same_ref_space_mask.float().unsqueeze(-1)), dim=-1)

    # mask out
    atompair_inputs = torch.where(
        same_ref_space_mask.unsqueeze(-1), atompair_inputs, torch.zeros_like(atompair_inputs)
    )

    # return
    return atompair_inputs


def lens_to_mask(lens, max_len=None):
    device = lens.device
    if max_len is None:
        max_len = lens.amax()
    arange = torch.arange(max_len, device=device)

    return arange[None, :] < lens.unsqueeze(-1)


def exclusive_cumsum(t, dim=-1):
    return t.cumsum(dim=dim) - t


def repeat_consecutive_with_lens(feats, lens):
    device, dtype = feats.device, feats.dtype

    batch, seq, *dims = feats.shape

    # get mask from lens
    mask = lens_to_mask(lens)

    # derive arange
    window_size = mask.shape[-1]
    arange = torch.arange(window_size, device=device)

    offsets = exclusive_cumsum(lens)
    indices = arange[None, None, :] + offsets.unsqueeze(-1)

    # create output tensor + a sink position on the very right (index max_len)
    total_lens = lens.clamp(min=0).sum(dim=-1)
    output_mask = lens_to_mask(total_lens)

    max_len = total_lens.amax()

    output_indices = torch.zeros((batch, max_len + 1), device=device, dtype=torch.long)

    indices = indices.masked_fill(~mask, max_len)   # scatter to sink position for padding
    indices = rearrange(indices, 'b n w -> b (n w)')

    # scatter
    seq_arange = torch.arange(seq, device=device)
    seq_arange = repeat(seq_arange, 'n -> b (n w)', b=batch, w=window_size)

    output_indices = output_indices.scatter(1, indices, seq_arange)

    # remove sink
    output_indices = output_indices[:, :-1]

    # gather
    feats, unpack_one = pack_one(feats, 'b n *')
    output_indices = repeat(output_indices, 'b m -> b m d', d=feats.shape[-1])
    output = feats.gather(1, output_indices)
    output = unpack_one(output)

    # set output padding value
    mask_value = torch.tensor(False, device=device) if dtype == torch.bool else torch.tensor(0, device=device)
    if len(output.shape) == 3:
        output = torch.where(output_mask.unsqueeze(-1), output, mask_value)
    elif len(output.shape) == 2:
        output = torch.where(output_mask, output, mask_value)
    else:
        raise ValueError("Unsupported shape for feats")

    return output


def calculate_bin_centers(breaks):
    """Calculate bin centers from bin edges."""
    step = breaks[1] - breaks[0]
    bin_centers = breaks + step / 2
    last_bin_center = breaks[-1] + step
    bin_centers = torch.concat([bin_centers, last_bin_center.unsqueeze(0)])

    return bin_centers


def compute_pde(logits, tok_repr_atm_mask):
    """Compute PDE from logits."""
    pde_breaks = torch.arance(0, 31.5, 0.5)
    bin_centers = calculate_bin_centers(pde_breaks)
    probs = F.softmax(logits, dim=-1)
    pde = einsum(probs, bin_centers, "b i j pde, pde -> b i j")
    mask = tok_repr_atm_mask.unsqueeze(1) & tok_repr_atm_mask.unsqueeze(0)
    pde = pde * mask

    return pde
