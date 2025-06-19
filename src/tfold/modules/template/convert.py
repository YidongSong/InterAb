"""Convert the template sequential & pairwise feature to sequential & pairwise feature"""

from torch import nn

from openfold.model.primitives import Attention
from openfold.model.template import TemplatePointwiseAttention


class SeqPointwiseAttention(nn.Module):
    """Multi-head pointwise for sequence feature"""

    def __init__(
            self,
            c_t: int,
            c_s: int,
            c_hidden: int = 64,
            n_heads: int = 4,
            inf: float = 1e9
    ):
        super().__init__()

        self.mha = Attention(
            c_t,
            c_s,
            c_s,
            c_hidden,
            n_heads,
            gating=False
        )
        self.inf = inf

    def forward(self, tfea_tns, sfea_tns, template_mask=None):
        """Perform the forward pass.

        Args:
        * tfea_tns: template feature of size N x L x T x c_t
        * sfea_tns: sequential feature of size N x L x c_s
        * template_mask: template mask of size N x T

        Returns:
        * pfea_tns: update sequential feature of size N x L x c_z

        """
        # [N, L, 1, C_s]
        sfea_tns = sfea_tns.unsqueeze(-2)

        # [N, L, T, C_t] for tfea_tns
        bias = self.inf * (template_mask[..., None, None, None, :] - 1)
        biases = [bias]

        sfea_tns = self.mha(q_x=sfea_tns, kv_x=tfea_tns, biases=biases)

        # [N, L, C_s]
        sfea_tns = sfea_tns.squeeze(-2)

        return sfea_tns


class Template2Seq(nn.Module):
    """Convert sequential template feature to sequential representation using pointwise-attention"""
    def __init__(self, c_t: int, c_s: int, c_hidden: int = 64, n_heads: int = 4, inf: float = 1e9):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c_s)
        self.template_pointwise_attn = SeqPointwiseAttention(c_t, c_s, c_hidden, n_heads, inf)

    def forward(self, tfea_tns, sfea_tns, tfea_mask):
        """Perform the forward pass.

        Args:
        * tfea_tns: template feature of size (N*T) x L x c_t
        * sfea_tns: sequential feature of size N x L x c_s
        * tfea_mask: template mask of size N x T

        Returns:
        * pfea_tns: update sequential feature of size N x L x c_z

        """
        n_batch, n_tpl = tfea_mask.shape
        n_length = tfea_tns.shape[1]

        tfea_tns = tfea_tns.reshape(n_batch, n_tpl, n_length, -1)
        tfea_tns = tfea_tns.permute(0, 2, 1, 3).contiguous()    # N, L, T, c_t

        sfea_tns_addi = self.template_pointwise_attn(tfea_tns, sfea_tns, template_mask=tfea_mask)

        sfea_tns = self.layer_norm(sfea_tns_addi) + sfea_tns

        return sfea_tns


class Template2Pair(nn.Module):
    """Convert pairwise template feature to pairwise representation using pointwise attention"""

    def __init__(self, c_t: int, c_z: int, c_hidden: int = 64, n_heads: int = 4, inf: float = 1e9):
        """Constructor function."""
        super().__init__()
        self.template_pointwise_attn = TemplatePointwiseAttention(c_t, c_z, c_hidden, n_heads, inf)
        self.layer_norm = nn.LayerNorm(c_z)

    def forward(self, tfea_tns, pfea_tns, tfea_mask, chunk_size=None):
        """Perform the forward pass.

        Args:
        * tfea_tns: template feature of size (N*T) x L x L x c_t
        * pfea_tns: pair feature of size N x L x L x c_z
        * tfea_mask: template mask of size N x T

        Returns:
        * pfea_tns: update pair feature of size N x L x L x c_z
        """
        n_batch, n_tpl = tfea_mask.shape
        n_length = tfea_tns.shape[1]

        tfea_tns = tfea_tns.reshape(n_batch, n_tpl, n_length, n_length, -1)

        pfea_tns_addi = self.template_pointwise_attn(tfea_tns, pfea_tns, chunk_size=chunk_size, template_mask=tfea_mask)

        pfea_tns = self.layer_norm(pfea_tns_addi) + pfea_tns

        return pfea_tns
