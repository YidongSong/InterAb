"""The template sequential & pairwise block."""

from torch import nn
from openfold.model.primitives import Attention
from openfold.model.dropout import DropoutRowwise, DropoutColumnwise
from openfold.model.triangular_attention import (
    TriangleAttentionStartingNode,
    TriangleAttentionEndingNode,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
)
from tfold.modules.evoformer import Transition


class TemplatePairBlock(nn.Module):
    """The Template pair block with template-wise attention for template feature update"""

    def __init__(
        self,
        c_z: int,
        c_h_pair_mul: int = 128,
        c_h_pair_att: int = 32,
        n_heads_pair: int = 8,
        dropout_templ: float = 0.15,
        inf: float = 1e9,
        use_templ_attn: bool = False,
    ):
        """Constructor function."""
        super().__init__()

        # triangle attn update
        self.tri_mul_out = TriangleMultiplicationOutgoing(c_z, c_h_pair_mul)
        self.tri_mul_in = TriangleMultiplicationIncoming(c_z, c_h_pair_mul)
        self.tri_att_start = TriangleAttentionStartingNode(c_z, c_h_pair_att, n_heads_pair, inf=inf)
        self.tri_att_end = TriangleAttentionEndingNode(c_z, c_h_pair_att, n_heads_pair, inf=inf)
        # template-wise attn update
        self.use_templ_attn = use_templ_attn
        if use_templ_attn:
            self.templ_wise_attn = Attention(c_z, c_z, c_z, c_h_pair_att, n_heads_pair)
        # transition
        self.pair_trans = Transition(c_z)
        # dropout
        self.pair_dropout_row = DropoutRowwise(dropout_templ)
        self.pair_dropout_col = DropoutColumnwise(dropout_templ)

    def forward(self, tfea_tns, tpl_mask=None, res_mask=None, chunk_size=None):
        """Perform the forward pass.

        Args:
        * tfea_tns: template feature of size (N * T) x L x L x c_z
        * res_mask: residue mask of size (N x L)
        * tpl_mask: template mask of size (N x T)

        Returns:
        * tfeat_tns: update template feature of size (N * T) x L x L x c_z
        """

        # triangle attn update
        tfea_tns = tfea_tns + self.pair_dropout_row(self.tri_att_start(tfea_tns, chunk_size=chunk_size))
        tfea_tns = tfea_tns + self.pair_dropout_col(self.tri_att_end(tfea_tns, chunk_size=chunk_size))
        tfea_tns = tfea_tns + self.pair_dropout_row(self.tri_mul_out(tfea_tns))
        tfea_tns = tfea_tns + self.pair_dropout_row(self.tri_mul_in(tfea_tns))

        # template-wise attn update
        if self.use_templ_attn:
            n_batch, n_tpl = tpl_mask.shape
            n_length = tfea_tns.shape[1]

            tfea_tns = (
                tfea_tns.reshape(n_batch, n_tpl, n_length, n_length, -1)
                .permute(0, 2, 3, 1, 4)
                .contiguous()
                .reshape(n_batch, n_length * n_length, n_tpl, -1)
            )

            tfea_tns = tfea_tns + self.pair_dropout_row(
                self.templ_wise_attn(q_x=tfea_tns, kv_x=tfea_tns)
            )

            tfea_tns = (
                tfea_tns.reshape(n_batch, n_length, n_length, n_tpl, -1)
                .permute(0, 3, 1, 2, 4)
                .contiguous()
                .reshape(n_batch * n_tpl, n_length, n_length, -1)
            )

        # transition
        tfea_tns = tfea_tns + self.pair_trans(tfea_tns)

        return tfea_tns


class TemplateSeqBlock(nn.Module):
    """The Template seq block with resnet for template feature update"""
    def __init__(self, c_s):
        super().__init__()
        self.linear_in = nn.Linear(c_s, c_s)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(c_s, c_s)

    def forward(self, tfea_tns):
        """Perform the forward pass.

        Args:
        * tfea_tns: template sequential feature of size (N * T) x L x c_t

        Returns:
        * tfea_tns: update template sequential feature of size  (N * T) x L x c_z
        """
        t_initial = tfea_tns

        tfea_tns = self.relu(tfea_tns)
        tfea_tns = self.linear_in(tfea_tns)
        tfea_tns = self.relu(tfea_tns)
        tfea_tns = self.linear_out(tfea_tns)

        return t_initial + tfea_tns
