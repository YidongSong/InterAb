"""The Template Embedding model for template feature."""

import torch
from torch import nn

from tfold.tools import PosiEncoder


class TemplateSeqEmbedding(nn.Module):
    """Embedding sequential template with relative position encoding"""

    def __init__(
        self,
        c_t: int,
        c_s: int,
    ):
        super().__init__()

        self.posi_encoder = PosiEncoder(n_dims=32, pos_max=1024)
        self.linear_in = nn.Linear(c_t + 32, c_s)
        self.relu = nn.ReLU()
        self.linear_out = nn.Linear(c_s, c_s)

    def forward(self, tfea_tns, res_idxs=None):
        """Perform the forward pass.

        Args:
        * tfea_tns: template sequential features of size  N x T x L x c_t
        * res_idxs: residue index of size  L

        Returns:
        * tfea_tns: template sequential embedding of size  (N * T) x L x c_s
        """
        N, T, L, _ = tfea_tns.shape
        if res_idxs is None:
            res_idxs = torch.arange(L).to(tfea_tns.device)

        penc_tns = self.posi_encoder.run(res_idxs).unsqueeze(dim=0).unsqueeze(1).expand(N, T, -1, -1)
        tfea_tns = self.linear_in(torch.cat((tfea_tns, penc_tns), -1))
        tfea_tns = self.relu(tfea_tns)
        tfea_tns = self.linear_out(tfea_tns).reshape(N * T, L, -1)

        return tfea_tns


class TemplatePairEmbedding(nn.Module):
    """Embedding pair template with relative position encoding"""

    def __init__(
        self,
        c_t: int,
        c_z: int,
    ):
        super().__init__()
        self.linear_out = nn.Linear(c_t + 1, c_z)

    def forward(self, tfea_tns, res_idxs=None):
        """Perform the forward pass.

        Args:
        * tfea_tns: template pairwise features of size  N x T x L x L x c_t
        * res_idxs: residue index of size   N x L

        Returns:
        * tfea_tns: template pairwise embedding of size  (N * T) x L x L x c_z
        """
        N, T, L, _, _ = tfea_tns.shape
        if res_idxs is None:
            res_idxs = torch.arange(L).view(1, L).expand(N, -1).to(tfea_tns.device)
        seq_sep = torch.abs(res_idxs[:, :, None] - res_idxs[:, None, :]) + 1
        seq_sep = torch.log(seq_sep.float()).view(N, L, L, 1).unsqueeze(1).expand(-1, T, -1, -1, -1)

        tfea_tns = torch.cat((tfea_tns, seq_sep), -1)
        tfea_tns = self.linear_out(tfea_tns).reshape(N * T, L, L, -1)
        return tfea_tns
