"""The network for confidence head predictions"""

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

from tfold.modules.af2_smod import compute_ptmscore
from tfold.modules.evoformer import EvoformerStackSS


class LinearNoBiasThenOuterSum(nn.Module):
    """linear and outer sum, transform single repr -> pairwise pattern throughout this architecture."""

    def __init__(self, dim, dim_out):
        super().__init__()
        dim_out = dim_out
        self.proj = nn.Linear(dim, dim_out * 2, bias=False)

    def forward(self, t):
        single_i, single_j = self.proj(t).chunk(2, dim=-1)
        out = torch.einsum('b i d, b j d -> b i j d', single_i, single_j)
        return out


class ConfidenceHead(nn.Module):
    """ Algorithm 31 """

    def __init__(
        self,
        n_dims_inpt,
        n_dims_sfea=384,
        n_dims_pfea=128,
        n_lyrs=4,
        cal_pae=False,
    ):
        super().__init__()

        self.n_bins_dist = 18
        self.dist_min = 3.375
        self.dist_max = 21.375
        self.bin_wid = (self.dist_max - self.dist_min) / self.n_bins_dist

        self.n_bins_lddt = 50
        self.bin_vals_lddt = (torch.arange(self.n_bins_lddt) + 0.5) / self.n_bins_lddt

        self.n_bins_pde = 64
        self.n_bins_pae = 64

        # pre-norm
        self.norm_s = nn.LayerNorm(n_dims_sfea)
        self.norm_p = nn.LayerNorm(n_dims_pfea)

        self.cal_pae = cal_pae
        self.dist_bin_pairwise_embed = nn.Embedding(self.n_bins_dist, n_dims_pfea)
        self.sinpt_to_pfea = LinearNoBiasThenOuterSum(n_dims_inpt, n_dims_pfea)

        # evoformer stack
        self.evoformer_stack = EvoformerStackSS(
            c_s=n_dims_sfea,
            c_z=n_dims_pfea,
            num_layers=n_lyrs,
        )

        if self.cal_pae:
            self.to_pae_logits = nn.Linear(n_dims_pfea, self.n_bins_pae, bias=False)

        self.to_pde_logits = nn.Linear(n_dims_pfea, self.n_bins_pde, bias=False)

        self.to_plddt_logits = nn.Linear(n_dims_sfea, self.n_bins_lddt, bias=False)

    def enable_activation_checkpoint(self, enabled=True):
        """Enable the activation_checkpoint."""

        self.evoformer_stack.enable_activation_checkpoint(enabled)

    def forward(
            self,
            inpt,
            sfea_tns,
            pfea_tns,
            cord_tns,
            asym_id=None,
            chunk_size=None,
    ):
        """Perform the forward pass

        Args:
        * sfea_tns: B x L x D_s
        * pfea_tns: B x L x L x D_p
        * cord_tns: B x L x 3
        """

        # initialization
        device = sfea_tns.device
        dtype = sfea_tns.dtype
        conf_logts = {}

        # pre-norm, which not use in AF3
        sfea_tns = self.norm_s(sfea_tns)
        pfea_tns = self.norm_p(pfea_tns)

        # combine the input to pfea_tns
        pfea_tns = pfea_tns + self.sinpt_to_pfea(inpt)

        # interatomic distances - embed and add to pairwise
        dist_mat = torch.cdist(cord_tns, cord_tns, p=2)

        # calculate update terms for pair features
        idxs_mat = torch.clip(torch.floor(
            (dist_mat - self.dist_min) / self.bin_wid).to(torch.int64), 0, self.n_bins_dist - 1)
        pfea_tns = pfea_tns + self.dist_bin_pairwise_embed(idxs_mat).to(dtype)

        # evoformer stack
        sfea_tns_updt, pfea_tns_updt = self.evoformer_stack(
            sfea_tns,
            pfea_tns,
            chunk_size
        )
        sfea_tns = sfea_tns + sfea_tns_updt
        pfea_tns = pfea_tns + pfea_tns_updt

        # pLDDT
        self.bin_vals_lddt = self.bin_vals_lddt.to(dtype).to(device)
        plddt_logits = self.to_plddt_logits(sfea_tns)
        conf_logts['plddt'] = plddt_logits
        plddt_res = torch.sum(self.bin_vals_lddt.view(1, 1, -1) * F.softmax(plddt_logits, dim=-1), dim=2)
        plddt_chn = torch.mean(plddt_res, dim=1)

        # PAE, (pTM, ipTM), only incorporate pae at some stage of training
        pae_logits = None
        tmsc_dict = {}
        if self.cal_pae:
            pae_logits = self.to_pae_logits(pfea_tns)
            conf_logts['pae'] = pae_logits
            ptm = torch.stack(
                [compute_ptmscore(pae_logits[i]) for i in range(pae_logits.shape[0])])
            tmsc_dict['ptm'] = ptm.detach()

            if torch.count_nonzero(asym_id) > 0:
                iptm = torch.stack(
                    [compute_ptmscore(pae_logits[i], asym_id=asym_id) for i in range(pae_logits.shape[0])])
                tmsc_dict['iptm'] = iptm.detach()
                tmsc_dict['ranking_confidence'] = 0.8 * tmsc_dict['iptm'] + 0.2 * tmsc_dict['ptm']

        # PDE
        symmetric_pairwise_repr = pfea_tns + rearrange(pfea_tns, 'b i j d -> b j i d')
        pde_logits = self.to_pde_logits(symmetric_pairwise_repr)
        conf_logts['pde'] = pde_logits

        # pack all the confidence logit and prediction into a dict
        conf_metric = {
            'plddt-r': plddt_res.detach(),  # L
            'plddt-c': plddt_chn.detach(),  # scalar
            **tmsc_dict
        }

        # return all logits
        return conf_logts, conf_metric
