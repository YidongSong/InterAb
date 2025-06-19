"""The network for predicting lDDT-Ca scores."""

import torch
from torch import nn


class PLddtNet(nn.Module):
    """The network for predicting lDDT-Ca scores."""

    def __init__(self, n_dims_sfea=384):
        """Constructor function."""

        super().__init__()

        # setup hyper-parameters
        self.n_dims_sfea = n_dims_sfea

        # additional configurations
        self.n_bins_lddt = 50  # number of bins for pLDDT-Ca predictions
        self.bin_vals = (torch.arange(self.n_bins_lddt) + 0.5) / self.n_bins_lddt

        # per-residue lDDT-Ca predictions
        self.net = nn.ModuleDict()
        self.net['lddt'] = nn.Sequential(
            nn.LayerNorm(self.n_dims_sfea),
            nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
            nn.ReLU(),
            nn.Linear(self.n_dims_sfea, self.n_dims_sfea),
            nn.ReLU(),
            nn.Linear(self.n_dims_sfea, self.n_bins_lddt),
        )
        self.net['sfmx'] = nn.Softmax(dim=2)


    def forward(self, sfea_tns):
        """Perform the forward pass.

        Args:
        * sfea_tns: single features of size N x L x D_s

        Returns:
        * plddt_dict: dict of pLDDT predictions
        """

        # initialization
        dtype = sfea_tns.dtype
        device = sfea_tns.device

        # convert <self.bin_vals> into the correct data type & device
        self.bin_vals = self.bin_vals.to(dtype).to(device)

        # predict per-residue & full-chain lDDT-Ca scores
        logt_tns = self.net['lddt'](sfea_tns)
        plddt_res = torch.sum(self.bin_vals.view(1, 1, -1) * self.net['sfmx'](logt_tns), dim=2)
        plddt_chn = torch.mean(plddt_res, dim=1)

        # pack all the pLDDT predictions into a dict
        plddt_dict = {
            'logit': logt_tns[0],  # L x 50
            'plddt-r': plddt_res[0],  # L
            'plddt-c': plddt_chn[0],  # scalar
        }

        return plddt_dict
