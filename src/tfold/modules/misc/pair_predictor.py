"""The inter-residue pairwise distance & orientation predictor."""

import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """The point-wise feed-forward network.

    Notes:
    * We follow Primer [1] to use squared ReLU (instead of ReLU) as the activation function.

    References:
    [1] So et al., Primer: Searching for Efficient Transformers for Language Modeling. NeurIPS '21.
    """

    def __init__(self, n_dims_in, n_dims_out=None):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_in = n_dims_in
        self.n_dims_out = n_dims_out if n_dims_out is not None else n_dims_in

        # additional configurations
        self.n_dims_hid = 4 * n_dims_in

        # build the network
        self.linear_in = nn.Linear(self.n_dims_in, self.n_dims_hid)
        self.actv = lambda x: nn.functional.relu(x) ** 2
        self.linear_out = nn.Linear(self.n_dims_hid, self.n_dims_out)

    def forward(self, x):
        """Perform the forward pass."""

        return self.linear_out(self.actv(self.linear_in(x)))


class PairPredictor(nn.Module):
    """The inter-residue pairwise distance & orientation predictor."""

    def __init__(self, n_dims_pfea=256, n_bins_list=None):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims_pfea = n_dims_pfea
        self.n_bins_list = n_bins_list if n_bins_list is not None else [37, 25, 25, 25]

        # build the network
        self.norm = nn.LayerNorm(self.n_dims_pfea)
        self.ffn_cb = FeedForwardNetwork(self.n_dims_pfea, self.n_bins_list[0])
        self.ffn_om = FeedForwardNetwork(self.n_dims_pfea, self.n_bins_list[1])
        self.ffn_th = FeedForwardNetwork(self.n_dims_pfea, self.n_bins_list[2])
        self.ffn_ph = FeedForwardNetwork(self.n_dims_pfea, self.n_bins_list[3])

    def forward(self, pfea_tns):
        """Perform the forward pass.

        Args:
        * pfea_tns: pair features of size N x L x L x c_z

        Returns:
        * logt_tns_cb: classification logits for CB-CB distance of size N x C_cb x L x L
        * logt_tns_om: classification logits for <omega> angles of size N x C_om x L x L
        * logt_tns_th: classification logits for <theta> angles of size N x C_th x L x L
        * logt_tns_ph: classification logits for <phi> angles of size N x C_ph x L x L
        """

        # make initial predictions w/ each FFN sub-module
        pfea_tns = self.norm(pfea_tns)
        logt_tns_cb = self.ffn_cb(pfea_tns)
        logt_tns_om = self.ffn_om(pfea_tns)
        logt_tns_th = self.ffn_th(pfea_tns)
        logt_tns_ph = self.ffn_ph(pfea_tns)

        # symmetrize predictions for CB-CB' distance and CA-CB-CB'-CA' angles
        logt_tns_cb = (logt_tns_cb + logt_tns_cb.permute(0, 2, 1, 3)) / 2.0
        logt_tns_om = (logt_tns_om + logt_tns_om.permute(0, 2, 1, 3)) / 2.0

        # move classification logits to the 2nd dimension
        logt_tns_cb = logt_tns_cb.permute(0, 3, 1, 2)
        logt_tns_om = logt_tns_om.permute(0, 3, 1, 2)
        logt_tns_th = logt_tns_th.permute(0, 3, 1, 2)
        logt_tns_ph = logt_tns_ph.permute(0, 3, 1, 2)

        return logt_tns_cb, logt_tns_om, logt_tns_th, logt_tns_ph
