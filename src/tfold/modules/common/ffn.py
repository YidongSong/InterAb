"""Feed-forward network (following AF2's MSA-Transition & Pair-Transition design)."""

from torch import nn


class FFN(nn.Module):
    """Feed-forward network (following AF2's MSA-Transition & Pair-Transition design)."""

    def __init__(self, n_dims, exp_fctr=4):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_dims = n_dims
        self.exp_fctr = exp_fctr

        # build the network
        self.net = nn.Sequential(
            nn.LayerNorm(self.n_dims),
            nn.Linear(self.n_dims, self.exp_fctr * self.n_dims),
            nn.ReLU(),
            nn.Linear(self.exp_fctr * self.n_dims, self.n_dims),
        )


    def forward(self, feat_tns):
        """Perform the forward pass.

        Args:
        * feat_tns: features of size N x (...) x D

        Returns:
        * feat_tns: updated features of size N x (...) x D
        """

        return self.net(feat_tns)
