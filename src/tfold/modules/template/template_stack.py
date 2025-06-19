"""Stacked <EvoformerBlockSS> layers."""

from torch import nn
from torch.utils.checkpoint import checkpoint

from tfold.modules.template.template_block import TemplatePairBlock, TemplateSeqBlock


class TemplatePairStack(nn.Module):
    """Stacked <TemplatePairBlock> layers."""

    def __init__(
            self,
            n_lyrs=4,             # number of <TemplatePairBlock> layers
            n_dims_pfea=256,      # number of dimensions in pair features (c_z)
            use_templ_attn=False,  # whether to use template-wise attention
            use_checkpoint=True,  # whether to use the checkpoint mechanism to avoid OMM
    ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_lyrs = n_lyrs
        self.n_dims_pfea = n_dims_pfea
        self.use_templ_attn = use_templ_attn
        self.use_checkpoint = use_checkpoint

        # build a stack of <TemplatePairBlock> modules
        self.blocks = nn.ModuleList()
        for _ in range(self.n_lyrs):
            block = TemplatePairBlock(self.n_dims_pfea, use_templ_attn=self.use_templ_attn)
            self.blocks.append(block)

    def forward(self, tfea_tns, tfea_mask=None, chunk_size=None):
        """Perform the forward pass.

        Args:
        * tfea_tns: template pairwise features of size N x T x L x L x c_z
        * tfea_mask: template mask of size N x T

        Returns:
        * sfea_tns: updated single features of size N x L x c_s
        * pfea_tns: updated pair features of size N x L x L x c_z
        """

        requires_grad = next(iter(self.parameters())).requires_grad
        for block in self.blocks:
            if not (self.training and self.use_checkpoint and requires_grad):
                tfea_tns = block(tfea_tns, tfea_mask, chunk_size)
            else:
                tfea_tns = checkpoint(block, tfea_tns, tfea_mask, chunk_size)

        return tfea_tns


class TemplateSeqStack(nn.Module):
    """Stacked <TemplateSeqBlock> layers."""
    def __init__(
            self,
            n_lyrs=4,             # number of <TemplateSeqBlock> layers
            n_dims_sfea=384,      # number of dimensions in pair features (c_z)
    ):
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_lyrs = n_lyrs
        self.n_dims_pfea = n_dims_sfea

        # build a stack of <EvoformerBlockSS> modules
        self.blocks = nn.ModuleList()
        for _ in range(self.n_lyrs):
            block = TemplateSeqBlock(self.n_dims_sfea)
            self.blocks.append(block)

    def forward(self, tfea_tns):
        """Perform the forward pass.

        Args:
        * tfea_tns: template sequential features of size N x T x L x c_z

        Returns:
        * tfea_tns: updated template sequential features of size N x T x L x c_s
        """
        for block in self.blocks:
            tfea_tns = block(tfea_tns)

        return tfea_tns
