"""The residue-atom message-passing neural network (MPNN) for AF2-like inputs."""

import itertools

import torch
from torch import nn

from tfold.tools import ProtStruct
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD
from tfold.modules.ra_mpnn.ra_mpnn import RAMpnn
from tfold.modules.ra_mpnn.ra_graph_builder import RAGraphBuilder
from tfold.modules.ra_mpnn.utils import sp2ds_atom
from tfold.modules.ra_mpnn.utils import ds2sp_atom


class RAMpnnAF2(RAMpnn):
    """The residue-atom message-passing neural network (MPNN) for AF2-like inputs."""

    def __init__(
            self,
            n_lyrs=4,  # number of <RAMpnnLayer> layers
            n_dims_sfea=384,  # number of dimensions in single features
            n_dims_pfea=256,  # number of dimensions in pair features
            n_dims_afea=64,  # number of dimensions in per-atom node features
            version='v1',  # EGCL version (choices: 'v1' / 'v2')
        ):
        """Constructor function."""

        super().__init__(
            n_lyrs,
            n_dims_sfea,  # per-residue node features
            n_dims_afea,  # per-atom node features
            n_dims_efea_r2r=n_dims_pfea,  # residue-to-residue edge features
            updt_nfrc=False,  # no need to predict node forces
            version=version,
        )

        # additional configurations
        self.atom_names = {'PAD'}
        for resd_name in RESD_NAMES_1C:
            self.atom_names.update(ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]])
        self.atom_names = sorted(list(self.atom_names))
        self.embed = nn.Embedding(len(self.atom_names), n_dims_afea)
        self.aidx_vec_dict = self.__build_aidx_vec_dict()
        self.ra_graph_builder = RAGraphBuilder()


    def forward(self, aa_seq, sfea_tns, pfea_tns, cord_tns, cmsk_mat=None, asym_id=None):
        """Perform the forward pass.

        Args:
        * aa_seq: amino-acid sequence
        * sfea_tns: single features of size N x L x D_s
        * pfea_tns: pair features of size N x L x L x D_p
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: (optional) per-atom 3D coordinates' validness masks of size L x M
        * asym_id: (optional) the asymmetric unit ID (chain ID) of size L (multimer only)

        Returns:
        * cord_tns: updated per-atom 3D coordinates of size L x M x 3
        """

        # initialize optional input tensors
        device = sfea_tns.device
        if cmsk_mat is None:
            cmsk_mat = ProtStruct.get_cmsk_vld(aa_seq, device)
        if asym_id is None:
            asym_id = torch.ones(len(aa_seq), dtype=torch.int8, device=device)

        # check whether all the CA atoms have valid 3D coordinates
        cmsk_vec_ca = ProtStruct.get_atoms(aa_seq, cmsk_mat, ['CA'])
        assert torch.all(torch.eq(cmsk_vec_ca, 1))

        # build per-atom node features
        aidx_mat = torch.stack([self.aidx_vec_dict[x] for x in aa_seq], dim=0).to(device)
        afea_tns = self.embed(aidx_mat)  # L x M x D_a
        afea_mat = sp2ds_atom(afea_tns, cmsk_mat)  # N_a x D_a (N_a: # of valid atoms)

        # build a heterogeneous graph of residues & atoms
        graph = self.ra_graph_builder.run(
            aa_seq, sfea_tns[0], pfea_tns[0], afea_mat, cord_tns, cmsk_mat, asym_id)

        # perform the forward pass
        graph = super().forward(graph, use_checkpoint=True)

        # extract updated per-atom 3D coordinates
        cord_tns = ds2sp_atom(graph['atom'].pos, cmsk_mat)

        return cord_tns


    def __build_aidx_vec_dict(self):
        """Build a dict of atom name indices, indexed by residue types."""

        aidx_vec_dict = {}
        for resd_name in RESD_NAMES_1C:
            atom_names_vld = ATOM_NAMES_PER_RESD[RESD_MAP_1TO3[resd_name]]
            atom_names_pad = atom_names_vld + ['PAD'] * (N_ATOMS_PER_RESD - len(atom_names_vld))
            aidx_vec_dict[resd_name] = torch.tensor([self.atom_names.index(x) for x in atom_names_pad])

        return aidx_vec_dict
