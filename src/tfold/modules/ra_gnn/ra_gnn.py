"""Residue/atom-aware graph neural network.

Notations:
* L: number of amino-acid residues
* M: maximal number of atoms per residue (=14)
* Na: total number of valid atoms
* Nr: total number of valid residues (for now, we assume all the residues are valid)
* Nap: total number of valid atom pairs
* Nrp: total number of valid residue pairs

Notes:
* Some residues may be unknown in the amino-acid sequence; 'X' will be used in that case. The atom
    ordering for unknown residues should be 'C', 'CA', 'N', and 'O'.
"""

import copy
import itertools
import functools

import dgl
import torch
from torch import nn
import numpy as np

from tfold.utils import cdist
from tfold.utils import send_to_device
from tfold.utils import calc_dihd_angl_batch
from tfold.tools import PosiEncoder
from tfold.tools import DistEncoder
from tfold.tools import AnglEncoder
from tfold.tools.prot_constants import RESD_NAMES_1C
from tfold.tools.prot_constants import RESD_MAP_1TO3
from tfold.tools.prot_constants import N_ATOMS_PER_RESD
from tfold.tools.prot_constants import ATOM_NAMES_PER_RESD
from tfold.modules.ra_gnn.utils import sp2ds
from tfold.modules.ra_gnn.utils import ds2sp
from tfold.modules.ra_gnn.onht_encoder import OnhtEncoder
from tfold.modules.ra_gnn.modules import Atom2Atom
from tfold.modules.ra_gnn.modules import Atom2Resd
from tfold.modules.ra_gnn.modules import Resd2Resd
from tfold.modules.ra_gnn.modules import Resd2Atom


class ResdAtomGNN(nn.Module):  # pylint: disable=too-many-instance-attributes
    """Residue/atom-aware graph neural network."""

    def __init__(
            self,
            n_blks=2,  # number of (Resd2Atom, Atom2Atom, Atom2Resd, Resd2Resd) blocks
            n_dims_ahid=32,  # number of dimensions in hidden per-atom embeddings
            n_dims_rhid=32,  # number of dimensions in hidden per-residue embeddings
            n_dims_aout=32,  # number of dimensions in output per-atom embeddings
            n_dims_rout=32,  # number of dimensions in output per-residue embeddings
            knn_dgr_ag=16,  # kNN degree in the atom-wise graph
            knn_dgr_rg=16,  # kNN degree in the residue-wise graph
            n_dims_posi=32,  # number of dimensions in positional encodings
            updt_graph=False,  # whether to update atom-wise & residue-wise kNN graphs
            skip_resd=False,  # whether to skip sub-networks for updating residues
        ):  # pylint: disable=too-many-arguments,too-many-statements
        """Constructor function."""

        super().__init__()

        # setup configurations
        self.n_blks = n_blks
        self.n_dims_ahid = n_dims_ahid
        self.n_dims_rhid = n_dims_rhid
        self.n_dims_aout = n_dims_aout
        self.n_dims_rout = n_dims_rout
        self.knn_dgr_ag = knn_dgr_ag
        self.knn_dgr_rg = knn_dgr_rg
        self.n_dims_posi = n_dims_posi
        self.updt_graph = updt_graph
        self.skip_resd = skip_resd

        # additional configurations for the unknown amino-acid type
        self.init_cord = False  # whether to initialize 3D coordinates of missing atoms
        self.idx_atom_ca = 1  # CA is the 2nd atom of any amino-acid
        self.resd_names_1c = RESD_NAMES_1C + ['X']
        self.resd_map_1to3 = copy.copy(RESD_MAP_1TO3)
        self.resd_map_1to3['X'] = 'UNK'
        self.atom_names_per_resd = copy.copy(ATOM_NAMES_PER_RESD)
        self.atom_names_per_resd['UNK'] = ['C', 'CA', 'N', 'O']
        self.nois_std = 0.3  # for initializing 3D coordinates of missing atoms
        self.atom_names_all = \
            sorted(list(set(itertools.chain.from_iterable(self.atom_names_per_resd.values()))))
        self.elem_names_all = sorted(list({x[0] for x in self.atom_names_all}))

        # base encoders
        self.posi_encoder = PosiEncoder(n_dims=self.n_dims_posi)
        self.dist_encoder = DistEncoder()
        self.angl_encoder = AnglEncoder()
        self.atom_encoder = OnhtEncoder(self.atom_names_all)
        self.elem_encoder = OnhtEncoder(self.elem_names_all)
        self.resd_encoder = OnhtEncoder(self.resd_names_1c)

        # additional configurations
        self.n_dims_emsg = 32  # number of dimensions in hidden edge messages (EGCL)
        self.n_dims_aemb = 32  # number of dimensions in atom name based embeddings
        self.n_dims_eemb = 16  # number of dimensions in element name based embeddings
        self.n_dims_remb = 32  # number of dimensions in residue name based embeddings
        self.n_dims_afea = self.n_dims_aemb + self.n_dims_eemb
        self.n_dims_rfea = self.n_dims_remb + \
            self.posi_encoder.n_dims + 3 * self.angl_encoder.n_dims  # phi / psi / omega
        self.n_dims_apfe = self.posi_encoder.n_dims
        self.n_dims_rpfe = self.posi_encoder.n_dims

        # build the dict of atom/residue/element indices for each residue type
        self.idxs_dict = self.__build_idxs_dict()

        # build the network
        self.net = nn.ModuleDict()
        self.net['embed-a'] = nn.Embedding(self.atom_encoder.n_dims, self.n_dims_aemb)
        self.net['embed-e'] = nn.Embedding(self.elem_encoder.n_dims, self.n_dims_eemb)
        self.net['embed-r'] = nn.Embedding(self.resd_encoder.n_dims + 1, self.n_dims_remb)
        self.net['linear-ai'] = nn.Linear(self.n_dims_afea, self.n_dims_ahid)
        self.net['linear-ri'] = nn.Linear(self.n_dims_rfea, self.n_dims_rhid)
        self.net['linear-ao'] = nn.Linear(self.n_dims_ahid, self.n_dims_aout)
        self.net['linear-ro'] = nn.Linear(self.n_dims_rhid, self.n_dims_rout)
        self.net['init-r2a'] = Resd2Atom(self.n_dims_ahid, self.n_dims_rhid)
        self.net['finl-a2r'] = Atom2Resd(
            self.n_dims_ahid, self.n_dims_rhid, dist_encoder=self.dist_encoder)
        for idx_blk in range(self.n_blks):
            self.net[f'blk{idx_blk}-a2a'] = Atom2Atom(
                self.n_dims_ahid,
                self.n_dims_apfe,
                n_dims_emsg=self.n_dims_emsg,
                dist_encoder=self.dist_encoder,
            )
            if self.skip_resd:
                continue  # do not build sub-networks for updating residues
            self.net[f'blk{idx_blk}-a2r'] = Atom2Resd(
                self.n_dims_ahid,
                self.n_dims_rhid,
                dist_encoder=self.dist_encoder,
            )
            self.net[f'blk{idx_blk}-r2r'] = Resd2Resd(
                self.n_dims_rhid,
                self.n_dims_rpfe,
                n_dims_emsg=self.n_dims_emsg,
                dist_encoder=self.dist_encoder,
            )
            self.net[f'blk{idx_blk}-r2a'] = Resd2Atom(
                self.n_dims_ahid,
                self.n_dims_rhid,
            )


    def forward(self, aa_seq, cord_tns, cmsk_mat, umsk_mat):  # pylint: disable=too-many-locals
        """Perform the forward pass.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' valid-or-not masks of size L x M
        * umsk_mat: per-atom 3D coordinates' update-or-not masks of size L x M

        Returns:
        * rfea_mat: per-residue features of size L x Dr
        * afea_tns: per-atom features of size L x M x Da
        * acrd_mat: per-atom 3D coordinates of size L x M x 3
        """

        # initialization
        n_resds_all = len(aa_seq)
        n_atoms_all = n_resds_all * N_ATOMS_PER_RESD
        device = cord_tns.device
        self.idxs_dict = send_to_device(self.idxs_dict, device)

        # initialize 3D coordinates of missing atoms
        if self.init_cord:
            cord_tns, cmsk_mat, umsk_mat = self.__init_cord(aa_seq, cord_tns, cmsk_mat, umsk_mat)

        # convert 3D coordinates & masks into the dense format
        acrd_mat = sp2ds(aa_seq, cord_tns, cmsk_mat)
        aumk_vec = sp2ds(aa_seq, umsk_mat, cmsk_mat)

        # build initial per-atom & per-residue features
        _, idxs_dict = self.__init_idx_vecs(aa_seq, cmsk_mat)
        afea_mat, rfea_mat, _ = self.__build_ar_feats(aa_seq, idxs_dict, cord_tns, cmsk_mat)

        # calculate initial per-residue 3D coordinates
        acrd_tns = torch.index_select(ds2sp(aa_seq, acrd_mat, cmsk_mat), 0, idxs_dict['rv2ra'])
        avmk_tns = torch.index_select(cmsk_mat, 0, idxs_dict['rv2ra']).unsqueeze(dim=2)
        rcrd_mat = torch.sum(avmk_tns * acrd_tns, dim=1) / torch.sum(avmk_tns, dim=1)

        # build initial atom-wise & residue-wise kNN graphs
        graph_atom, apfe_mat = self.__build_atom_graph(acrd_mat, idxs_dict['av2ra'])
        graph_resd, rpfe_mat = self.__build_resd_graph(rcrd_mat, idxs_dict['rv2ra'])

        # perform initial mappings on per-atom & per-residue features
        afea_mat = self.net['linear-ai'](afea_mat)
        rfea_mat = self.net['linear-ri'](rfea_mat)

        # perform the initial <Resd2Atom> update
        afea_mat = self.net['init-r2a'](afea_mat, rfea_mat, idxs_dict['av2rv'])

        # pass through multiple sub-networks to model atom-residue interations
        for idx_blk in range(self.n_blks):
            # perform the <Atom2Atom> update
            afea_mat, acrd_mat = \
                self.net[f'blk{idx_blk}-a2a'](graph_atom, afea_mat, apfe_mat, acrd_mat, aumk_vec)

            # (optional) update the atom-wise kNN graph
            if self.updt_graph and (idx_blk != self.n_blks - 1):
                graph_atom, apfe_mat = self.__build_atom_graph(acrd_mat, idxs_dict['av2ra'])

            # skip sub-networks for updating residues
            if self.skip_resd:
                continue

            # perform the <Atom2Resd> update
            rfea_mat, rcrd_mat = self.net[f'blk{idx_blk}-a2r'](
                aa_seq, cmsk_mat, afea_mat, acrd_mat, rfea_mat, idxs_dict['rv2ra'])

            # (optional) update the residue-wise kNN graph
            if self.updt_graph:
                graph_resd, rpfe_mat = self.__build_resd_graph(rcrd_mat, idxs_dict['rv2ra'])

            # perform <Resd2Resd> and <Resd2Atom> updates
            rfea_mat, _ = self.net[f'blk{idx_blk}-r2r'](graph_resd, rfea_mat, rpfe_mat, rcrd_mat)
            afea_mat = self.net[f'blk{idx_blk}-r2a'](afea_mat, rfea_mat, idxs_dict['av2rv'])

        # perform the final <Atom2Resd> update
        rfea_mat, rcrd_mat = self.net['finl-a2r'](
            aa_seq, cmsk_mat, afea_mat, acrd_mat, rfea_mat, idxs_dict['rv2ra'])

        # perform final mappings on per-atom & per-residue features
        afea_mat = self.net['linear-ao'](afea_mat)
        rfea_mat = self.net['linear-ro'](rfea_mat)

        # scatter per-atom/residue features & 3D coordinates for consistency
        n_resds_vld = rfea_mat.shape[0]
        n_atoms_vld = afea_mat.shape[0]
        rfea_mat = torch.scatter(
            torch.zeros((n_resds_all, self.n_dims_rout), dtype=torch.float32, device=device),
            0, idxs_dict['rv2ra'].view(n_resds_vld, 1).repeat(1, self.n_dims_rout), rfea_mat,
        )
        afea_tns = torch.scatter(
            torch.zeros((n_atoms_all, self.n_dims_aout), dtype=torch.float32, device=device),
            0, idxs_dict['av2aa'].view(n_atoms_vld, 1).repeat(1, self.n_dims_aout), afea_mat,
        ).view(n_resds_all, N_ATOMS_PER_RESD, self.n_dims_aout)
        acrd_tns = torch.scatter(
            torch.zeros((n_atoms_all, 3), dtype=torch.float32, device=device),
            0, idxs_dict['av2aa'].view(n_atoms_vld, 1).repeat(1, 3), acrd_mat,
        ).view(n_resds_all, N_ATOMS_PER_RESD, 3)

        return rfea_mat, afea_tns, acrd_tns


    @torch.no_grad()
    def __init_cord(self, aa_seq, cord_tns, cmsk_mat, umsk_mat):  # pylint: disable=too-many-locals
        """Initialize 3D coordinates of missing atoms."""

        def _calc_idx_gap(idx, base=0):
            return abs(idx - base)

        # initialization
        n_resds = len(aa_seq)

        # fill-up missing 3D coordinates for CA atoms
        idxs_resd_vld = torch.nonzero(cmsk_mat[:, self.idx_atom_ca], as_tuple=True)[0].tolist()
        for idx_resd in range(n_resds):
            if cmsk_mat[idx_resd, self.idx_atom_ca] == 1:
                continue
            idxs_resd_vld.sort(key=functools.partial(_calc_idx_gap, base=idx_resd))
            idx_resd_pri = idxs_resd_vld[0]
            idx_resd_sec = idxs_resd_vld[1]
            cord_vec_pri = cord_tns[idx_resd_pri, self.idx_atom_ca]
            cord_vec_sec = cord_tns[idx_resd_sec, self.idx_atom_ca]
            cord_tns[idx_resd, self.idx_atom_ca] = cord_vec_pri + (idx_resd - idx_resd_pri) \
                / (idx_resd_sec - idx_resd_pri) * (cord_vec_sec - cord_vec_pri)
            cmsk_mat[idx_resd, self.idx_atom_ca] = 1

        # build validness masks from the amino-acid sequence alone
        vmsk_mat = torch.zeros_like(cmsk_mat)
        for idx_resd, resd_name in enumerate(aa_seq):
            atom_names = self.atom_names_per_resd[self.resd_map_1to3[resd_name]]
            vmsk_mat[idx_resd, :len(atom_names)] = 1

        # initialize 3D coordinates of missing atoms
        nois_tns = self.nois_std * torch.randn_like(cord_tns)
        cord_tns = vmsk_mat.unsqueeze(dim=2) * torch.where(
            cmsk_mat.unsqueeze(dim=2).to(torch.bool),
            cord_tns, cord_tns[:, self.idx_atom_ca].unsqueeze(dim=1) + nois_tns,
        )

        # modify update-or-not masks
        umsk_mat = torch.where(cmsk_mat.to(torch.bool), umsk_mat, vmsk_mat)  # update missing atoms

        return cord_tns, vmsk_mat, umsk_mat  # NOTE <vmsk_mat> is returned as validness masks


    def __init_idx_vecs(self, aa_seq, cmsk_mat):  # pylint: disable=too-many-locals
        """Initialize per-atom & per-residue indexing vectors.

        Note:
        * aa: atom / all
        * av: atom / valid-only
        * ra: resd / all
        * rv: resd / valid-only
        """

        # initialization
        n_resds = len(aa_seq)
        device = cmsk_mat.device

        # atom-to-atom / residue-to-residue indexing vectors
        idx_vec_av2aa = torch.nonzero(cmsk_mat.flatten(), as_tuple=True)[0]
        idx_vec_ra2ra = torch.arange(n_resds, dtype=torch.int64, device=device)
        rsiz_vec_all = torch.sum(cmsk_mat, dim=1)
        idx_vec_rv2ra = torch.nonzero(rsiz_vec_all, as_tuple=True)[0]
        rsiz_vec_vld = torch.take(rsiz_vec_all, idx_vec_rv2ra)

        # atom-to-residue indexing vectors
        n_atoms_vld = idx_vec_av2aa.shape[0]
        rsiz_vec_all_np = rsiz_vec_all.detach().cpu().numpy()
        idx_vec_av2rv_np = np.zeros((n_atoms_vld), dtype=np.int64)
        idx_vec_av2ra_np = np.zeros((n_atoms_vld), dtype=np.int64)
        idx_resd_vld = 0
        idx_atom_base = 0
        for idx_resd_all in range(n_resds):
            n_atoms_sel = rsiz_vec_all_np[idx_resd_all]
            if n_atoms_sel != 0:
                idx_vec_av2rv_np[idx_atom_base:idx_atom_base + n_atoms_sel] = idx_resd_vld
                idx_vec_av2ra_np[idx_atom_base:idx_atom_base + n_atoms_sel] = idx_resd_all
                idx_resd_vld += 1
                idx_atom_base += n_atoms_sel
        idx_vec_av2rv = torch.tensor(idx_vec_av2rv_np, dtype=torch.int64, device=device)
        idx_vec_av2ra = torch.tensor(idx_vec_av2ra_np, dtype=torch.int64, device=device)

        # pack all the indexing vectors into a dict
        idxs_dict = {
            'av2aa': idx_vec_av2aa,  # [0, Na) => [0, L x M)
            'av2rv': idx_vec_av2rv,  # [0, Na) => [0, Nr)
            'av2ra': idx_vec_av2ra,  # [0, Na) => [0, L)
            'ra2ra': idx_vec_ra2ra,  # [0, L) => [0, L)
            'rv2ra': idx_vec_rv2ra,  # [0, Nr) => [0, L)
        }

        return rsiz_vec_vld, idxs_dict


    def __build_ar_feats(self, aa_seq, idxs_dict, cord_tns, cmsk_mat):
        """Build initial per-atom & per-residue features."""

        # build per-atom features
        afea_mat_all = torch.cat([
            self.net['embed-a'](torch.cat([self.idxs_dict[x]['atom'] for x in aa_seq], dim=0)),
            self.net['embed-e'](torch.cat([self.idxs_dict[x]['elem'] for x in aa_seq], dim=0)),
        ], dim=1)  # (L x M) x Da
        afea_mat_vld = torch.index_select(afea_mat_all, 0, idxs_dict['av2aa'])  # Na x Da

        # build per-residue features
        rfea_mat_all = torch.cat([
            self.net['embed-r'](torch.cat([self.idxs_dict[x]['resd'] for x in aa_seq], dim=0)),
            self.posi_encoder.run(idxs_dict['ra2ra']),
            self.__build_rfea_mat_angl(aa_seq, cord_tns, cmsk_mat),
        ], dim=1)  # L x Dr
        rfea_mat_vld = torch.index_select(rfea_mat_all, 0, idxs_dict['rv2ra'])  # Nr x Dr

        # build per-atom indices of chemical element names
        eidx_vec_all = torch.cat([self.idxs_dict[x]['elem'] for x in aa_seq], dim=0)  # (L x M)
        eidx_vec_vld = torch.take(eidx_vec_all, idxs_dict['av2aa'])  # Na

        return afea_mat_vld, rfea_mat_vld, eidx_vec_vld


    def __build_idxs_dict(self):
        """Build the dict of atom/residue/element indices for each residue type."""

        idxs_dict = {}
        for resd_name_1c, resd_name_3c in self.resd_map_1to3.items():
            atom_names = self.atom_names_per_resd[resd_name_3c]
            elem_names = [x[0] for x in atom_names]
            resd_names = [resd_name_1c]
            pad_size = N_ATOMS_PER_RESD - len(atom_names)
            idxs_dict[resd_name_1c] = {
                'atom': nn.functional.pad(self.atom_encoder.name2idx(atom_names), (0, pad_size)),
                'elem': nn.functional.pad(self.elem_encoder.name2idx(elem_names), (0, pad_size)),
                'resd': self.resd_encoder.name2idx(resd_names),
            }

        return idxs_dict


    def __build_atom_graph(self, acrd_mat, ridx_vec):
        """Build an atom-wise kNN graph from per-atom 3D coordinates."""

        # build a kNN graph
        graph, nidx_vec_src, nidx_vec_dst = self.__build_knn_graph(acrd_mat, self.knn_dgr_ag)

        # build per-atom-pair features
        ridx_vec_src = torch.take(ridx_vec, nidx_vec_src)
        ridx_vec_dst = torch.take(ridx_vec, nidx_vec_dst)
        apfe_mat = self.posi_encoder.run(ridx_vec_src - ridx_vec_dst)

        return graph, apfe_mat


    def __build_resd_graph(self, rcrd_mat, ridx_vec):
        """Build a residue-wise kNN graph from per-residue 3D coordinates."""

        # build a kNN graph
        graph, nidx_vec_src, nidx_vec_dst = self.__build_knn_graph(rcrd_mat, self.knn_dgr_rg)

        # build per-residue-pair features
        ridx_vec_src = torch.take(ridx_vec, nidx_vec_src)
        ridx_vec_dst = torch.take(ridx_vec, nidx_vec_dst)
        rpfe_mat = self.posi_encoder.run(ridx_vec_src - ridx_vec_dst)

        return graph, rpfe_mat


    def __build_knn_graph(self, cord_mat, degree, method='dgl'):
        """Build a kNN graph."""

        # initialization
        device = cord_mat.device
        n_nodes = cord_mat.shape[0]
        degree = min(degree, n_nodes)

        # build a kNN graph
        if method == 'dgl':
            graph = dgl.knn_graph(cord_mat, degree).to(device)
            nidx_vec_src, nidx_vec_dst = graph.edges()
        elif method == 'built-in':
            dist_mat = cdist(cord_mat)
            nidx_vec_src = torch.argsort(dist_mat, dim=1)[:, :degree].flatten()
            nidx_vec_dst = torch.repeat_interleave(
                torch.arange(n_nodes, dtype=torch.int64, device=device), degree)
            graph = dgl.graph((nidx_vec_src, nidx_vec_dst), num_nodes=n_nodes).to(device)
        else:
            raise ValueError('unrecognized kNN graph construction method: {method}')

        return graph, nidx_vec_src, nidx_vec_dst


    def __build_rfea_mat_angl(self, aa_seq, cord_tns, cmsk_mat):  # pylint: disable=too-many-locals
        """Build per-residue features from dihedral angles.

        Notes:
        * phi: C_prev - N_curr - CA_curr - C_curr
        * psi: N_curr - CA_curr - C_curr - N_next
        * omega: CA_curr - C_curr - N_next - CA_next
        """

        # obtain 3D coordinates & validness masks for C/CA/N atoms
        atom_names = ['C', 'CA', 'N']
        cord_tns_sel = self.__get_atoms(aa_seq, cord_tns, atom_names)
        cmsk_mat_sel = self.__get_atoms(aa_seq, cmsk_mat, atom_names)

        # pad 3D coordinates & validness masks by one residue at each end
        cord_mat_c = nn.functional.pad(cord_tns_sel[:, 0], (0, 0, 1, 1))
        cmsk_vec_c = nn.functional.pad(cmsk_mat_sel[:, 0], (1, 1))
        cord_mat_ca = nn.functional.pad(cord_tns_sel[:, 1], (0, 0, 1, 1))
        cmsk_vec_ca = nn.functional.pad(cmsk_mat_sel[:, 1], (1, 1))
        cord_mat_n = nn.functional.pad(cord_tns_sel[:, 2], (0, 0, 1, 1))
        cmsk_vec_n = nn.functional.pad(cmsk_mat_sel[:, 2], (1, 1))

        # calculate dihedral angles (phi, psi, and omega)
        angl_vec_ph = calc_dihd_angl_batch(torch.stack([
            cord_mat_c[:-2], cord_mat_n[1:-1], cord_mat_ca[1:-1], cord_mat_c[1:-1]], dim=1))
        amsk_vec_ph = cmsk_vec_c[:-2] * cmsk_vec_n[1:-1] * cmsk_vec_ca[1:-1] * cmsk_vec_c[1:-1]
        angl_vec_ps = calc_dihd_angl_batch(torch.stack([
            cord_mat_n[1:-1], cord_mat_ca[1:-1], cord_mat_c[1:-1], cord_mat_n[2:]], dim=1))
        amsk_vec_ps = cmsk_vec_n[1:-1] * cmsk_vec_ca[1:-1] * cmsk_vec_c[1:-1] * cmsk_vec_n[2:]
        angl_vec_om = calc_dihd_angl_batch(torch.stack([
            cord_mat_ca[1:-1], cord_mat_c[1:-1], cord_mat_n[2:], cord_mat_ca[2:]], dim=1))
        amsk_vec_om = cmsk_vec_ca[1:-1] * cmsk_vec_c[1:-1] * cmsk_vec_n[2:] * cmsk_vec_ca[2:]

        # convert dihedral angles into sine/cosine encodings
        rfea_mat = torch.cat([
            amsk_vec_ph.unsqueeze(dim=1) * self.angl_encoder.run(angl_vec_ph),
            amsk_vec_ps.unsqueeze(dim=1) * self.angl_encoder.run(angl_vec_ps),
            amsk_vec_om.unsqueeze(dim=1) * self.angl_encoder.run(angl_vec_om),
        ], dim=1)

        return rfea_mat


    def __get_atoms(self, aa_seq, atom_tns_all, atom_names_sel):  # pylint: disable=too-many-locals
        """Get 3D coordinates or validness masks for selected atoms."""

        # initialization
        device = atom_tns_all.device
        n_atoms = len(atom_names_sel)

        # build the indexing tensor for selected atom(s)
        idxs_vec_dict = {}  # atom indices
        msks_vec_dict = {}  # atom indices' validness masks
        for resd_name in self.resd_names_1c:
            atom_names_all = self.atom_names_per_resd[self.resd_map_1to3[resd_name]]
            idxs_vec = torch.zeros((n_atoms), dtype=torch.int64)
            msks_vec = torch.zeros((n_atoms), dtype=torch.int8)
            for idx_atom_sel, atom_name_sel in enumerate(atom_names_sel):
                if atom_name_sel in atom_names_all:  # otherwise, keep zeros unchanged
                    idxs_vec[idx_atom_sel] = atom_names_all.index(atom_name_sel)
                    msks_vec[idx_atom_sel] = 1
            idxs_vec_dict[resd_name] = idxs_vec
            msks_vec_dict[resd_name] = msks_vec

        # determine the overall indexing tensor based on the amino-acid sequence
        idxs_mat = torch.stack([idxs_vec_dict[x] for x in aa_seq], dim=0).to(device)
        msks_mat = torch.stack([msks_vec_dict[x] for x in aa_seq], dim=0).to(device)

        # get per-atom 3D coordinates or validness masks for specified residue(s) & atom(s)
        if atom_tns_all.ndim == 2:
            atom_tns_sel = msks_mat * torch.gather(atom_tns_all, 1, idxs_mat)
        else:
            n_dims_addi = atom_tns_all.shape[-1]
            atom_tns_sel = msks_mat.unsqueeze(dim=2) * torch.gather(
                atom_tns_all, 1, idxs_mat.unsqueeze(dim=2).repeat(1, 1, n_dims_addi))

        return atom_tns_sel
