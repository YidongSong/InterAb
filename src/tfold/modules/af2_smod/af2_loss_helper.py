"""The helper class for calculating the loss and evaluation metrics related to <AF2SMod>.

List of loss functions:
> FAPE: frame aligned point error
> Angl: regression & L2-norm regularization for torsion angle matrix predictions
> LDDT: classification loss for per-residue lDDT-Ca predictions
> QNrm: L2-norm regularization loss for quaternion vector predictions
> Clsh: structural violation loss (steric clashes among non-bonded atoms
> TMscore: classification loss for TMscore predictions)
> iFAPE: inter-chain frame aligned point error
"""

import numpy as np
import torch
from torch import nn

from tfold.utils import cdist
from tfold.utils import quat2rot
from tfold.utils import apply_trans
# from tfold.tools import ClashCheckerV2 as ClashChecker
from tfold.tools import ClashCheckerV3 as ClashChecker
from tfold.tools import LddtAssessor
from tfold.tools import ProtStruct
from tfold.tools import ProtConverter


class AF2LossHelper():  # pylint: disable=too-many-instance-attributes
    """The helper class for calculating the loss and evaluation metrics related to <AF2SMod>."""

    def __init__(
            self,
            wc_fape=1.0,       # weighting coefficient for Loss-FAPE
            wc_angl=1.0,       # weighting coefficient for Loss-Angl
            wc_lddt=0.1,       # weighting coefficient for Loss-LDDT
            wc_qnrm=0.02,      # weighting coefficient for Loss-QNrm
            wc_clsh=0.01,      # weighting coefficient for Loss-Clsh
            wc_tmsc=0.1,       # weighting coefficient for Loss-TMscore
            wc_ifape=0.0,      # weighting coefficient for Loss-iFAPE
            wc_itfape=0.0,     # Weighting coefficient for Loss-itFAPE
            loss_nb_max=1.0,   # maximal loss for inter-atom distance between non-bonded atoms
            dist_clamp_ca=10.0,  # CA-atom distance clamping threshold for Loss-FAPE
            dist_clamp_fa=None,  # full-atom distance clamping threshold for Loss-FAPE
            dist_clamp_ica=1000.0,  # CA-atom distance clamping threshold for Loss-iFAPE
            debug=False,       # whether to enable the debug mode to evaluate all the losses
        ):  # pylint: disable=too-many-arguments
        """Constructor function."""

        # basic configurations
        self.wc_fape = wc_fape
        self.wc_angl = wc_angl
        self.wc_lddt = wc_lddt
        self.wc_qnrm = wc_qnrm
        self.wc_clsh = wc_clsh
        self.wc_tmsc = wc_tmsc
        self.wc_ifape = wc_ifape
        self.wc_itfape = wc_itfape
        self.loss_nb_max = loss_nb_max
        self.dist_clamp_ca = dist_clamp_ca
        self.dist_clamp_fa = dist_clamp_fa if dist_clamp_fa is not None else dist_clamp_ca
        self.dist_clamp_ica = dist_clamp_ica
        self.debug = debug

        # additional configurations
        self.eps = 1e-4
        self.n_bins_lddt = 50  # number of bins for pLDDT-Ca predictions
        # self.clash_checker = ClashChecker(loss_nb_max=self.loss_nb_max)
        self.clash_checker = ClashChecker(
            norm_by_cpairs=True,
            loss_bd_max=3.0,
            loss_nb_max=self.loss_nb_max,
            check_irbd=False,
        )
        self.lddt_assessor = LddtAssessor()
        self.prot_struct = ProtStruct()  # this is the NATIVE structure
        self.prot_converter = ProtConverter()
        self.reso_thres = 3.0
        self.mthds_hq = ['x-ray diffraction', 'electron microscopy']
        self.is_hq_struct = False  # whether the native structure is of high quality
        self.conf_vec = None
        self.use_clamp_dict = None  # whether loss clamping is enabled for FAPE & iFAPE
        self.asym_id = None  # asym id for multimer predictions
        self.itef_id = None  # itef_id for multimer predictions


    def init(self, aa_seq, cord_tns, cmsk_mat, reso, mthd, conf_vec=None, asym_id=None, itef_id=None):  # pylint: disable=too-many-arguments
        """Initialize the helper class w/ the native structure.

        Args:
        * aa_seq: amino-acid sequence
        * cord_tns: native structure's per-atom 3D coordinates of size L x M x 3
        * cmsk_mat: per-atom 3D coordinates' validness masks of size L x M
        * reso: experimental resolution (scalar)
        * mthd: structure determination method (string)
        * conf_vec: (optional) per-residue confidence vector of size L
        * asym_id: (optional) the asymmetric unit ID - the chain ID of size L

        Returns: n/a
        """

        # centralize per-atom 3D coordinates
        cord_vec_avg = torch.sum(
            cmsk_mat.view(-1, 1) * cord_tns.view(-1, 3), dim=0) / (torch.sum(cmsk_mat) + self.eps)
        cord_tns_cen = cord_tns - cord_vec_avg.view(1, 1, 3)  # raw structure

        # initialize the native structure
        self.prot_struct.init_from_cord(aa_seq, cord_tns_cen, cmsk_mat)

        # calculate backbone & side-chain local frames & torsion angles
        self.prot_struct.build_fram_n_angl(self.prot_converter, build_sc=True)

        # calculate valid/symmetric-or-not masks for atoms & torsion angles
        self.prot_struct.build_mask()

        # determine whether the native structure is of high quality
        self.is_hq_struct = \
            (reso is not None) and (reso <= self.reso_thres) and (mthd in self.mthds_hq)

        # record the per-residue confidence vector
        if conf_vec is not None:
            self.conf_vec = conf_vec
        else:
            self.conf_vec = torch.ones((len(aa_seq)), dtype=torch.float32, device=cord_tns.device)

        # determine whether loss clamping is enabled for FAPE & iFAPE
        self.use_clamp_dict = {
            'fape-fa': True,  # always apply loss clamping
            'fape-ca': (np.random.uniform() <= 0.9),  # 90% clamped + 10% unclamped
            'ifape': True,  # always apply loss clamping (both intra-chain and inter-chain)
        }

        # record the asym_id for multimer predictions
        self.asym_id = asym_id

        # record the itef_id for multimer predictions
        self.itef_id = itef_id


    def calc_loss(
            self, params_list, plddt_list, cord_list, fram_tns_sc, tmsc_dict=None,
        ):  # pylint: disable=too-many-arguments,too-many-locals
        """Calculate the loss function and evaluation metrics.

        Args:
        * params_list: list of QTA parameters, one per layer
          > quat: quaternion vectors of size L x 4
          > trsl: translation vectors of size L x 3
          > angl: torsion angle matrices of size L x K x 2
          > quat-u: update signal of quaternion vectors of size L x 4
        * plddt_list: list of per-residue & full-chain lDDT-Ca predictions, one per layer
          > logit: raw classification logits of size L x 50
          > plddt-r: per-residue predicted lDDT-Ca scores of size L
          > plddt-c: full-chain predicted lDDT-Ca score (scalar)
        * cord_list: list of per-atom 3D coordinates of size L x M x 3, one per layer
        * fram_tns_sc: final layer's per-residue side-chain frames of size L x K x 4 x 3
        * tmsc_dict: dict of pTM (and ipTM) predictions

        Returns:
        * loss: loss function
        * metrics: dict of evaluation metrics
        """

        # initialization
        loss_list, metrics = [], {}

        # rename symmetric ground-truth atoms in the native structure
        with torch.no_grad():
            self.prot_struct.rename_sym_atoms(
                cord_list[-1], self.prot_struct.cmsk_mat, self.prot_converter)

        # calculate the frame aligned point error (FAPE)
        if self.debug or (self.wc_fape > 0.0):
            loss_fape, metrics_fape = self.__calc_loss_fape(params_list, cord_list, fram_tns_sc)
            loss_list.append(self.wc_fape * loss_fape)
            metrics.update(**metrics_fape)

        # calculate the intra_chain and interface frame aligned point error (iFAPE)
        if (self.debug or (self.wc_ifape > 0.0)) and (self.asym_id is not None):
            loss_ifape, metrics_ifape = self.__calc_loss_ifape(params_list[-1], cord_list[-1])
            loss_list.append(self.wc_ifape * loss_ifape)
            metrics.update(**metrics_ifape)

        # calculate the inteface frame aligned point error (itFAPE) for antibody / TCR & antigen interface
        if (self.debug or (self.wc_itfape > 0.0)) and (self.itef_id is not None):
            loss_itfape, metrics_itface = self.__calc_loss_itfape(params_list[-1], cord_list[-1])
            loss_list.append(self.wc_itfape * loss_itfape)
            metrics.update(**metrics_itface)

        # calculate the torsion angle prediction loss
        if self.debug or (self.wc_angl > 0.0):
            loss_angl, metrics_angl = self.__calc_loss_angl(params_list)
            loss_list.append(self.wc_angl * loss_angl)
            metrics.update(**metrics_angl)

        # calculate the classification loss for per-residue lDDT-Ca predictions
        if self.debug or (self.wc_lddt > 0.0):
            loss_lddt, metrics_lddt = self.__calc_loss_lddt(plddt_list, cord_list)
            wc_lddt = self.wc_lddt if self.is_hq_struct else 0.0  # skip low-quality structures
            loss_list.append(wc_lddt * loss_lddt)
            metrics.update(**metrics_lddt)  # metrics should always be recorded

        # calculate the L2-norm regularization loss for quaternion vector predictions
        if self.debug or (self.wc_qnrm > 0.0):
            loss_qnrm, metrics_qnrm = self.__calc_loss_qnrm(params_list)
            loss_list.append(self.wc_qnrm * loss_qnrm)
            metrics.update(**metrics_qnrm)

        # calculate the structural violation loss (steric clashes among non-bonded atoms)
        if self.debug or (self.wc_clsh > 0.0):
            loss_clsh, metrics_clsh = self.__calc_loss_clsh(cord_list[-1])
            loss_list.append(self.wc_clsh * loss_clsh)
            metrics.update(**metrics_clsh)

        # calculate the TMscore loss
        if self.debug or (self.wc_tmsc > 0.0):
            loss_tmsc, metrics_tmsc = \
                self.__calc_loss_tmsc(tmsc_dict, params_list[-1], cord_list[-1])
            loss_list.append(self.wc_tmsc * loss_tmsc)
            metrics.update(**metrics_tmsc)

        # aggregate all the loss functions and evaluation metrics
        loss = torch.sum(torch.stack(loss_list))
        metrics['Loss'] = loss.item()

        return loss, metrics


    def __calc_loss_fape(self, params_list, cord_list, fram_tns_sc):
        """Calculate the frame aligned point error (FAPE)."""

        # initialization
        metrics = {}

        # calculate the FAPE loss w/ CA-atom and backbone frames
        loss_ca_list = []
        for idx_lyr, (params, cord_tns) in enumerate(zip(params_list, cord_list)):
            loss_ca = self.__calc_loss_fape_impl(
                params, cord_tns, atom_set='ca', fram_set='bb',
                dist_clamp=self.dist_clamp_ca, use_clamp=self.use_clamp_dict['fape-ca'],
            )
            loss_ca_list.append(loss_ca)
            #metrics[f'dRMSD-L{idx_lyr + 1}'] = self.__calc_drmsd(cord_tns, atom_set='ca')
        loss_ca = torch.mean(torch.stack(loss_ca_list))

        # calculate the FAPE loss w/ full-atom and backbone & side-chain frames
        loss_fa = self.__calc_loss_fape_impl(
            params_list[-1], cord_list[-1], atom_set='fa', fram_set='bs',
            dist_clamp=self.dist_clamp_fa, use_clamp=self.use_clamp_dict['fape-fa'],
            fram_tns_sc=fram_tns_sc,
        )
        #metrics['dRMSD-FA'] = self.__calc_drmsd(cord_list[-1], atom_set='fa')

        # calculate the overall loss function
        loss = loss_ca + loss_fa
        metrics.update({
            'Loss-FAPE': loss.item(),
            'Loss-FAPE-CA': loss_ca.item(),
            'Loss-FAPE-FA': loss_fa.item(),
        })

        return loss, metrics


    def __calc_loss_angl(self, params_list):
        """Calculate the torsion angle prediction loss."""

        # calculate the torsion angle prediction loss
        loss_list = []
        for params in params_list:
            angl_tns_norm = torch.norm(params['angl'], dim=-1, keepdim=True)
            angl_tns_pred = params['angl'] / (angl_tns_norm + self.eps)
            angl_tns_natv_bsc = self.prot_struct.angl_tns
            angl_tns_natv_alt = \
                (1 - 2 * self.prot_struct.amsk_mat_sym).unsqueeze(dim=2) * angl_tns_natv_bsc
            aerr_mat_bsc = torch.norm(angl_tns_pred - angl_tns_natv_bsc, dim=-1).square()
            aerr_mat_alt = torch.norm(angl_tns_pred - angl_tns_natv_alt, dim=-1).square()
            aerr_mat = torch.minimum(aerr_mat_bsc, aerr_mat_alt)
            loss_aval = torch.sum(self.prot_struct.amsk_mat * aerr_mat) / \
                (torch.sum(self.prot_struct.amsk_mat) + self.eps)
            loss_anrm = torch.mean(torch.abs(angl_tns_norm - 1.0))
            loss_list.append(loss_aval + 0.02 * loss_anrm)

        # take the averaged value of all the torsion angle prediction losses
        loss = torch.mean(torch.stack(loss_list))
        metrics = {'Loss-Angl': loss.item()}

        return loss, metrics


    def __calc_loss_lddt(self, plddt_list, cord_list):
        """Calculate the classification loss for per-residue lDDT-Ca predictions."""

        # calculate the classification loss for each layer's per-residue lDDT-Ca predictions
        n_lyrs = len(plddt_list)
        loss_list, metrics = [], {}
        for idx_lyr, (plddt_dict, cord_tns) in enumerate(zip(plddt_list, cord_list)):
            # calculate ground-truth per-residue lDDT-Ca scores
            plddt_vec_true, plmsk_vec, lddt_val_true = self.lddt_assessor.run(
                self.prot_struct.cord_tns, cord_tns, self.prot_struct.cmsk_mat, atom_set='ca')
            labl_vec = torch.clip(
                torch.floor(self.n_bins_lddt * plddt_vec_true).to(torch.int64),
                min=0, max=(self.n_bins_lddt - 1),
            )

            # calculate the classification loss
            if idx_lyr == n_lyrs - 1:  # only use the pLDDT loss at the final layer
                loss_vec = nn.CrossEntropyLoss(reduction='none')(plddt_dict['logit'], labl_vec)
                loss = torch.sum(plmsk_vec * loss_vec) / (torch.sum(plmsk_vec) + self.eps)
                loss_list.append(loss)
            #metrics[f'lDDT-L{idx_lyr + 1}'] = lddt_val_true.item()

        # loss function & evaluation metrics
        loss = torch.mean(torch.stack(loss_list))
        metrics['Loss-lDDT'] = loss.item()

        return loss, metrics


    def __calc_loss_tmsc(self, tmsc_dict, params, cord_tns):  # pylint: disable=too-many-locals
        """Calculate the classification loss for per-position TMscore predictions."""

        metrics = {}
        logits = tmsc_dict['ptm_logt']

        # extract 3D coordinates for CA-atom: L x 3 and coordinates mask L
        cord_mat_pred = ProtStruct.get_atoms(self.prot_struct.aa_seq, cord_tns, ['CA']).view(-1, 3)
        cord_mat_true = ProtStruct.get_atoms(
            self.prot_struct.aa_seq, self.prot_struct.cord_tns, ['CA']).view(-1, 3)

        # obtain local frames for the specified frame set
        rota_tns_bb = quat2rot(params['quat'])
        fram_tns_bb = torch.cat(
            [rota_tns_bb, params['trsl'].unsqueeze(dim=1)],
            dim=1).unsqueeze(dim=1)

        fram_tns_true = self.prot_struct.fram_tns_bb.view(-1, 4, 3)
        fram_tns_pred = fram_tns_bb.view(-1, 4, 3)
        fmsk_vec = self.prot_struct.fmsk_mat_bb.view(-1)

        # decompose per-residue local frames into rotation matrices & translation vectors
        rot_tns_true, tsl_mat_true = fram_tns_true[:, :3], fram_tns_true[:, 3]
        rot_tns_pred, tsl_mat_pred = fram_tns_pred[:, :3], fram_tns_pred[:, 3]

        # align 3D coordinates under all the per-residue local frames
        n_atoms = cord_mat_true.shape[0]
        n_frams = fram_tns_true.shape[0]
        cord_tns_true_aln = apply_trans(
            cord_mat_true, rot_tns_true, tsl_mat_true, reverse=True).view(n_frams, n_atoms, 3)
        cord_tns_pred_aln = apply_trans(
            cord_mat_pred, rot_tns_pred, tsl_mat_pred, reverse=True).view(n_frams, n_atoms, 3)

        sq_diff = torch.sum((cord_tns_true_aln - cord_tns_pred_aln) ** 2, dim=-1).detach()

        boundaries = torch.linspace(0, 31, steps=63, device=logits.device)
        boundaries = boundaries ** 2

        true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

        labl_vec = torch.nn.functional.one_hot(true_bins, 64)

        errors_vec = -torch.sum(labl_vec * torch.nn.functional.log_softmax(logits, dim=-1), dim=-1)
        square_mask = fmsk_vec[..., None] * fmsk_vec[..., None, :]

        # scale: help FP16 training along
        scale = 0.5
        loss = torch.sum(errors_vec * square_mask, dim=-1)
        denom = torch.sum(scale * square_mask, dim=(-1, -2)) + self.eps
        loss = loss / denom[..., None]
        loss = torch.sum(loss, dim=-1)
        loss = loss * scale

        # Average over the loss dimension
        loss = torch.mean(loss)

        metrics['Loss-TMscore'] = loss.item()

        return loss, metrics


    @classmethod
    def __calc_loss_qnrm(cls, params_list):
        """Calculate the L2-norm regularization loss for quaternion vector predictions."""

        # calculate the L2-norm regularization loss for each layer's quaternion vectors
        loss_list = []
        for params in params_list:
            quat_tns = params['quat-u']  # L x 4
            loss = torch.mean(torch.abs(torch.norm(quat_tns, dim=-1) - 1.0))
            loss_list.append(loss)

        # take the averaged value of all the L2-norm regularization losses
        loss = torch.mean(torch.stack(loss_list))
        metrics = {'Loss-QNrm': loss.item()}

        return loss, metrics


    def __calc_loss_clsh(self, cord_tns):
        """Calculate the structural violation loss (steric clashes among non-bonded atoms)."""

        # calculate the structural violation loss
        loss, metrics = self.clash_checker.run(
            self.prot_struct.aa_seq, cord_tns, self.prot_struct.cmsk_mat_vld, self.asym_id)
        metrics['Loss-Clsh'] = loss.item()

        return loss, metrics


    def __calc_loss_fape_impl(
            self, params, cord_tns, atom_set, fram_set,
            dist_clamp=10.0, use_clamp=True, loss_norm=None, fram_tns_sc=None, pos_msk=None,
        ):  # pylint: disable=too-many-arguments,too-many-locals
        """Calculate the frame aligned point error (FAPE) loss - core implementation.

        Args:
        * params: dict of QTA parameters (must contain 'quat', 'trsl', and 'angl')
          > quat: per-residue quaternion vectors of size L x 4 (full) / L x 3 (part)
          > trsl: per-residue translation vectors size size L x 3
          > angl: per-residue torsion angles of size L x K x 2
        * cord_tns: predicted per-atom 3D coordinates of size L x M x 3
        * atom_set: atom set (choices: 'ca' / 'fa')
        * fram_set: frame set (choices: 'bb' / 'bs')
        * dist_clamp: distance clamping threshold for Loss-FAPE
        * use_clamp: whether loss clamping is enabled
        * loss_norm: length scale by which the loss is divided (equal to dist_clamp if not set)
        * fram_tns_sc: (optional) final layer's per-residue side-chain frames of size L x K x 4 x 3
        * pos_msk: (optional) positions mask of size L x L, only for multimer predictions

        Returns:
        * loss: loss function
        """

        # initialization
        assert atom_set in ['ca', 'fa'], f'unrecognized atom set: {atom_set}'
        assert fram_set in ['bb', 'bs'], f'unrecognized frame set: {fram_set}'
        loss_norm = dist_clamp if loss_norm is None else loss_norm

        # obtain 3D coordinates for the specified atom set
        if atom_set == 'ca':
            atom_names = ['CA']
            cord_mat_true = ProtStruct.get_atoms(
                self.prot_struct.aa_seq, self.prot_struct.cord_tns, atom_names).view(-1, 3)
            cord_mat_pred = ProtStruct.get_atoms(
                self.prot_struct.aa_seq, cord_tns, atom_names).view(-1, 3)
            cmsk_vec = ProtStruct.get_atoms(
                self.prot_struct.aa_seq, self.prot_struct.cmsk_mat, atom_names).view(-1)
        else:  # then <atom_set> must be 'fa'
            cord_mat_true = self.prot_struct.cord_tns.view(-1, 3)  # (L x M) x 3
            cord_mat_pred = cord_tns.view(-1, 3)  # (L x M) x 3
            cmsk_vec = self.prot_struct.cmsk_mat.view(-1)  # (L x M)

        # obtain local frames for the specified frame set
        rota_tns_bb = quat2rot(params['quat'])
        fram_tns_bb = torch.cat(
            [rota_tns_bb, params['trsl'].unsqueeze(dim=1)], dim=1).unsqueeze(dim=1)
        if fram_set == 'bb':
            fram_tns_true = self.prot_struct.fram_tns_bb.view(-1, 4, 3)
            fram_tns_pred = fram_tns_bb.view(-1, 4, 3)
            fmsk_vec = self.prot_struct.fmsk_mat_bb.view(-1)
        else:  # then <fram_set> must be 'bs'
            assert fram_tns_sc is not None, 'side-chain local frames is not provided'
            fram_tns_true = torch.cat(
                [self.prot_struct.fram_tns_bb, self.prot_struct.fram_tns_sc], dim=1).view(-1, 4, 3)
            fram_tns_pred = torch.cat([fram_tns_bb, fram_tns_sc], dim=1).view(-1, 4, 3)
            fmsk_vec = torch.cat(
                [self.prot_struct.fmsk_mat_bb, self.prot_struct.fmsk_mat_sc], dim=1).view(-1)

        # decompose per-residue local frames into rotation matrices & translation vectors
        rot_tns_true, tsl_mat_true = fram_tns_true[:, :3], fram_tns_true[:, 3]
        rot_tns_pred, tsl_mat_pred = fram_tns_pred[:, :3], fram_tns_pred[:, 3]

        # align 3D coordinates under all the per-residue local frames
        n_atoms = cord_mat_true.shape[0]
        n_frams = fram_tns_true.shape[0]
        cord_tns_true_aln = apply_trans(
            cord_mat_true, rot_tns_true, tsl_mat_true, reverse=True).view(n_frams, n_atoms, 3)
        cord_tns_pred_aln = apply_trans(
            cord_mat_pred, rot_tns_pred, tsl_mat_pred, reverse=True).view(n_frams, n_atoms, 3)

        # determine weighting coefficients for all the atoms & local frames
        n_resds = self.conf_vec.shape[0]
        fwei_vec = (fmsk_vec.view(n_resds, -1) * self.conf_vec.unsqueeze(dim=1)).view(-1)
        cwei_vec = (cmsk_vec.view(n_resds, -1) * self.conf_vec.unsqueeze(dim=1)).view(-1)

        # calculate the FAPE loss
        dwei_mat = torch.outer(fwei_vec, cwei_vec)
        if pos_msk is not None:  # calculate interface and inter-chain FAPE loss
            dwei_mat = dwei_mat * pos_msk
        dist_mat = torch.sqrt(
            torch.sum(torch.square(cord_tns_true_aln - cord_tns_pred_aln), dim=-1) + self.eps)
        dist_mat_clip = torch.clip(dist_mat, 0.0, dist_clamp) if use_clamp else dist_mat
        dist_mat_norm = dist_mat_clip / loss_norm
        loss = torch.sum(dwei_mat * dist_mat_norm) / (torch.sum(dwei_mat) + self.eps)

        return loss


    def __calc_drmsd(self, cord_tns, atom_set):
        """Calculate the distance RMSD (root-mean-square deviation).

        Args:
        * cord_tns: predicted per-atom 3D coordinates of size L x M x 3
        * atom_set: atom set (choices: 'ca' / 'fa')

        Returns:
        * drmsd: distance RMSD (root-mean-square deviation)
        """

        # obtain 3D coordinates for the specified atom set
        if atom_set == 'ca':
            atom_names = ['CA']
            cord_mat_true = ProtStruct.get_atoms(
                self.prot_struct.aa_seq, self.prot_struct.cord_tns, atom_names).view(-1, 3)
            cord_mat_pred = ProtStruct.get_atoms(
                self.prot_struct.aa_seq, cord_tns, atom_names).view(-1, 3)
            cmsk_vec = ProtStruct.get_atoms(
                self.prot_struct.aa_seq, self.prot_struct.cmsk_mat, atom_names).view(-1)
        else:  # then <atom_set> must be 'fa'
            cord_mat_true = self.prot_struct.cord_tns.view(-1, 3)  # (L x M) x 3
            cord_mat_pred = cord_tns.view(-1, 3)  # (L x M) x 3
            cmsk_vec = self.prot_struct.cmsk_mat.view(-1)  # (L x M)

        # calculate the distance RMSD (root-mean-square deviation)
        dist_mat_true = cdist(cord_mat_true)
        dist_mat_pred = cdist(cord_mat_pred)
        dmsk_mat = torch.outer(cmsk_vec, cmsk_vec)
        diff_mat = torch.abs(dist_mat_pred - dist_mat_true)
        drmsd = torch.sum(dmsk_mat * diff_mat) / (torch.sum(dmsk_mat) + self.eps)

        return drmsd.item()


    def __calc_loss_ifape(self, params, cord_tns):
        """Calculate the Loss of intra_chain and interface frame aligned point error (iFAPE)"""

        # initialization
        metrics = {}

        # calculate the intra_chain FAPE loss w/ CA-atom and backbone frames
        intra_chain_mask = (self.asym_id[:, None] == self.asym_id[None, :]).float()
        loss_intra_chain = self.__calc_loss_fape_impl(
            params, cord_tns, atom_set='ca', fram_set='bb',
            dist_clamp=self.dist_clamp_ca, use_clamp=self.use_clamp_dict['ifape'],
            loss_norm=10.0, pos_msk=intra_chain_mask,
        )

        # calculate the interface FAPE loss w/ CA-atom and backbone frames
        interface_mask = 1.0 - intra_chain_mask
        loss_interface = self.__calc_loss_fape_impl(
            params, cord_tns, atom_set='ca', fram_set='bb',
            dist_clamp=self.dist_clamp_ica, use_clamp=self.use_clamp_dict['ifape'],
            loss_norm=20.0, pos_msk=interface_mask,
        )

        # calculate the iFAPE loss
        loss = loss_intra_chain + loss_interface
        metrics.update({
            'Loss-iFAPE': loss.item(),
            'Loss-iFAPE-intra_chain': loss_intra_chain.item(),
            'Loss-iFAPE-interface': loss_interface.item()
        })

        return loss, metrics


    def __calc_loss_itfape(self, params, cord_tns):
        """Calculate the Loss of interface frame aligned point error (itFAPE)"""

        # initialization
        metrics = {}

        # calculate the itFAPE loss w/ CA-atom and backbone frames
        intra_chain_mask = (self.itef_id[:, None] == self.itef_id[None, :]).float()
        interface_mask = 1.0 - intra_chain_mask
        loss = self.__calc_loss_fape_impl(
            params, cord_tns, atom_set='ca', fram_set='bb',
            dist_clamp=self.dist_clamp_ica, use_clamp=self.use_clamp_dict['ifape'],
            loss_norm=20.0, pos_msk=interface_mask,
        )

        metrics.update({
            'Loss-itFAPE': loss.item(),
        })

        return loss, metrics
