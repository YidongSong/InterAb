# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
import logging
import time
from collections import OrderedDict

import torch

from tfold.third_parties.tfold_released.model import AgModel, PPIModel
from tfold.third_parties.tfold_released.protein import ProtStruct, export_fasta
from tfold.third_parties.tfold_released.protein.data_transform import get_asym_ids
from tfold.third_parties.tfold_released.protein.parser import PdbParser
from tfold.third_parties.tfold_released.protein.utils import get_mlm_masks
from tfold.third_parties.tfold_released.utils import to_device
from .base_structure_predictor import BaseStructurePredictor
from .psp_featurizer import PspFeaturizer


class AgPredictor(BaseStructurePredictor):
    """The antibody & antigen multimer structure predictor.
    """

    def __init__(self, ppi_path, psp_path, ag_path):
        super().__init__()
        logging.info('restoring the pre-trained tFold-PPI-SeqPT model ...')
        self.plm_featurizer = PPIModel.restore(ppi_path)
        logging.info('restoring the pre-trained AlphaFold2 model ...')
        self.psp_featurizer = PspFeaturizer.restore(psp_path)
        logging.info('restoring the pre-trained tFold-Ag model ...')
        self.model = AgModel.restore(ag_path)
        self.eval()

    def _build_inputs(self, chains, ic_feat=None):
        """Build an input dict for model inference."""
        num_chains = len(chains)
        inputs = super()._build_inputs(chains)
        if num_chains == 2:  # nanobody, two chains, H and A
            ligand_id = 'H'
        else:  # antibody, three chains, H, L, A
            ligand_id = ':'.join(['H', 'L'])  # pseudo chain ID for the VH-VL complex
            h_seq = inputs["H"]["base"]["seq"]
            l_seq = inputs["L"]["base"]["seq"]
            inputs[ligand_id] = {
                'base': {'seq': h_seq + l_seq},
                'asym_id': get_asym_ids([h_seq, l_seq]).unsqueeze(dim=0),
                'feat': {},
            }

        inputs["base"]["ligand_id"] = ligand_id
        inputs["base"]["receptor_id"] = "A"

        # prepare for sequence recovery
        if 'X' in inputs[ligand_id]["base"]["seq"]:
            mask_vec = torch.LongTensor([resd == 'X' for resd in inputs[ligand_id]['base']['seq']])
            inputs[ligand_id]['base']['mask'] = get_mlm_masks(
                [inputs[ligand_id]['base']['seq']], mask_prob=1, mask_vecs=[mask_vec, ])

            if num_chains == 3:
                inputs['H']['base']['mask'] = inputs[ligand_id]['base']['mask'][:len(inputs['H']['base']['seq'])]
                inputs['L']['base']['mask'] = inputs[ligand_id]['base']['mask'][len(inputs['H']['base']['seq']):]

        complex_id = ':'.join([ligand_id, 'A'])  # H:A or H:L:A
        aa_seq_ligand = inputs[ligand_id]['base']['seq']
        aa_seq_receptor = inputs['A']['base']['seq']
        inputs[complex_id] = {
            'base': {'seq': aa_seq_ligand + aa_seq_receptor},
            'asym_id': get_asym_ids([aa_seq_ligand, aa_seq_receptor]).unsqueeze(dim=0),
            'feat': {},
        }
        inputs["base"]["complex_id"] = complex_id
        if ic_feat is not None:
            ic_feat = ic_feat.unsqueeze(dim=0).unsqueeze(dim=-1).float()
            inputs[complex_id]['feat']['icf'] = ic_feat

        return inputs

    def forward(self, inputs, chunk_size=None):
        """Run the antibody & antigen multimer structure predictor.
        """
        start = time.time()
        inputs = to_device(inputs, device=self.device)
        ligand_id = inputs["base"]["ligand_id"]
        ligand_seqs = [inputs[x]['base']['seq'] for x in ligand_id.split(':')]  # H or H:L chains

        if 'mask' not in inputs[ligand_id]['base']:
            plm_outputs = self.plm_featurizer(ligand_seqs)
        else:
            mask_vecs = [inputs[x]['base']['mask'] for x in ligand_id.split(':')]
            plm_outputs = self.plm_featurizer(ligand_seqs, mask_prob=1, mask_vecs=mask_vecs)

        inputs[ligand_id]['feat']['sfea'] = plm_outputs['sfea']
        inputs[ligand_id]['feat']['pfea'] = plm_outputs['pfea']

        logging.info('[%s] antigen feature predicted using pretrained structure prediction model')
        inputs["A"]["feat"] = self.psp_featurizer(inputs["A"]["base"]["msa"],
                                                  inputs["A"]["base"]["deletion_matrix"],
                                                  )[0]
        logging.info('start ag model in %.2f second', time.time() - start)
        outputs = self.model(inputs, chunk_size=chunk_size)

        return outputs

    @torch.no_grad()
    def infer(self, chains, ic_feat=None, *args, **kwargs):
        assert all(x["id"] in {'H', 'L', 'A'} for x in chains), 'chain ID must be "H", "L" or "A"'
        assert len(chains) in (2, 3), f'FASTA file should contain 2 or 3 chains'
        inputs = self._build_inputs(chains, ic_feat=ic_feat)
        outputs = self.forward(inputs, *args, **kwargs)
        return inputs, outputs

    def infer_pdb(self, chains, filename, *args, **kwargs):
        inputs, outputs = self.infer(chains, *args, **kwargs)
        complex_id = inputs["base"]["complex_id"]
        raw_seqs = {}
        for chain_id in complex_id.split(":"):
            raw_seq = inputs[chain_id]["base"]["seq"]
            raw_seqs[chain_id] = raw_seq

        if any(['X' in x for x in raw_seqs.values()]):
            self.output_to_fasta(inputs, outputs, filename[:-3] + ".fasta")

        self.output_to_pdb(inputs, outputs, filename)

    def output_to_pdb(self, inputs, outputs, filename):
        """Build a dict of protein structure data."""
        pred_info = 'REMARK 250 Structure predicted by tFold-Ag\n'
        complex_id = inputs["base"]["complex_id"]

        prot_data = OrderedDict()
        start = 0
        for chn_id in complex_id.split(":"):
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs[complex_id]['1d']['seq'][start:start + len(aa_seq)]
                pred_info += f'REMARK 250 Predicted Sequence for chain {chn_id}: {aa_seq}\n'

            prot_data[chn_id] = {
                'seq': aa_seq,
                'cord': outputs[complex_id]['3d']['cord'][-1][start:start + len(aa_seq)],
                'cmsk': ProtStruct.get_cmsk_vld(aa_seq, self.device),
                'plddt': outputs[complex_id]['3d']['plddt'][-1]['plddt-r'][start:start + len(aa_seq)],
            }
            start += len(aa_seq)

        receptor_id = inputs["base"]["receptor_id"]
        iptm_score = outputs[complex_id]['3d']['tmsc_dict']['iptm'].item()
        ptm_score = outputs[complex_id]['3d']['tmsc_dict']['ptm'].item()
        pred_info += f'REMARK 250 Predicted ipTM score: {iptm_score:.4f}\n'
        pred_info += f'REMARK 250 Predicted pTM score: {ptm_score:.4f}\n'
        plddt_receptor = torch.mean(prot_data[receptor_id]['plddt']).item()
        pred_info += f'REMARK 250 Antigen Predicted lDDT-Ca score: {plddt_receptor:.4f}\n'
        PdbParser.save_multimer(prot_data, filename, pred_info=pred_info)
        logging.info(f'PDB file generated: {filename}')

    def output_to_fasta(self, inputs, outputs, filename):
        """export the predicted sequence to a FASTA file"""
        complex_id = inputs["base"]["complex_id"]
        ligand_id = inputs["base"]["ligand_id"]
        ligand_ids = ligand_id.split(':')
        start = 0
        sequences = []
        ids = []
        descriptions = []
        for chn_id in ligand_ids:
            aa_seq = inputs[chn_id]['base']['seq']
            if 'X' in aa_seq:  # for sequence recovery
                aa_seq = outputs[complex_id]['1d']['seq'][start:start + len(aa_seq)]
                plddt = outputs[complex_id]['3d']['plddt'][-1]['plddt-r'][start:start + len(aa_seq)].detach()
                mask = inputs[ligand_id]['base']['mask'][start:start + len(aa_seq)]
                region_plddt = torch.sum(plddt * mask) / (torch.sum(mask) + 1e-6)
                desc = f"design confidence {region_plddt:.3f}"
            else:
                desc = ""

            descriptions.append(desc)
            sequences.append(aa_seq)
            ids.append(chn_id)
            start += len(aa_seq)

        export_fasta(sequences, ids=ids, descriptions=descriptions, output=filename)
