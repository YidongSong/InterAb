import pickle, os
import numpy as np
import torch
from tqdm import tqdm
from Bio import pairwise2
import os
import pdb 
import pandas as pd


monomer_path = './data/Geo_data/monomer_pdb/'
monomer_tensor = './data/Geo_data/monomer_data/'
dssp_save = './data/Geo_data/dssp/'
dssp_path = "./tools/dssp-2.0.4/"
os.makedirs(os.path.dirname(monomer_tensor), exist_ok=True)
os.makedirs(os.path.dirname(dssp_save), exist_ok=True)


########## Process PDB ##########
def get_pdb_xyz(pdb_file):
    current_pos = -1000
    X = []
    current_aa = {} # N, CA, C, O, R
    for line in pdb_file:
        if (line[0:4].strip() == "ATOM" and int(line[22:26].strip()) != current_pos) or line[0:4].strip() == "TER":
            if current_aa != {}:
                R_group = []
                for atom in current_aa:
                    if atom not in ["N", "CA", "C", "O"]:
                        R_group.append(current_aa[atom])
                if R_group == []:
                    R_group = [current_aa["CA"]]
                R_group = np.array(R_group).mean(0)
                X.append([current_aa["N"], current_aa["CA"], current_aa["C"], current_aa["O"], R_group])
                current_aa = {}
            if line[0:4].strip() != "TER":
                current_pos = int(line[22:26].strip())

        if line[0:4].strip() == "ATOM":
            atom = line[13:16].strip()
            if atom != "H":
                xyz = np.array([line[30:38].strip(), line[38:46].strip(), line[46:54].strip()]).astype(np.float32)
                current_aa[atom] = xyz
    return np.array(X)


########## Get DSSP ##########
def process_dssp(dssp_file):
    aa_type = "ACDEFGHIKLMNPQRSTVWY"
    SS_type = "HBEGITSC"
    rASA_std = [115, 135, 150, 190, 210, 75, 195, 175, 200, 170,
                185, 160, 145, 180, 225, 115, 140, 155, 255, 230]

    with open(dssp_file, "r") as f:
        lines = f.readlines()

    seq = ""
    dssp_feature = []

    p = 0
    while lines[p].strip()[0] != "#":
        p += 1
    for i in range(p + 1, len(lines)):
        aa = lines[i][13]
        if aa == "!" or aa == "*":
            continue
        seq += aa
        SS = lines[i][16]
        if SS == " ":
            SS = "C"
        SS_vec = np.zeros(8)
        SS_vec[SS_type.find(SS)] = 1
        ACC = float(lines[i][34:38].strip())
        ASA = min(1, ACC / rASA_std[aa_type.find(aa)])
        dssp_feature.append(np.concatenate((np.array([ASA]), SS_vec)))

    return seq, dssp_feature


def match_dssp(seq, dssp, ref_seq):
    alignments = pairwise2.align.globalxx(ref_seq, seq)
    ref_seq = alignments[0].seqA
    seq = alignments[0].seqB

    padded_item = np.zeros(9)

    new_dssp = []
    for aa in seq:
        if aa == "-":
            new_dssp.append(padded_item)
        else:
            new_dssp.append(dssp.pop(0))

    matched_dssp = []
    for i in range(len(ref_seq)):
        if ref_seq[i] == "-":
            continue
        matched_dssp.append(new_dssp[i])

    return matched_dssp


def get_dssp(ID, ref_seq):
    os.system("{}mkdssp -i {}{}.pdb -o {}{}.dssp".format(dssp_path,monomer_path, ID,dssp_save,ID))
    dssp_seq, dssp_matrix = process_dssp("{}{}.dssp".format(dssp_save,ID))
    if dssp_seq != ref_seq:
        dssp_matrix = match_dssp(dssp_seq, dssp_matrix, ref_seq)

    torch.save(torch.tensor(dssp_matrix, dtype = torch.float32), "{}{}.tensor".format(dssp_save, ID))
    os.system("rm {}{}.dssp".format(dssp_save, ID))

def prepare_geo_data():
    all_ID = []
    for root,dirs,files in os.walk(monomer_path):
        for ele in files:
            name = ele.replace('.pdb', '')
            all_ID.append(name)

    # Replace with your sequence data
    seq_data = pickle.load(open('./data/Geo_data/exa_data.pkl', 'rb'))

    for ID in tqdm(all_ID):
        try:
            with open(monomer_path + ID + ".pdb", "r") as f:
                X = get_pdb_xyz(f.readlines()) # [L, 5, 3]
            torch.save(torch.tensor(X, dtype = torch.float32), monomer_tensor + ID + '.tensor')
        except:
            print(ID)
            continue

    for ID in tqdm(all_ID):
            
            ref_seq = seq_data[ID]
            try:
                get_dssp(ID, ref_seq)
            except:
                print(ID)
