import torch
import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
import pickle
import torch_geometric
import pdb


def geo_data(batch_name):
    ab_h_X = {}
    ab_h_seq = {}
    ab_l_seq = {}
    ag_seq = {}
    ab_l_X = {}
    ag_X = {}
    ab_h_esm2 = {}
    ab_l_esm2 = {}
    ag_esm2 = {}
    ab_h_dssp_feat = {}
    ab_l_dssp_feat = {}
    ag_dssp_feat = {}
    ab_h_node_feat = {}
    ab_l_node_feat = {}
    ag_node_feat = {}
    ab_h_X_ca = {}
    ab_l_X_ca = {}
    ag_X_ca = {}
    ab_h_edge_index = {}
    ab_l_edge_index = {}
    ag_edge_index = {}
    ab_h_batch_id = []
    ab_l_batch_id = []
    ag_batch_id = []
    ab_h_edge_len = []
    ab_l_edge_len = []
    ag_edge_len = []
    radius = 10
    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
    
    # Replace with your sequence data
    with open("Replace with your sequence data", "rb") as f:
            dataset = pickle.load(f)
    
    monomer_tensor = '../data/Geo_data/monomer_data/'
    esm2_path = '../data/Geo_data/esm2'
    dssp_path = '../data/Geo_data/dssp'
    for i in tqdm(range(len(batch_name))):
        ab_h_X_i = torch.load(monomer_tensor + batch_name[i].split('_')[0] + "_h.tensor")
        ab_h_X[batch_name[i]] = ab_h_X_i

        ab_h_seq[batch_name[i]] = torch.tensor([letter_to_num[aa] for aa in dataset[batch_name[i]]['h']], dtype=torch.long)

        ab_h_esm2_i = torch.load(esm2_path + batch_name[i] + "_h.pt")['representations'][33]
        ab_h_esm2[batch_name[i]] = ab_h_esm2_i
        
        ab_h_dssp_feat_i = torch.load(dssp_path + batch_name[i].split('_')[0] + "_h.tensor")    
        ab_h_dssp_feat[batch_name[i]] = ab_h_dssp_feat_i
            
        ab_h_node_feat_i = torch.cat([ab_h_esm2_i, ab_h_dssp_feat_i], dim=-1)
        ab_h_node_feat[batch_name[i]] = ab_h_node_feat_i
        
        ab_h_X_ca_i = ab_h_X_i[:, 1]
        ab_h_X_ca[batch_name[i]] = ab_h_X_ca_i
        
        ab_h_edge_index_i = torch_geometric.nn.radius_graph(ab_h_X_ca_i, r=radius, loop=True, max_num_neighbors = 1000, num_workers = 4)
        ab_h_edge_index[batch_name[i]] = ab_h_edge_index_i
    
        ab_l_X_i = torch.load(monomer_tensor + batch_name[i] + "_l.tensor")
        ab_l_X[batch_name[i]] = ab_l_X_i
        
        ab_l_seq[batch_name[i]] = torch.tensor([letter_to_num[aa] for aa in dataset[batch_name[i]]['l']], dtype=torch.long)
        
        ab_l_esm2_i = torch.load(esm2_path + batch_name[i] + "_l.pt")['representations'][33]
        ab_l_esm2[batch_name[i]] = ab_l_esm2_i
        
        ab_l_dssp_feat_i = torch.load(dssp_path + batch_name[i] + "_l.tensor")
        ab_l_dssp_feat[batch_name[i]] = ab_l_dssp_feat_i
        
        ab_l_node_feat_i = torch.cat([ab_l_esm2_i, ab_l_dssp_feat_i], dim=-1)
        ab_l_node_feat[batch_name[i]] = ab_l_node_feat_i
        
        ab_l_X_ca_i = ab_l_X_i[:, 1]
        ab_l_X_ca[batch_name[i]] = ab_l_X_ca_i
        
        ab_l_edge_index_i = torch_geometric.nn.radius_graph(ab_l_X_ca_i, r=radius, loop=True, max_num_neighbors = 1000, num_workers = 4)
        ab_l_edge_index[batch_name[i]] = ab_l_edge_index_i
                 
        ag_X_i = torch.load(monomer_tensor + batch_name[i] + "_ag.tensor")
        ag_X[batch_name[i]] = ag_X_i
    
        ag_seq[batch_name[i]] = torch.tensor([letter_to_num[aa] for aa in dataset[batch_name[i]]['ag']], dtype=torch.long)

        ag_esm2_i = torch.load(esm2_path + batch_name[i] + "_ag.pt")['representations'][33]
        ag_esm2[batch_name[i]] = ag_esm2_i
        
        ag_dssp_feat_i = torch.load(dssp_path + batch_name[i] + "_ag.tensor")
        ag_dssp_feat[batch_name[i]] = ag_dssp_feat_i
        
        ag_node_feat_i = torch.cat([ag_esm2_i, ag_dssp_feat_i], dim=-1)
        ag_node_feat[batch_name[i]] = ag_node_feat_i
        
        ag_X_ca_i = ag_X_i[:, 1]
        ag_X_ca[batch_name[i]] = ag_X_ca_i
        
        ag_edge_index_i = torch_geometric.nn.radius_graph(ag_X_ca_i, r=radius, loop=True, max_num_neighbors = 1000, num_workers = 4)
        ag_edge_index[batch_name[i]] = ag_edge_index_i
    
       
    data_dict = {}
    data_dict['ab_h_X'] = ab_h_X
    data_dict['ab_l_X'] = ab_l_X
    data_dict['ag_X'] = ag_X
    data_dict['ab_h_node_feat'] = ab_h_node_feat
    data_dict['ab_l_node_feat'] = ab_l_node_feat
    data_dict['ag_node_feat'] = ag_node_feat
    data_dict['ab_h_edge_index'] = ab_h_edge_index
    data_dict['ab_l_edge_index'] = ab_l_edge_index
    data_dict['ag_edge_index'] = ag_edge_index
    data_dict['ab_h_seq'] = ab_h_seq
    data_dict['ab_l_seq'] = ab_l_seq
    data_dict['ag_seq'] = ag_seq
    
    return data_dict

# Please replace with your data path
data = pickle.load(open('your_data_path', 'rb'))
name = []
for key in data.keys():
     name.append(key)

data_dict = geo_data(name)
pickle.dump(data_dict, open('./data/Geo_data/geo_data.pkl', 'wb'))