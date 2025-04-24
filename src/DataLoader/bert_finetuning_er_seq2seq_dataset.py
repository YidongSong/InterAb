# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd
from os.path import join, exists
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from Tokenizer.tokenizer import get_tokenizer
from Tokenizer.utility import is_valid_aaseq
import copy
from DataLoader import BaseDataLoader
from transformers import AutoTokenizer


class Antibody_Antigen_Dataset_AbDab(BaseDataLoader):
    def __init__(self, logger, 
                       seed,
                       batch_size,
                       validation_split,
                       test_split,
                       num_workers,
                       data_dir, 
                       antibody_vocab_dir,
                       antibody_tokenizer_dir,
                       tokenizer_name='common',
                       #receptor_tokenizer_name='common',
                       token_length_list='2,3',
                       #receptor_token_length_list='2,3',
                       protein_name='pdb',
                       antigen_seq_name='antigen',
                       heavy_seq_name='Heavy',
                       light_seq_name='Light',
                       label_name='Label',
                       #receptor_seq_name='beta',
                       test_antibodys=100,
                       #neg_ratio=1.0,
                       shuffle=True,
                       antigen_max_len=None,
                       heavy_max_len=None,
                       light_max_len=None,):
        self.logger = logger
        self.seed = seed
        self.data_dir = data_dir
        self.protein_name = protein_name
        self.heavy_seq_name = heavy_seq_name
        self.light_seq_name = light_seq_name
        self.antigen_seq_name = antigen_seq_name
        self.label_name = label_name

        self.test_antibodys = test_antibodys
        self.shuffle = shuffle
        self.heavy_max_len = heavy_max_len
        self.light_max_len = light_max_len
        self.antigen_max_len = antigen_max_len

        self.rng = np.random.default_rng(seed=self.seed)

        self.pair_df = self._create_pair()

        
        self.HeavyTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=token_length_list)
        self.heavy_tokenizer = self.HeavyTokenizer.get_bert_tokenizer(
            max_len=self.heavy_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)



        self.LightTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                              add_hyphen=False,
                                              logger=self.logger,
                                              vocab_dir=antibody_vocab_dir,
                                              token_length_list=token_length_list)
        self.light_tokenizer = self.LightTokenizer.get_bert_tokenizer(
            max_len=self.light_max_len, 
            tokenizer_dir=antibody_tokenizer_dir)

        
        self.AntigenTokenizer = get_tokenizer(tokenizer_name=tokenizer_name,
                                               add_hyphen=False,
                                               logger=self.logger,
                                               vocab_dir=antibody_vocab_dir,
                                               token_length_list=token_length_list)
        self.antigen_tokenizer = self.AntigenTokenizer.get_bert_tokenizer(
            max_len=self.antigen_max_len,
            tokenizer_dir=antibody_tokenizer_dir)


        esm_dir = 'facebook/esm2_t30_150M_UR50D'
        self.antigen_tokenizer = AutoTokenizer.from_pretrained(esm_dir,cache_dir = "./ESM_models/esm2/esm2_150m/",max_len = self.antigen_max_len)
        
        dataset = self._get_dataset(pair_df=self.pair_df)
        super().__init__(dataset, batch_size, seed, shuffle, validation_split, test_split,
                         num_workers)


    def get_heavy_tokenizer(self):
        return self.heavy_tokenizer

    def get_light_tokenizer(self):
        return self.light_tokenizer

    def get_antibody_tokenizer(self):
        return self.heavy_tokenizer

    def get_antigen_tokenizer(self):
        return self.antigen_tokenizer

    def get_test_dataloader(self):
        return self.test_dataloader

    def _get_dataset(self, pair_df):
        # print(pair_df)
        abag_dataset = AbAGDataset_CovAbDab(
                                        name = pair_df[self.protein_name],
                                        heavy_seqs = list(pair_df[self.heavy_seq_name]),
                                        light_seqs = list(pair_df[self.light_seq_name]),
                                        antigen_seqs = list(pair_df[self.antigen_seq_name]),
                                        # labels = list(pair_df[self.label_name]),
                                        antibody_split_fun = self.HeavyTokenizer.split,
                                        antigen_split_fun = self.AntigenTokenizer.split,
                                        antibody_tokenizer = self.heavy_tokenizer,
                                        antigen_tokenizer = self.antigen_tokenizer,
                                        antibody_max_len = self.heavy_max_len,
                                        antigen_max_len = self.antigen_max_len,
                                        logger = self.logger
                               )
        return abag_dataset

    def _split_dataset(self):
        # if exists(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv')):
        #     test_pair_df = pd.read_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'))
        #     self.logger.info(f'Loading created unseen antibodys for test with shape {test_pair_df.shape}')
        
        antibody_list = list(set(self.pair_df['antibody']))
        selected_antibody_index_list = self.rng.integers(len(antibody_list), size=self.test_antibodys)
        self.logger.info(f'Select {self.test_antibodys} from {len(antibody_list)} antibody')
        selected_antibodys = [antibody_list[i] for i in selected_antibody_index_list]
        test_pair_df = self.pair_df[self.pair_df['antibody'].isin(selected_antibodys)]
        #test_pair_df.to_csv(join(self.neg_pair_save_dir, 'unseen_antibodys-seed-'+str(self.seed)+'.csv'), index=False)

        selected_antibodys = list(set(test_pair_df['antibody']))
        train_valid_pair_df = self.pair_df[~self.pair_df['antibody'].isin(selected_antibodys)]
            
        self.logger.info(f'{len(train_valid_pair_df)} pairs for train and valid and {len(test_pair_df)} pairs for test.')

        return train_valid_pair_df, test_pair_df

    def _create_pair(self):
        pair_df = pd.read_csv(self.data_dir)

        if self.shuffle:
            pair_df = pair_df.sample(frac=1).reset_index(drop=True)
            self.logger.info("Shuffling dataset")
        self.logger.info(f"There are {len(pair_df)} samples")

        return pair_df

    def _load_seq_pairs(self):
        self.logger.info(f'Loading from {self.using_dataset}...')
        self.logger.info(f'Loading {self.antibody_seq_name} and {self.receptor_seq_name}')
        column_map_dict = {'alpha': 'cdr3a', 'beta': 'cdr3b', 'antibody': 'antibody'}
        keep_columns = [column_map_dict[c] for c in [self.antibody_seq_name, self.receptor_seq_name]]
        
        df_list = []
        for dataset in self.using_dataset:
            df = pd.read_csv(join(self.data_dir, dataset, 'full.csv'))
            df = df[keep_columns]
            df = df[(df[keep_columns[0]].map(is_valid_aaseq)) & (df[keep_columns[1]].map(is_valid_aaseq))]
            self.logger.info(f'Loading {len(df)} pairs from {dataset}')
            df_list.append(df[keep_columns])
        df = pd.concat(df_list)
        self.logger.info(f'Current data shape {df.shape}')
        df_filter = df.dropna()
        df_filter = df_filter.drop_duplicates()
        self.logger.info(f'After dropping na and duplicates, current data shape {df_filter.shape}')

        column_rename_dict = {column_map_dict[c]: c for c in [self.antibody_seq_name, self.receptor_seq_name]}
        df_filter.rename(columns=column_rename_dict, inplace=True)

        df_filter['label'] = [1] * len(df_filter)
        df_filter.to_csv(join(self.neg_pair_save_dir, 'pos_pair.csv'), index=False)

        return df_filter

class AbAGDataset_CovAbDab(Dataset):
    def __init__(self, name,
                       heavy_seqs,
                       light_seqs,
                       antigen_seqs,
                    #    labels,
                       antibody_split_fun,
                       antigen_split_fun,
                       antibody_tokenizer,
                       antigen_tokenizer,
                       antibody_max_len,
                       antigen_max_len,
                       logger):
        self.name = name
        self.heavy_seqs = heavy_seqs
        self.light_seqs = light_seqs
        self.antigen_seqs = antigen_seqs
        # self.labels = labels
        self.antibody_split_fun = antibody_split_fun
        self.antigen_split_fun = antigen_split_fun
        self.antibody_tokenizer = antibody_tokenizer
        self.antigen_tokenizer = antigen_tokenizer
        self.antibody_max_len = antibody_max_len
        self.antigen_max_len = antigen_max_len
        self.logger = logger
        self._has_logged_example = False

    def __len__(self):
        return len(self.heavy_seqs)
        
    def __getitem__(self, i):
        protein_name, heavy,light,antigen = self.name[i], self.heavy_seqs[i], self.light_seqs[i] , self.antigen_seqs[i]
        #label = self.labels[i]
        heavy_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(heavy)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt")
        light_tensor = self.antibody_tokenizer(self._insert_whitespace(self.antibody_split_fun(light)),
                                                padding="max_length",
                                                max_length=self.antibody_max_len,
                                                truncation=True,
                                                return_tensors="pt")

  
        antigen_tensor = self.antigen_tokenizer(antigen,
                                                  padding="max_length",
                                                  max_length=self.antigen_max_len,
                                                  truncation=True,
                                                  return_tensors="pt")

        

        # label_tensor = torch.FloatTensor(np.atleast_1d(label))


        heavy_tensor = {k: torch.squeeze(v) for k, v in heavy_tensor.items()}
        light_tensor = {k: torch.squeeze(v) for k, v in light_tensor.items()}
        antigen_tensor = {k: torch.squeeze(v) for k,v in antigen_tensor.items()}
        return protein_name, heavy_tensor, light_tensor, antigen_tensor#, label_tensor




    def _insert_whitespace(self, token_list):
        """
        Return the sequence of tokens with whitespace after each char
        """
        return " ".join(token_list)

def Geo_data(geo_data, name):
    ab_h_X = None
    ab_l_X = None
    ag_X = None
    ab_h_esm2 = None
    ab_l_esm2 = None
    ag_esm2 = None
    ab_h_dssp_feat = None
    ab_l_dssp_feat = None
    ag_dssp_feat = None
    ab_h_node_feat = None
    ab_l_node_feat = None
    ag_node_feat = None
    ab_h_X_ca = None
    ab_l_X_ca = None
    ag_X_ca = None
    ab_h_seq = None
    ab_l_seq = None
    ag_seq = None
    ab_h_batch_id = []
    ab_l_batch_id = []
    ag_batch_id = []
    ab_h_edge_len = []
    ab_l_edge_len = []
    ag_edge_len = []
    
    for i in range(len(name)):
        ab_h_X_i = geo_data['ab_h_X'][name[i]]
        if ab_h_X == None:
            ab_h_X = ab_h_X_i
        else:
            ab_h_X = torch.cat([ab_h_X, ab_h_X_i], dim=0)

        ab_h_batch_id.extend([i]*ab_h_X_i.shape[0])

        ab_h_seq_i = geo_data['ab_h_seq'][name[i]]
        if ab_h_seq == None:
            ab_h_seq = ab_h_seq_i
        else:
            ab_h_seq = torch.cat([ab_h_seq, ab_h_seq_i])
        
        ab_h_node_feat_i = geo_data['ab_h_node_feat'][name[i]]
        if ab_h_node_feat == None:
            ab_h_node_feat = ab_h_node_feat_i
        else:
            ab_h_node_feat = torch.cat([ab_h_node_feat, ab_h_node_feat_i], dim=0)
        
        ab_h_edge_len.append(ab_h_X_i.shape[0])
        ab_h_edge_index_i = copy.deepcopy(geo_data['ab_h_edge_index'][name[i]])
        if i == 0:
            ab_h_edge_index = ab_h_edge_index_i
        else:
            gap = 0
            for j in range(i):
                gap += ab_h_edge_len[j]
            ab_h_edge_index_i[0]  += gap
            ab_h_edge_index_i[1]  += gap
            ab_h_edge_index = torch.cat([ab_h_edge_index, ab_h_edge_index_i], dim=-1)
    
        ab_l_X_i = geo_data['ab_l_X'][name[i]]
        if ab_l_X == None:
            ab_l_X = ab_l_X_i
        else:
            ab_l_X = torch.cat([ab_l_X, ab_l_X_i], dim=0)
        
        ab_l_batch_id.extend([i]*ab_l_X_i.shape[0])

        ab_l_seq_i = geo_data['ab_l_seq'][name[i]]
        if ab_l_seq == None:
            ab_l_seq = ab_l_seq_i
        else:
            ab_l_seq = torch.cat([ab_l_seq, ab_l_seq_i])
        
        ab_l_node_feat_i = geo_data['ab_l_node_feat'][name[i]]
        if ab_l_node_feat == None:
            ab_l_node_feat = ab_l_node_feat_i
        else:
            ab_l_node_feat = torch.cat([ab_l_node_feat, ab_l_node_feat_i], dim=0)

        ab_l_edge_len.append(ab_l_X_i.shape[0]) 
        ab_l_edge_index_i = copy.deepcopy(geo_data['ab_l_edge_index'][name[i]])
        if i == 0:
            ab_l_edge_index = ab_l_edge_index_i
        else:
            gap = 0
            for j in range(i):
                gap += ab_l_edge_len[j]
            ab_l_edge_index_i[0]  += gap
            ab_l_edge_index_i[1]  += gap
            ab_l_edge_index = torch.cat([ab_l_edge_index, ab_l_edge_index_i], dim=-1)
        
        ag_X_i = geo_data['ag_X'][name[i]]
        if ag_X == None:
            ag_X = ag_X_i
        else:
            ag_X = torch.cat([ag_X, ag_X_i], dim=0)
        
        ag_batch_id.extend([i]*ag_X_i.shape[0])
        
        ag_seq_i = geo_data['ag_seq'][name[i]]
        if ag_seq == None:
            ag_seq = ag_seq_i
        else:
            ag_seq = torch.cat([ag_seq, ag_seq_i])
        
        ag_node_feat_i = geo_data['ag_node_feat'][name[i]]
        if ag_node_feat == None:
            ag_node_feat = ag_node_feat_i
        else:
            ag_node_feat = torch.cat([ag_node_feat, ag_node_feat_i], dim=0)

        ag_edge_len.append(ag_X_i.shape[0])
        ag_edge_index_i = copy.deepcopy(geo_data['ag_edge_index'][name[i]])
        if i == 0:
            ag_edge_index = ag_edge_index_i
        else:
            gap = 0
            for j in range(i):
                gap += ag_edge_len[j]
            ag_edge_index_i[0]  += gap
            ag_edge_index_i[1]  += gap
            ag_edge_index = torch.cat([ag_edge_index, ag_edge_index_i], dim=-1) 
    
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
    data_dict['ab_h_batch_id'] = torch.tensor(ab_h_batch_id)
    data_dict['ab_l_batch_id'] = torch.tensor(ab_l_batch_id)
    data_dict['ag_batch_id'] = torch.tensor(ag_batch_id)
    
    return data_dict

def max_atom_seq(data):
    """
    Obtain the maximum of atomic count and sequence length.
    """
    atom_len = 0
    seq_len = 0
    for key in data.keys():
        tem_atom = data[key]['atom_feats'].shape[1]
        tem_seq = data[key]['molecule_atom_lens'].shape[-1]
        if atom_len<tem_atom:
            atom_len = tem_atom
        if seq_len<tem_seq:
            seq_len = tem_seq
    return atom_len, seq_len

def batch_data(name, data, device):
    data_batch = {}
    data_batch_ver1 = {}
    for ele in name:
        data_batch[ele] = data[ele]
    atom_len, seq_len = max_atom_seq(data_batch)
    for key in data_batch.keys():
        data_batch[key]['atom_feats'] = torch.nn.functional.pad(data_batch[key]['atom_feats'], (0,0,0,atom_len-data_batch[key]['atom_feats'].shape[1]))
        data_batch[key]['atom_ref_pos'] = torch.nn.functional.pad(data_batch[key]['atom_ref_pos'], (0,0,0,atom_len-data_batch[key]['atom_ref_pos'].shape[1]))
        data_batch[key]['atom_ref_space_uid'] = torch.nn.functional.pad(data_batch[key]['atom_ref_space_uid'], (0,atom_len-data_batch[key]['atom_ref_space_uid'].shape[1]))
        data_batch[key]['molecule_atom_lens'] = torch.nn.functional.pad(data_batch[key]['molecule_atom_lens'], (0,seq_len-data_batch[key]['molecule_atom_lens'].shape[1]))
    
    for key2 in data_batch.keys():
        if 'atom_feats' not in data_batch_ver1.keys():
            data_batch_ver1['atom_feats'] = data_batch[key2]['atom_feats'].to(device)
        else:
            data_batch_ver1['atom_feats'] = torch.cat([data_batch_ver1['atom_feats'], data_batch[key2]['atom_feats'].to(device)])
        if 'atom_ref_pos' not in data_batch_ver1.keys():
            data_batch_ver1['atom_ref_pos'] = data_batch[key2]['atom_ref_pos'].to(device)
        else:
            data_batch_ver1['atom_ref_pos'] = torch.cat([data_batch_ver1['atom_ref_pos'], data_batch[key2]['atom_ref_pos'].to(device)])
        if 'atom_ref_space_uid' not in data_batch_ver1.keys():
            data_batch_ver1['atom_ref_space_uid'] = data_batch[key2]['atom_ref_space_uid'].to(device)
        else:
            data_batch_ver1['atom_ref_space_uid'] = torch.cat([data_batch_ver1['atom_ref_space_uid'], data_batch[key2]['atom_ref_space_uid'].to(device)])
        if 'molecule_atom_lens' not in data_batch_ver1.keys():
            data_batch_ver1['molecule_atom_lens'] = data_batch[key2]['molecule_atom_lens'].to(device)
        else:
            data_batch_ver1['molecule_atom_lens'] = torch.cat([data_batch_ver1['molecule_atom_lens'], data_batch[key2]['molecule_atom_lens'].to(device)])
        
    return data_batch_ver1