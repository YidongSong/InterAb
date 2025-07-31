#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import collections
import torch
import numpy as np
import transformers
import pickle
import copy
from os.path import join
import DataLoader.bert_finetuning_er_seq2seq_dataset as module_data
import loss as module_loss
import metric as module_metric
import interaction as module_arch
from parse_config import ConfigParser
import pdb
import pandas as pd
import sys


def predict(model ,data_loader, antibody_tokenizer, antigen_tokenizer,device,config, task):
    if task == 'affinity':
        atom_inputs = pickle.load(open('./data/atom/exa_affinity_atom_inputs.pkl', 'rb'))
        geo_data = pickle.load(open('./data/Geo_data/exa_affinity_geo_data.pkl', 'rb'))
    if task == 'specificity':
        atom_inputs = pickle.load(open('./data/atom/exa_specificity_atom_inputs.pkl', 'rb'))
        geo_data = pickle.load(open('./data/Geo_data/exa_specificity_geo_data.pkl', 'rb'))

    model.eval()
    result_dict = {'heavy': [], 
                    'light':[], 
                    'antigen':[],
                    'y_pred': []}
    with torch.no_grad():
        for batch_idx, (protein_name, antibody_a_tokenized,antibody_b_tokenized, receptor_tokenized) in enumerate(data_loader):
            antibody_a_tokenized = {k: v.to(device) for k, v in antibody_a_tokenized.items()}
            antibody_b_tokenized = {k: v.to(device) for k, v in antibody_b_tokenized.items()}
            receptor_tokenized = {k: v.to(device) for k, v in receptor_tokenized.items()}
            
            data_dict = module_data.Geo_data(geo_data, protein_name)
            data_batch = module_data.batch_data(protein_name, atom_inputs, device)
            # output = model(protein_name, antibody_a_tokenized, antibody_b_tokenized, receptor_tokenized, data_dict, data_batch, device)
            output = model(antibody_a_tokenized, antibody_b_tokenized, receptor_tokenized, data_dict, data_batch, device)
            y_pred = output
            
            if task == 'specificity':
                y_pred = torch.sigmoid(output)
            
            y_pred = y_pred.cpu().detach().numpy()
            result_dict['y_pred'].append(y_pred)

            antibody = antibody_tokenizer.batch_decode(antibody_a_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            antibody = [s.replace(" ", "") for s in antibody]

            light = antibody_tokenizer.batch_decode(antibody_b_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            light = [s.replace(" ", "") for s in light]


            receptor = antigen_tokenizer.batch_decode(receptor_tokenized['input_ids'],
                                                    skip_special_tokens=True)
            receptor = [s.replace(" ", "") for s in receptor]
            result_dict['heavy'].append(antibody)
            result_dict['light'].append(light)
            result_dict['antigen'].append(receptor)


    y_pred = np.concatenate(result_dict['y_pred'])



    test_df = pd.DataFrame({'heavy': [v for l in result_dict['heavy'] for v in l],
                            'light': [v for l in result_dict['light'] for v in l],
                            'antigen': [v for l in result_dict['antigen'] for v in l],
                            'y_pred': list(y_pred.flatten())})
    test_df.to_csv(join(config.log_dir, 'test_result.csv'), index=False)

    return test_df

def main(config, task):
    logger = config.get_logger('eval_generation')

    # fix random seeds for reproducibility
    seed = config['data_loader']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # setup data_loader instances
    config['data_loader']['args']['logger'] = logger
    data_loader = config.init_obj('data_loader', module_data)

    antibody_tokenizer = data_loader.get_antibody_tokenizer()
    antigen_tokenizer = data_loader.get_antigen_tokenizer()

    # build model architecture, then print to console

    model = config.init_obj('arch', module_arch)
    logger.info('Loading checkpint from {}'.format(
        config['discriminator_resume']))
    
    checkpoint = torch.load(config['discriminator_resume'])
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to("cuda")

    """Test."""
    logger = config.get_logger('test')
    predict(model=model, data_loader=data_loader, antibody_tokenizer=antibody_tokenizer, antigen_tokenizer=antigen_tokenizer, device="cuda", config=config, task=task)


if __name__ == '__main__':
        args = argparse.ArgumentParser(description='PyTorch Template')
        args.add_argument('-c', '--config', default='', type=str,
                        help='config file path (default: None)')
        args.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
        args.add_argument('-d', '--device', default=None, type=str,
                        help='indices of GPUs to enable (default: all)')
        args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                        help='local rank for nGPUs training')
        args.add_argument('--task', default='affinity', type=str,
                        help='the task: affinity or specificity')
        # custom cli options to modify configuration from default values given in json file.
        CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
        options = [
            CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
            CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
        ]
        config = ConfigParser.from_args(args, options)
        args = args.parse_args()
        main(config, args.task)
