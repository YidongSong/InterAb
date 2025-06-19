import shutil

import torch
from box import Box

def load_pretrain(model, pretrain_file, is_GPU = False, is_fair = False,
                  is_origin = False,
                  is_exit = False,
                  fill = None,
                  skip_keys = None,
                  from_xfold_to_xfold2d = False):

    print('    loading', pretrain_file)

    if is_GPU:
        pretrain_state_dict = torch.load(pretrain_file)
    else:
        pretrain_state_dict = torch.load(pretrain_file, map_location='cpu')

    if is_fair:
        pretrain_state_dict = pretrain_state_dict['model']

    elif is_origin:
        pretrain_state_dict = pretrain_state_dict['model_state_dict']

    state_dict = model.state_dict()

    for key in state_dict.keys():
        skip = False
        if isinstance(skip_keys, list):
            for sk in skip_keys:
                if sk in key:
                    skip = True
        if skip:
            print(f'skip loading: {key}')
            continue

        # TODO Dirty code for grow depth
        if fill is not None and 'feat_extractor.iter_block_1' in key:
            n = fill[0]
            cur_idx = int(key.split('.')[3]) if 'xfold2d.' in key else int(key.split('.')[2])
            new_idx = cur_idx % n

            if cur_idx == new_idx:
                new_key = key
            else:
                if 'xfold2d.' in key:
                    new_key = f'xfold2d.feat_extractor.iter_block_1.{new_idx}' + key[29 + 8:]
                else:
                    new_key = f'feat_extractor.iter_block_1.{new_idx}' + key[29:]

                print(f'replace {key} to {new_key}')
        else:
            new_key = key

        if from_xfold_to_xfold2d:
            new_key = 'xfold2d.' + new_key

        if new_key in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict[new_key]

        elif ('encoder.xfold.' + new_key) in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict['encoder.xfold.' + new_key]

        elif ('encoder.model.' + new_key) in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict['encoder.model.' + new_key]

        elif ('model.' + new_key) in pretrain_state_dict.keys():
            state_dict[key] = pretrain_state_dict['model.' + new_key]

        else:
            print(f'{new_key} not in pretrain_state_dicts')
            if is_exit or fill is not None:
                raise NotImplementedError

    model.load_state_dict(state_dict)

    return model

def convert_fair_pt(fair_pt, save_pt):
    data = torch.load(fair_pt)
    torch.save(data['model'], save_pt)
    print(f'convert {fair_pt} to {save_pt}')

def convert_model_yaml(model_yaml, merged_yaml):
    params = Box.from_yaml(filename=model_yaml)
    if 'msa_bert_config' in params and params['msa_bert_config'] is not None:
        print('xxxx')
        msa_bert = Box.from_yaml(filename=params['msa_bert_config']['model_yaml'])
        params['msa_bert'] = msa_bert

    params.to_yaml(merged_yaml)
    print(f'convert {model_yaml} to {merged_yaml}')

def convert_example(root, save_root):
    fair_pt = f'{root}/checkpoint_best.pt'
    save_pt = f'{save_root}/checkpoint_slim.pt'
    if not os.path.exists(save_pt):
        convert_fair_pt(fair_pt, save_pt)
