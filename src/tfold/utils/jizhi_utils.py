"""Jizhi-related utility functions."""

import os
import json
import socket
import logging
import hashlib
import traceback
import yaml

import requests


def get_ip():
    """Get the current node's IP.

    Args: n/a

    Returns:
    * ip_addr: current node's IP
    """

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(('10.223.30.51', 53))
        ip_addr = sock.getsockname()[0]
    except Exception as err:  # pylint: disable=broad-except
        ip_addr = socket.gethostbyname(socket.gethostname())
        logging.warning('failed to get the current node\'s IP: %s', err)
    finally:
        sock.close()

    return ip_addr


def report_progress(msg):
    """Report the training progress message.

    Args:
    * msg: (key, value) pairs of critical indexes for the training progress

    Returns:
    * flag: whether the training progress message has been successfully reported
    * text: server's response text
    """

    ip_addr = os.environ.get('CHIEF_IP', '')
    if not ip_addr:
        ip_addr = get_ip()
    url = f'http://{ip_addr}:8080/v1/worker/report-progress'
    err_frmt = 'send progress info to worker failed!\nprogress_info: %s, \n%s: %s'

    try:
        response = requests.post(url, json=json.dumps(msg), proxies={"http": None, "https": None})
    except Exception as err:  # pylint: disable=broad-except
        logging.warning(err_frmt, msg, 'traceback', traceback.format_exc())
        return False, str(err)

    if response.status_code != 200:
        logging.warning(err_frmt, msg, 'reason', response.reason)
        return False, response.text

    return True, ''


def report_error(code, msg=''):
    """Report the error message.

    Args:
    * code: error code
    * msg: error message

    Returns:
    * flag: whether the error message has been successfully reported
    * text: server's response text
    """

    err_msg = {'type': 'error', 'code': code, 'msg': msg}

    return report_progress(err_msg)


def report_completion():
    """Report the task completion message.

    Args: n/a

    Returns:
    * flag: whether the task completion message has been successfully reported
    * text: server's response text
    """

    msg = {'type': 'completed'}

    return report_progress(msg)


def get_mdl_dpath(task_flag, user):
    """Get the model directory path.

    Args:
    * task_flag: TJ task flag
    * user: user name

    Returns:
    * mdl_dpath: model directory path
    """

    # configurations
    mdl_dpath_root_list = [
        f'/apdcephfs/share_1364275/{user}/JiZhi.Models',  # DrugAI_CQ
        f'/apdcephfs_cq3/share_2934111/{user}/JiZhi.Models',  # AILab_MLC_CQ
        f'/apdcephfs/share_1436367/{user}/JiZhi.Models',  # DrugAI_QY
        f'/apdcephfs/share_1594716/{user}/JiZhi.Models',  # AILab_MLC_QY
        f'/mnt/ai4x_ceph/{user}/private/JiZhi.Models',
    ]

    # get the model directory path
    has_found = False
    md5sum = hashlib.md5(task_flag.encode('utf-8')).hexdigest()
    for mdl_dpath_root in mdl_dpath_root_list:
        mdl_dpath = os.path.join(mdl_dpath_root, md5sum)
        if os.path.exists(mdl_dpath):
            has_found = True
            break
    # assert has_found, f'failed to find the model directory path: {task_flag}'
    if not has_found:
        mdl_dpath = None

    return mdl_dpath


def get_ceph_dir(business_flag, user):
    """Get the root directory of ceph disk based on the business flag.

    Args:
    * business_flag: TJ business flag
    * user: user name

    Returns:
    * ceph_dir: root directory of ceph disk
    """

    if business_flag == 'DrugAI_CQ':
        ceph_dir = f'/apdcephfs/share_1364275/{user}'
    elif business_flag == 'AILab_MLC_CQ':
        ceph_dir = f'/apdcephfs_cq3/share_2934111/{user}'
    elif business_flag == 'DrugAI_QY':
        ceph_dir = f'/apdcephfs/share_1436367/{user}'
    elif business_flag == 'AILab_MLC_QY':
        ceph_dir = f'/apdcephfs_qy3/share_1594716/{user}'
    elif business_flag == 'AILab_AI4S_kemo_QY':
        ceph_dir = f'/apdcephfs_qy3/share_301997302/{user}'
    else:
        raise ValueError(f'unrecognized business flag: {business_flag}')

    return ceph_dir


def update_taiji_config(config):
    """Update TaiJi configurations w/ personal configurations injected.

    Args:
    * config: dict of TaiJi configurations

    Returns:
    * config: dict of TaiJi configurations w/ personal configurations injected
    """

    # parse the personal configuration file
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    yml_fpath = os.path.join(curr_dir, '../../config.yaml')
    assert os.path.exists(yml_fpath), f'personal configuration file not found: {yml_fpath}'
    with open(yml_fpath, 'r', encoding='UTF-8') as i_file:
        config_addi = yaml.safe_load(i_file)

    # update TaiJi configurations
    config['User'] = config_addi['TAIJI_USER']
    config['Token'] = config_addi['TAIJI_TOKEN']

    if 'dataset_params' in config:
        datasets = config['dataset_params']['dataset_name'].split(',')

        for num, dataset in enumerate(datasets):
            if dataset == 'superCeph2':
                config['dataset_params']['ceph_info'][num]['addr'] = config_addi['SUPERCEPH2_ADDR']
                config['dataset_params']['ceph_info'][num]['secret'] = config_addi['SUPERCEPH2_SECRET']
            if dataset == 'ai4x_ceph_jonathanwu':
                config['dataset_params']['ceph_info'][num]['addr'] = config_addi['AI4X_CEPH_ADDR']
                config['dataset_params']['ceph_info'][num]['secret'] = config_addi['AI4X_CEPH_SECRET_ROOT_JONATHANWU']
            if dataset == 'ai4x_ceph_fandiwu':
                config['dataset_params']['ceph_info'][num]['addr'] = config_addi['AI4X_CEPH_ADDR']
                config['dataset_params']['ceph_info'][num]['secret'] = config_addi['AI4X_CEPH_SECRET_ROOT_FANDIWU']
            if dataset == 'ai4x_ceph_jianhuayao':
                config['dataset_params']['ceph_info'][num]['addr'] = config_addi['AI4X_CEPH_ADDR']
                config['dataset_params']['ceph_info'][num]['secret'] = config_addi['AI4X_CEPH_SECRET_ROOT_JIANHUAYAO']

    if 'model_params' in config and config['model_params']['model_source'] == 'git':
        config['model_params']['path_info']['private_token'] = config_addi['GIT_TOKEN']

    return config
