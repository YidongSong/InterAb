from .utils import *
from .pretrained import *
from .gen_fea import *
from .gen_lbl import *
from .calculate_cnt import *
from .calculate_gdt import *

def get_xfold_model(model, model_config):

    if model == 'XFold2D':
        from tfold.modules.xfold.xfold_2d import XFold2D
        net = XFold2D(**model_config)
    else:
        raise RuntimeException('Model not exists {}'.format(model))

    return net
