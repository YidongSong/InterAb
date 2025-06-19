"""Import all the modules."""

from tfold.modules.gfp.cp_net import CPNetV1
from tfold.modules.gfp.cp_net import CPNetV2
from tfold.modules.gfp.cp_net import CPNetV3
from tfold.modules.gfp.gfp import GFP
from tfold.modules.gfp.gfp_net import GFPNet


__all__ = [
    'CPNetV1',
    'CPNetV2',
    'CPNetV3',
    'GFP',
    'GFPNet',
]
