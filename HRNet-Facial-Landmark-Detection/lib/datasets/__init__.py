# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .aflw import AFLW
from .cofw import COFW
from .face300w import Face300W
from .wflw import WFLW
from .NTA import NTA
from .NTA_LIPS import NTA_LIPS

__all__ = ['AFLW', 'COFW', 'Face300W', 'WFLW', 'NTA', 'NTA_LIPS' 'get_dataset']


def get_dataset(config):

    if config.DATASET.DATASET == 'AFLW':
        return AFLW
    elif config.DATASET.DATASET == 'COFW':
        return COFW
    elif config.DATASET.DATASET == '300W':
        return Face300W
    elif config.DATASET.DATASET == 'WFLW':
        return WFLW
    elif config.DATASET.DATASET == 'NTA':
        return NTA
    elif config.DATASET.DATASET == 'NTA_LIPS':
        return NTA_LIPS
    else:
        raise NotImplemented()

