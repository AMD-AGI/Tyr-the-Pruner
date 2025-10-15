# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import random
import dataclasses
from typing import Any, Sequence

import numpy as np
import torch


def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def none_or_str(value):
    if value == "None":
        return None
    return value

def calculate_tensor_memory(tensor):
    return tensor.element_size() * tensor.numel() / (1024 ** 3)

def to(data: Any, *args, **kwargs):
    def _to(x):
        return to(x, *args, **kwargs)

    if isinstance(data, torch.Tensor):
        return data.to(*args, **kwargs)
    elif isinstance(data, (tuple, list, set)):
        return type(data)(_to(x) for x in data)
    elif isinstance(data, dict):
        return type(data)((k, _to(v)) for k, v in data.items())
    elif dataclasses.is_dataclass(data):
        return type(data)(**{k: _to(v) for k, v in vars(data).items()})
    else:
        return data


def maybe_first_element(x):
    if isinstance(x, Sequence):
        x = x[0]
    return x
