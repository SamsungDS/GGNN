from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode





@compile_mode("script")
class SpeciesWiseRescale(nn.Module):
    """
    Scaling and shifting energy (and automatically force and stress)
    Use as it is if given list, expand to list if one of them is float
    If two lists are given and length is not the same, raise error
    """

    def __init__(
        self,
        shift: Union[List[float], float],
        scale: Union[List[float], float],
        type_map,
        train_shift: bool = False,
        train_scale: bool = False,
    ):
        super().__init__()
        num_species = len(type_map)
        shift = [shift] * num_species if isinstance(shift, float) else [shift[z] for z in sorted(type_map, key=lambda x: type_map[x])]
        scale = [scale] * num_species if isinstance(scale, float) else [scale[z] for z in sorted(type_map, key=lambda x: type_map[x])]

        self.shift = nn.Parameter(torch.FloatTensor(shift), requires_grad=train_shift)
        self.scale = nn.Parameter(torch.FloatTensor(scale), requires_grad=train_scale)

    def forward(self, x, idx):
        x = x * self.scale[idx].view(-1, 1) + self.shift[idx].view(-1, 1)
        return x
