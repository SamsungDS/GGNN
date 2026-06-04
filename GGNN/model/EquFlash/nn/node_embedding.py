from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional

from e3nn.o3 import Irreps, Linear


class OneHotNodeEmbedding(nn.Module):
    def __init__(
        self,
        num_species: int,
        irreps_out: Irreps,
        use_bias: bool,
        type_map: Dict[int, int],
    ):
        super().__init__()
        self.num_species = num_species
        self.type_map = self._make_z_to_index(type_map)  # Long[120]
        self.input_linear = Linear(
            Irreps(f"{num_species}x0e"), irreps_out, biases=use_bias
        )

    @staticmethod
    def _make_z_to_index(type_map):
        z2i = torch.full((120,), -1, dtype=torch.long)
        for z, idx in type_map.items():
            z2i[z] = idx
        return z2i

    def forward(self, batch):
        if "atomic_numbers" in batch.keys():
            atomic_numbers = batch["atomic_numbers"].long()
            mapper = self.type_map.to(atomic_numbers.device)[atomic_numbers]
        else:
            mapper = batch["x"]
        x = torch.nn.functional.one_hot(mapper, self.num_species).float()
        x = self.input_linear(x)

        batch["x"] = x
        batch["atom_mapper"] = mapper
        return batch
