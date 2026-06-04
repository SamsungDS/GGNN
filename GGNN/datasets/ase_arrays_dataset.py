"""
This file is derived from fairchem-core.
Original source: https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License. See LICENSE for details.

Modifications copyright (c) 2026 Samsung Electronics.
Licensed under CC BY-NC-SA 4.0. See LICENSE for details.
"""
from __future__ import annotations

from copy import deepcopy
import warnings
from typing import Any, Callable
from pathlib import Path
from tqdm import tqdm
import ase

from fairchem.core.common.registry import registry
from fairchem.core.datasets.ase_datasets import AseReadMultiStructureDataset, apply_one_tags

from GGNN.preprocessing.atoms_arrays_to_graphs import AtomsArraysToGraphs



@registry.register_dataset("ase_arrays_read_multi")
class AseArraysReadMultiStructureDataset(AseReadMultiStructureDataset):
    
    # Oct 1, 2025
    # This class is a cousin of fairchem.core.datasets.ase_datasets.AseReadMultiStructureDataset
    # with functionalities very similar to the fairchem (tested v1.10.0) counterpart,
    # except that we override 'self.a2g' with custom 'AtomsArraysToGraphs'
    # which allows one to translate ase.Atoms.arrays to graph data as well.

    # REFERENCE:
    # https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.10.0/src/fairchem/core/datasets/ase_datasets.py#L284
    
    def __init__(
        self,
        config: dict,
        atoms_transform: Callable[[ase.Atoms, Any, ...], ase.Atoms] = apply_one_tags
    ) -> None:

        # Pop the custom 'r_array_keys' from config before passing onto super().__init__()
        mod_config = deepcopy(config)
        a2g_args = mod_config.get("a2g_args", {}) or {}
        self.r_array_keys = a2g_args.pop("r_array_keys", [])

        super().__init__(config=mod_config, atoms_transform=atoms_transform)

        # Replace self.a2g with custom class that can handle atoms.arrays
        if "r_edges" not in a2g_args:
            a2g_args["r_edges"] = False
        a2g_args["r_pbc"] = True
        self.a2g = AtomsArraysToGraphs(**a2g_args, r_array_keys=self.r_array_keys)
