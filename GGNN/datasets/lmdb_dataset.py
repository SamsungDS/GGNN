"""
This file is derived from fairchem-core.
Original source: https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License. See LICENSE for details.

Modifications copyright (c) 2026 Samsung Electronics.
Licensed under CC BY-NC-SA 4.0. See LICENSE for details.
"""

import bisect
import logging
import pickle
import warnings
from pathlib import Path
from typing import TypeVar

import lmdb
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch
from torch_geometric.data.data import BaseData

from GGNN.common.utils import generate_graph

from typing import TYPE_CHECKING, TypeVar, List

if TYPE_CHECKING:
    from pathlib import Path
    from torch_geometric.data.data import BaseData


def data_list_collater(
    data_list: list[BaseData],
    cutoff: float,
    max_neighbors: int,
    use_pbc: bool,
    otf_graph: bool = False,
    exclude_keys: List[str] = [],
) -> BaseData:
    for data in data_list:
        data.natoms = torch.tensor([data.natoms])
        edge_index, a1, distance_vec, cell_offsets, cell_offsets_distance_vec, a2 = (
            generate_graph(
                data,
                cutoff,
                max_neighbors,
                use_pbc,  ## use_pbc
                otf_graph,
            )
        )
        data.edge_index = edge_index
        data.cell_offsets = cell_offsets
        data.edge_vec = distance_vec

    batch = Batch.from_data_list(data_list, exclude_keys=exclude_keys)

    if not otf_graph:
        try:
            n_neighbors = []
            for _, data in enumerate(data_list):
                n_index = data.edge_index[1, :]
                n_neighbors.append(n_index.shape[0])
            batch.neighbors = torch.tensor(n_neighbors)
        except (NotImplementedError, TypeError):
            logging.warning(
                "LMDB does not contain edge index information, set otf_graph=True"
            )

    return batch
