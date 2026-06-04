"""
This file is derived from fairchem-core.
Original source: https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License. See LICENSE for details.

Modifications copyright (c) 2026 Samsung Electronics.
Licensed under CC BY-NC-SA 4.0. See LICENSE for details.
"""

import copy
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch_scatter import scatter, segment_coo, segment_csr
import numpy as np


@contextmanager
def new_trainer_context(*, config: dict[str, Any], distributed: bool = False):
    from fairchem.core.common import distutils, gp_utils
    from fairchem.core.common.registry import registry
    from fairchem.core.common.utils import (
        setup_env_vars,
        setup_imports,
        setup_logging,
    )  
    if True:  # TYPE_CHECKING: #to be fixed
        from fairchem.core.tasks.task import BaseTask
        from fairchem.core.trainers import (
            BaseTrainer,
        )  # To trainer.base_trainer. to be fixed

    @dataclass
    class _TrainingContext:
        config: dict[str, Any]
        task: BaseTask
        trainer: BaseTrainer

    setup_logging()
    setup_env_vars()
    original_config = config
    config = copy.deepcopy(original_config)

    distutils.setup(config)
    if config["gp_gpus"] is not None:
        gp_utils.setup_gp(config)

    setup_imports(config)
    trainer_name = config.get("trainer", "default")
    # backwards compatibility for older configs
    if trainer_name in ["forces", "equiformerv2_forces"]:
        task_name = "s2ef"
    elif trainer_name in ["energy", "equiformerv2_energy"]:
        task_name = "is2re"
    elif "multitask" in trainer_name:
        task_name = "multitask"
    else:
        task_name = "ocp"


    from ..trainer.trainer import Trainer  # added

    if config["trainer"] == "default":
        trainer_cls = Trainer  
    else:
        trainer_cls = registry.get_trainer_class(trainer_name)

    trainer_config = {
        "model": config["model"],
        "optimizer": config["optim"],
        "identifier": config["identifier"],
        "timestamp_id": config.get("timestamp_id", None),
        "run_dir": config.get("run_dir", "./"),
        "is_debug": config.get("is_debug", False),
        "print_every": config.get("print_every", 10),
        "seed": config.get("seed", 0),
        "logger": config.get("logger", "files"),
        "local_rank": config["local_rank"],
        "amp": config.get("amp", False),
        "cpu": config.get("cpu", False),
        "slurm": config.get("slurm", {}),
        "name": task_name,
        "gp_gpus": config.get("gp_gpus"),
    }
    if "val_dataset" in config and "src" in config["dataset"]:
        config["dataset"] = {"train": config["dataset"], "val": config["val_dataset"]}
    if task_name == "multitask":
        trainer_config.update(
            {
                "tasks": config.get("tasks", {}),
                "dataset_configs": config["datasets"],
                "combined_dataset_config": config.get("combined_dataset", {}),
                "evaluations": config.get("evaluations", {}),
            }
        )
    else:
        trainer_config.update(
            {
                "task": config.get("task", {}),
                "outputs": config.get("outputs", {}),
                "dataset": config["dataset"],
                "loss_functions": config.get("loss_functions", {}),
                "evaluation_metrics": config.get("evaluation_metrics", {}),
            }
        )

    trainer = trainer_cls(**trainer_config)
    task_cls = registry.get_task_class(config["mode"])
    assert task_cls is not None, "Task not found"
    task = task_cls(config)

    start_time = time.time()
    ctx = _TrainingContext(config=original_config, task=task, trainer=trainer)
    yield ctx
    distutils.synchronize()
    distutils.cleanup()


from torch_geometric.nn import radius_graph
from torch_geometric.data import Batch

from fairchem.core.common.utils import (
    compute_neighbors,
    get_pbc_distances,
)


# to be relocated.
def add_edge_for_data(data, cutoff, max_neighbors, use_pbc, otf_graph):
    d_list = []
    d_list.append(data)
    data_single_batch = Batch.from_data_list(d_list)
    (edge_index, _, _, cell_offsets, _, _) = generate_graph(
        data_single_batch, cutoff, max_neighbors, use_pbc, otf_graph
    )
    data.__setitem__("edge_index", edge_index)
    data.__setitem__("cell_offsets", cell_offsets)
    return data


def radius_graph_pbc(
    data,
    radius,
    max_num_neighbors_threshold,
    enforce_max_neighbors_strictly: bool = False,
    pbc=None,
):
    if pbc is None:
        pbc = [True, True, True]
    device = data.pos.device
    batch_size = len(data.natoms)

    if hasattr(data, "pbc"):
        data.pbc = torch.atleast_2d(data.pbc)
        for i in range(3):
            if not torch.any(data.pbc[:, i]).item():
                pbc[i] = False
            elif torch.all(data.pbc[:, i]).item():
                pbc[i] = True
            else:
                raise RuntimeError(
                    "Different structures in the batch have different PBC configurations. This is not currently supported."
                )

    # position of the atoms
    atom_pos = data.pos

    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = data.natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image

    index_offset_expand = torch.repeat_interleave(index_offset, num_atoms_per_image_sqr)
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = torch.arange(num_atom_pairs, device=device) - index_sqr_offset

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor")
    ) + index_offset_expand
    index2 = (atom_count_sqr % num_atoms_per_image_expand) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).

    cross_a2a3 = torch.cross(data.cell[:, 1], data.cell[:, 2], dim=-1)
    cell_vol = torch.sum(data.cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)

    if pbc[0]:
        inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
        rep_a1 = torch.ceil(radius * inv_min_dist_a1)
    else:
        rep_a1 = data.cell.new_zeros(1)

    if pbc[1]:
        cross_a3a1 = torch.cross(data.cell[:, 2], data.cell[:, 0], dim=-1)
        inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
        rep_a2 = torch.ceil(radius * inv_min_dist_a2)
    else:
        rep_a2 = data.cell.new_zeros(1)

    if pbc[2]:
        cross_a1a2 = torch.cross(data.cell[:, 0], data.cell[:, 1], dim=-1)
        inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
        rep_a3 = torch.ceil(radius * inv_min_dist_a3)
    else:
        rep_a3 = data.cell.new_zeros(1)

    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = [rep_a1.max(), rep_a2.max(), rep_a3.max()]

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep.item(), rep.item() + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    unit_cell = torch.cartesian_prod(*cells_per_dim)
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(len(index2), 1, 1)
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(batch_size, -1, -1)

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(data.cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

    # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    mask_within_radius = torch.le(atom_distance_sqr, radius * radius)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = unit_cell_per_atom.view(-1, 3)[mask.view(-1)]
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)

    mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
        natoms=data.natoms,
        index=index1,
        atom_distance=atom_distance_sqr,
        max_num_neighbors_threshold=max_num_neighbors_threshold,
        enforce_max_strictly=enforce_max_neighbors_strictly,
    )

    if not torch.all(mask_num_neighbors):
        # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
        index1 = torch.masked_select(index1, mask_num_neighbors)
        index2 = torch.masked_select(index2, mask_num_neighbors)
        unit_cell = torch.masked_select(
            unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
        )
        unit_cell = unit_cell.view(-1, 3)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image


def get_max_neighbors_mask(
    natoms,
    index,
    atom_distance,
    max_num_neighbors_threshold,
    degeneracy_tolerance: float = 0.01,
    enforce_max_strictly: bool = False,
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.

    Enforcing the max strictly can force the arbitrary choice between
    degenerate edges. This can lead to undesired behaviors; for
    example, bulk formation energies which are not invariant to
    unit cell choice.

    A degeneracy tolerance can help prevent sudden changes in edge
    existence from small changes in atom position, for example,
    rounding errors, slab relaxation, temperature, etc.
    """

    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(max=max_num_neighbors_threshold)

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(natoms.shape[0] + 1, device=device, dtype=torch.long)
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor([True], dtype=bool, device=device).expand_as(
            index
        )
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full([num_atoms * max_num_neighbors], np.inf, device=device)

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)

    # Select the max_num_neighbors_threshold neighbors that are closest
    if enforce_max_strictly:
        distance_sort = distance_sort[:, :max_num_neighbors_threshold]
        index_sort = index_sort[:, :max_num_neighbors_threshold]
        max_num_included = max_num_neighbors_threshold

    else:
        effective_cutoff = (
            distance_sort[:, max_num_neighbors_threshold] + degeneracy_tolerance
        )
        is_included = torch.le(distance_sort.T, effective_cutoff)

        # Set all undesired edges to infinite length to be removed later
        distance_sort[~is_included.T] = np.inf

        # Subselect tensors for efficiency
        num_included_per_atom = torch.sum(is_included, dim=0)
        max_num_included = torch.max(num_included_per_atom)
        distance_sort = distance_sort[:, :max_num_included]
        index_sort = index_sort[:, :max_num_included]

        # Recompute the number of neighbors
        num_neighbors_thresholded = num_neighbors.clamp(max=num_included_per_atom)

        num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # Offset index_sort so that it indexes into index
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_included
    )
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
    index_sort = torch.masked_select(index_sort, mask_finite)

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image


def generate_graph(
    data,
    cutoff=None,
    max_neighbors=None,
    use_pbc=None,
    otf_graph=None,
):
    if not otf_graph:
        try:
            edge_index = data.edge_index

            if use_pbc:
                cell_offsets = data.cell_offsets
                neighbors = data.neighbors

        except AttributeError:
            # logging.warning(
            #     "Turning otf_graph=True as required attributes not present in data object"
            # )
            otf_graph = True

    if use_pbc:
        if otf_graph:
            edge_index, cell_offsets, neighbors = radius_graph_pbc(
                data, cutoff, max_neighbors
            )

        out = get_pbc_distances(
            data.pos,
            edge_index,
            data.cell,
            cell_offsets,
            neighbors,
            return_offsets=True,
            return_distance_vec=True,
        )
        edge_index = out["edge_index"]
        edge_dist = out["distances"]
        cell_offset_distances = out["offsets"]
        distance_vec = out["distance_vec"]
    else:
        if otf_graph:
            edge_index = radius_graph(
                data.pos,
                r=cutoff,
                batch=data.batch,
                max_num_neighbors=max_neighbors,
            )

        j, i = edge_index
        distance_vec = data.pos[j] - data.pos[i]

        edge_dist = distance_vec.norm(dim=-1)
        cell_offsets = torch.zeros(edge_index.shape[1], 3, device=data.pos.device)
        cell_offset_distances = torch.zeros_like(cell_offsets, device=data.pos.device)
        neighbors = compute_neighbors(data, edge_index)

    return (
        edge_index,
        edge_dist,
        -distance_vec,  # align sign of distance vec btw lammps.
        cell_offsets,
        cell_offset_distances,
        neighbors,
    )

try :
    from nvalchemiops.torch.neighbors import neighbor_list


    def generate_graph_nvalchemi(
        data,
        cutoff=None,
    ):
    
        edge_index, edge_ptr, cell_offsets = neighbor_list(
            data.pos,
            cutoff,
            data.cell,
            pbc=data.pbc.reshape(-1, 3),
            batch_idx=data.batch.to(torch.int32),
            return_neighbor_list=True,
            max_neighbors=500,
        )
        neigh = torch.scatter_add(
            torch.zeros(data.batch.max() + 1, dtype=torch.int32, device=data.batch.device),
            0,
            data.batch,
            edge_ptr[1:] - edge_ptr[:-1],
        )
        distance_vec = data.pos[edge_index[1, :]] - data.pos[edge_index[0, :]]
        cell = torch.repeat_interleave(data.cell, neigh, dim=0)
        offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
        distance_vec += offsets
        return (
            edge_index,
            distance_vec,  # align sign of distance vec btw lammps.
            cell_offsets,
            neigh,
        )
except ImportError:
    pass
    

