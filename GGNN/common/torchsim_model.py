"""Wrapper for GGNN model in TorchSim.

This module provides a TorchSim wrapper of the GGNN model for computing
energies, forces, and stresses for atomistic systems. It integrates the GGNN model
with TorchSim's simulation framework, handling batched computations for multiple
systems simultaneously.

The implementation supports various features including:

* Computing energies, forces, and stresses
* Batched calculations for multiple systems

Notes:
    This module depends on the GGNN package and implements the ModelInterface
    for compatibility with the broader TorchSim framework.
"""

import traceback
import warnings
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from typing import Any

import torch
import copy
import torch_sim as ts

from torch_geometric.data import Batch
from torch_sim.models.interface import ModelInterface
from torch_sim.neighbors import torchsim_nl


from fairchem.core.common.utils import update_config

try:
    from GGNN.model.EquFlash.equflash import EquFlash
    from GGNN.trainer.trainer import Trainer
    from GGNN.common.calculator import convert_compiled_ckpt
except (ImportError, ModuleNotFoundError) as exc:
    warnings.warn(f"GGNN import failed: {traceback.format_exc()}", stacklevel=2)


class GgnnModel(ModelInterface):
    """ """

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        compute_forces: bool = True,
        compute_stress: bool = True,
    ) -> None:
        """Initialize the GGNN model for energy and force calculations"""
        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress

        checkpoint = torch.load(
            model, map_location=torch.device("cpu"), weights_only=False
        )
        checkpoint = convert_compiled_ckpt(checkpoint)
        config = checkpoint["config"]


        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        ### backwards compatability with OCP v<2.0
        config = update_config(config)

        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = str(model)

        self.trainer = Trainer(
            task=config.get("task", {}),
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_functions=config["loss_functions"],
            evaluation_metrics=config["evaluation_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=False,
            amp=config.get("amp", False),
            inference_only=True,
        )
        self.trainer.load_checkpoint(None, checkpoint, inference_only=True)

    def forward(  # noqa: C901
        self, state: ts.SimState 
    ) -> dict[str, torch.Tensor]:
        """Compute energies, forces, and stresses for the given atomic systems.

        Processes the provided state information and computes energies, forces, and
        stresses using the underlying GGNN model. Handles batched calculations for
        multiple systems and constructs the necessary neighbor lists.

        Args:
            state (SimState ): State object containing positions, cell,
                and other system information. Can be either a SimState object or a
                dictionary with the relevant fields.

        Returns:
            dict[str, torch.Tensor]: Computed properties:
                - 'energy': System energies with shape [n_systems]
                - 'forces': Atomic forces with shape [n_atoms, 3] if compute_forces=True
                - 'stress': System stresses with shape [n_systems, 3, 3] if
                    compute_stress=True

        """
        sim_state = (
            state
            if isinstance(state, ts.SimState)
            else ts.SimState(**state, masses=torch.ones_like(state["positions"]))
        )

        # Use system_idx from init if not provided
        if sim_state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError(
                    "System indices must be provided if not set during initialization"
                )
            sim_state.system_idx = self.system_idx

        # Batched neighbor list using linked-cell algorithm
        # edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
        #     sim_state.positions,
        #     sim_state.row_vector_cell,
        #     sim_state.pbc,
        #     self.r_max,
        #     sim_state.system_idx,
        # )
        # # Convert unit cell shift indices to Cartesian shifts
        # shifts = ts.transforms.compute_cell_shifts(
        #     sim_state.row_vector_cell, unit_shifts, mapping_system
        # )

        natoms = torch.bincount(state.system_idx)
        # Build data for GGNN
        3 * (sim_state.system_idx.max() + 1)
        data = Batch(
            pos=state.positions.to(torch.float32),
            cell=state.row_vector_cell.to(torch.float32).contiguous(),
            atomic_numbers=state.atomic_numbers,
            natoms=natoms,
            pbc=state.pbc.repeat(sim_state.system_idx.max() + 1),
            batch=sim_state.system_idx,
        )
        out = self.trainer.predict(data, per_image=False, disable_tqdm=True)

        results: dict[str, torch.Tensor] = {}

        # Process energy
        energy = out["energy"]
        if energy is not None:
            results["energy"] = energy.detach().to(self._dtype)
        else:
            results["energy"] = torch.zeros(self.n_systems, device=self.device).to(
                self._dtype
            )

        # Process forces
        if self.compute_forces:
            forces = out["forces"]
            if forces is not None:
                results["forces"] = forces.detach().to(self._dtype)

        # Process stress
        if self.compute_stress:
            stress = out["stress"]
            if stress is not None:
                results["stress"] = stress.detach().reshape(-1, 3, 3).to(self._dtype)
        return results
