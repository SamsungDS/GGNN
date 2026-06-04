from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    
import torch
from ase import Atoms

from fairchem.core.preprocessing import AtomsToGraphs

class AtomsArraysToGraphs(AtomsToGraphs):
    
    # Oct 1, 2025
    # This class inherits from AtomsToGraphs in fairchem-core (tested in v1.10.0)
    # with new feature, allowing one to store custom data from ase.Atoms.arrays
    # in graph data by supplying in config YAML file the following:
    
    # dataset: a2g_args: r_array_keys

    # REFERENCE:
    # https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.10.0/src/fairchem/core/preprocessing/atoms_to_graphs.py                 ORIGINAL AtomsToGraphs
    
    def __init__(
        self,
        max_neigh: int = 200,
        radius: int = 6,
        r_energy: bool = False,
        r_forces: bool = False,
        r_distances: bool = False,
        r_edges: bool = True,
        r_fixed: bool = True,
        r_pbc: bool = False,
        r_stress: bool = False,
        r_data_keys: Sequence[str] | None = None,
        molecule_cell_size: float | None = None,
        r_array_keys: Sequence[str] | None = None
    ) -> None:
        
        # One new argument:    r_array_keys,
        #     which designates keys in atoms.arrays to extract
        #     EX. r_array_keys = [ 'born_charge' ]
        # All else are identical to AtomsToGraphs.
        
        super().__init__(
            max_neigh=max_neigh,
            radius=radius,
            r_energy=r_energy,
            r_forces=r_forces,
            r_distances=r_distances,
            r_edges=r_edges,
            r_fixed=r_fixed,
            r_pbc=r_pbc,
            r_stress=r_stress,
            r_data_keys=r_data_keys,
            molecule_cell_size=molecule_cell_size
        )
        self.r_array_keys = r_array_keys

    def convert(self, atoms: Atoms, sid=None):

        data = super().convert(atoms=atoms, sid=sid)

        if self.r_array_keys is not None:
            for data_key in self.r_array_keys:
                if data_key in atoms.arrays:
                    data[data_key] = torch.tensor(
                        atoms.arrays[data_key], dtype=torch.float32
                    )

        return data