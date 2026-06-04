from fairchem.core.preprocessing import AtomsToGraphs as AtomsToGraphs_fairchem
from ase.geometry import wrap_positions
import ase
import torch
import numpy as np
from torch_geometric.data import Data

class AtomsToGraphs(AtomsToGraphs_fairchem):
    
    def convert(self, atoms: ase.Atoms, sid=None):
        """Convert a single atomic structure to a graph.

        Args:
            atoms (ase.atoms.Atoms): An ASE atoms object.

            sid (uniquely identifying object): An identifier that can be used to track the structure in downstream
            tasks. Common sids used in OCP datasets include unique strings or integers.

        Returns:
            data (torch_geometric.data.Data): A torch geometic data object with positions, atomic_numbers, tags,
            and optionally, energy, forces, distances, edges, and periodic boundary conditions.
            Optional properties can included by setting r_property=True when constructing the class.
        """

        # set the atomic numbers, positions, and cell
        positions = np.array(atoms.get_positions(), copy=True)
        pbc = np.array(atoms.pbc, copy=True)
        cell = np.array(atoms.get_cell(complete=True), copy=True)
        positions = wrap_positions(positions, cell, pbc=pbc, eps=0)

        atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.uint8)
        positions = torch.from_numpy(positions)
        cell = torch.from_numpy(cell).view(1, 3, 3)
        natoms = positions.shape[0]

        # initialized to torch.zeros(natoms) if tags missing.
        # https://wiki.fysik.dtu.dk/ase/_modules/ase/atoms.html#Atoms.get_tags
        tags = torch.tensor(atoms.get_tags(), dtype=torch.int)

        # put the minimum data in torch geometric data object
        data = Data(
            cell=cell,
            pos=positions,
            atomic_numbers=atomic_numbers,
            natoms=natoms,
            tags=tags,
        )

        # Optionally add a systemid (sid) to the object
        if sid is not None:
            data.sid = sid

        # optionally include other properties
        if self.r_edges:
            # run internal functions to get padded indices and distances
            atoms_copy = atoms.copy()
            atoms_copy.set_positions(positions)
            split_idx_dist = self._get_neighbors_pymatgen(atoms_copy)
            edge_index, edge_distances, cell_offsets = self._reshape_features(
                *split_idx_dist
            )

            data.edge_index = edge_index
            data.cell_offsets = cell_offsets
            data.edge_distance_vec = self.get_edge_distance_vec(
                positions, edge_index, cell, cell_offsets
            )

            del atoms_copy
        if self.r_energy:
            energy = atoms.get_potential_energy(apply_constraint=False)
            data.energy = energy
        if self.r_forces:
            forces = torch.tensor(
                atoms.get_forces(apply_constraint=False), dtype=torch.float32
            )
            data.forces = forces
        if self.r_stress:
            stress = torch.tensor(
                atoms.get_stress(apply_constraint=False, voigt=False),
                dtype=torch.float32,
            )
            data.stress = stress
        if self.r_distances and self.r_edges:
            data.distances = edge_distances
        if self.r_fixed:
            fixed_idx = torch.zeros(natoms, dtype=torch.int)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms

                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            data.fixed = fixed_idx
        if self.r_pbc:
            data.pbc = torch.tensor(atoms.pbc, dtype=torch.bool)
        if self.r_data_keys is not None:
            for data_key in self.r_data_keys:
                data[data_key] = (
                    atoms.info[data_key]
                    if isinstance(atoms.info[data_key], (int, float, str))
                    else torch.tensor(atoms.info[data_key])
                )

        return data
