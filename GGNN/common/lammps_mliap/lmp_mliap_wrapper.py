# This file is a part of the `nequip` package. Please see LICENSE and README at the root for information on using it.

import torch
import os


try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ModuleNotFoundError:
    raise ImportError(
        "LAMMPS ML-IAP has to be installed in the Python environment for NequIP's ML-IAP integration. "
        "See https://nequip.readthedocs.io/en/latest/integrations/lammps/mliap.html for installation instructions."
    )

from typing import List
from GGNN.common.calculator import UCalculator,convert_compiled_ckpt
from GGNN.model.SevenNet.sevennet import SevenNet, convert_compiled_to_original_sevennet

from GGNN.model.EquFlash import EquFlash
from fairchem.core.common.utils import (
    match_state_dict,
)
from fairchem.core.common.registry import registry
from fairchem.core.modules.normalization.element_references import (
    create_element_references,
)
from fairchem.core.modules.normalization.normalizer import (
    create_normalizer,
)

from torch_geometric.data import Batch


class GGNNLAMMPSMLIAPWrapper(MLIAPUnified):
    """LAMMPS-MLIAP interface for NequIP framework models."""

    def __init__(
        self,
        ckpt_path: str,
        **kwargs,
    ):
        # this is a white lie, unsure if strictly necessary, but just in case
        super().__init__()
        self.ckpt_path = ckpt_path
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.model = None
        self.device = None
        # model_name = ckpt["config"]["model"].pop("name")
        # self.model = registry.get_model_class(model_name)(self.ckpt["config"]["model"])
        # self.model = registry.get_model_class(model_name)(ckpt["config"]["model"])
        # import pdb;pdb.set_trace()
        # to placate the interface
        self.nparams = 1
        self.ndescriptors = 1

        # === set model-depnedent params ===
        # what is rcutfac
        # self.rcutfac = 0.5 * float(model.metadata[graph_model.R_MAX_KEY])
        self.rcutfac = 3.0
        import ase

        self.element_types = (
            ase.data.chemical_symbols
        )  # ckpt["config"]["model"]["chemical_species"]
        self.ckpt = ckpt

    def compute_forces(self, lmp_data):
        # === lazily load model ===
        if self.model == None:
            self.device = (
                "cuda" if "kokkos" in lmp_data.__class__.__module__.lower() else "cpu"
            )
            self.ckpt = convert_compiled_ckpt(self.ckpt)
            model_name = self.ckpt["config"]["model"].pop("name")
            # self.model = registry.get_model_class(model_name)(self.ckpt["config"]["model"])
            if "conv_kwargs" not in self.ckpt["config"]["model"].keys():
                self.ckpt["config"]["model"]["conv_kwargs"] = {}
            self.ckpt["config"]["model"]["conv_kwargs"]["use_lammps_mliap"] = True
            

            self.model = registry.get_model_class(model_name)(
                self.ckpt["config"]["model"], lammps_mliap=True
            ).to(self.device)

            new_dict = match_state_dict(
                self.model.state_dict(), self.ckpt["state_dict"]
            )
            self.model.load_state_dict(new_dict, strict=False)
            # print(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
            rank = eval(os.environ.get("OMPI_COMM_WORLD_RANK", "0"))
            if rank == 0:
                print(self.model)
            self.normalizers = {}
            for key, state_dict in self.ckpt.get("normalizers", {}).items():
                ### Convert old normalizer keys to new target keys
                if key == "target":
                    target_key = "energy"
                elif key == "grad_target":
                    target_key = "forces"
                else:
                    target_key = key

                if target_key not in self.normalizers:
                    self.normalizers[target_key] = create_normalizer(
                        state_dict=state_dict
                    )
                self.normalizers[target_key].to(self.device)
            self.elementrefs = {}
            for key, state_dict in self.ckpt.get("elementrefs", {}).items():
                if key not in self.elementrefs:
                    self.elementrefs[key] = create_element_references(
                        state_dict=state_dict
                    )
                else:
                    mkeys = self.elementrefs[key].load_state_dict(state_dict)
                    assert len(mkeys.missing_keys) == 0
                    assert len(mkeys.unexpected_keys) == 0

                self.elementrefs[key].to(self.device)
            self.model.regress_forces = False
            self.model.regress_stress = False

        if lmp_data.nlocal == 0 or lmp_data.npairs <= 1:
            return

        # === create input data ===

        # NOTE
        # This LAMMPS ML-IAP integration introduces a new dimension of having `num_local` vs `num_local + num_ghost` number of nodes.
        # There are three crucial dimensions to be aware of `num_edges`, `num_local`, `num_local + num_ghost`.
        # The following input tensors have the following shapes.
        # - `edge_vectors`: (num_edges, 3)
        # - `edge_idxs`: (2, num_edges)
        # - `atom_types`: (num_local + num_ghost)

        # This LAMMPS ML-IAP wrapper can handle output `atomic_energy` having either shape `num_local` or `num_local + num_ghost` based on the (uncompiled) size check.

        # Models can perform optimizations based on an understanding of when ghost atoms matter or not and are responsible for carefully handling internal shape logic.
        # Examples include:
        # - edge -> node scatter operations / nodewise operations (e.g. in `nequip/nn/interaction_block.py`)
        # - nodewise operations that involve `atom_types` (since `atom_types` is `num_local + num_ghost`), e.g. in `PerTypeScaleShift` and `ZBL`.

        # TODO: we have yet to exploit per-edge-type cutoffs by pruning the edge vectors and neighborlist
        # make sure edge vectors `requires_grad`
        edge_vectors = (
            torch.as_tensor(lmp_data.rij, dtype=torch.float64).to(self.device).float()
        )
        edge_vectors.requires_grad_(True)
        edge_index = torch.vstack(
            [
                torch.as_tensor(lmp_data.pair_i, dtype=torch.int64).to(self.device),
                torch.as_tensor(lmp_data.pair_j, dtype=torch.int64).to(self.device),
            ],
        )

        atomic_type = torch.as_tensor(lmp_data.elems, dtype=torch.int64).to(self.device)

        num_atoms = torch.tensor(
            [lmp_data.nlocal, lmp_data.ntotal - lmp_data.nlocal], dtype=torch.int64
        ).to(self.device)
        natoms = torch.tensor(lmp_data.nlocal, dtype=torch.int64).to(self.device)
        # import pdb;pdb.set_trace()
        # === run model ===
        # run model and backwards for edge forces
        batch = Batch(
            edge_vec=edge_vectors,
            atomic_numbers=(atomic_type[: lmp_data.nlocal].long()),
            edge_index=edge_index,
            num_atoms=num_atoms,
            natoms=natoms,
            lmp_data=lmp_data,
        )
        out = self.model(batch)
        # correct sign convention for consistency with LAMMPS
        edge_forces = torch.autograd.grad(
            [out["energy"].sum()],
            [edge_vectors],
        )[0]
        # === pass outputs to LAMMPS ===
        # handle ghosts
        atomic_energies = out["atomic_energy"].squeeze()

        # = nequip_data_out[
        #     AtomicDataDict.PER_ATOM_ENERGY_KEY
        # ].view(-1)

        # shape-dependent control flow, but should be outside of compiled model
        if atomic_energies.size(0) != lmp_data.nlocal:
            atomic_energies = torch.narrow(atomic_energies, 0, 0, lmp_data.nlocal)
            total_energy = torch.sum(atomic_energies)
        else:
            total_energy = torch.sum(atomic_energies)
        if "forces" in self.normalizers:
            edge_forces = self.normalizers["forces"](edge_forces)
        if "energy" in self.normalizers:
            total_energy = self.normalizers["energy"](total_energy)
        if "energy" in self.elementrefs:
            batch.batch = torch.zeros(natoms, device=self.device, dtype=torch.int)
            total_energy = self.elementrefs["energy"](
                total_energy.reshape(1, -1), batch
            )

        # update LAMMPS variables
        lmp_data.energy = total_energy
        lmp_data.update_pair_forces_gpu(edge_forces.double())

    def compute_descriptors(self, lmp_data):
        pass

    def compute_gradients(self, lmp_data):
        pass
