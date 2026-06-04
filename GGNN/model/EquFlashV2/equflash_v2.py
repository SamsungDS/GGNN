#############
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad


from typing import Dict, Type
from copy import deepcopy
import os

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode


from e3nn.o3 import Irreps, Linear

from .statistics import read_statistics
from .nn.interaction_blocks import (
    InteractionBlock,
    NequIPInteractionBlock,
    FeedForwardBlock,
)
from .nn.node_embedding import OneHotNodeEmbedding
from .nn.edge_embedding import EdgeEmbedding
from .nn.atomic_reduce import AtomicReduce
from .nn.force_output import ForceStressOutputFromEdge
from .nn.normalization import EquivariantRMSNormArraySphericalHarmonicsV2


import e3nn
from e3nn.o3 import Irreps, FullTensorProduct
import torch


def generate_irreps(channel, lmax, num_layer, is_parity):
    irreps_manual = [Irreps(f"{channel}x0e")]
    irreps_edge = Irreps(
        [
            (1, (l, 1 if (l % 2 == 0 or is_parity == False) else -1))
            for l in range(lmax + 1)
        ]
    )

    for i in range(num_layer):
        irreps = irreps_manual[-1]
        tp = FullTensorProduct(irreps, irreps_edge)
        irreps_out = tp.irreps_out.simplify()
        target = []
        for mul, ir in irreps_out:
            if ir.l < lmax + 1:
                target.append((channel, ir))
        irreps_manual.append(Irreps(target))
    irreps_manual[-1] = irreps_manual[0]
    target = irreps_manual[-1]
    for i in range(num_layer - 1):
        irreps_in = irreps_manual[-i - 2]
        target_ = [ir for (mul, ir) in target]
        filtered = []
        for mul1, ir1 in irreps_in:
            is_available = False
            for mul2, ir2 in irreps_edge:
                for ir in ir1 * ir2:
                    if ir in target_:
                        is_available = True
                        break
            if is_available:
                filtered.append((mul1, ir1))

        target = Irreps(filtered)
        irreps_manual[-i - 2] = target
    return irreps_manual


INTERACTION_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "equflash": InteractionBlock,
    "nequip": NequIPInteractionBlock,
}

FF_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "feedforward": FeedForwardBlock,
    "identity": torch.nn.Identity,
}


@registry.register_model("equflashv2")
class EquFlashV2(nn.Module):
    def __init__(self, config: dict, dataset=None, device=None, lammps_mliap=False):
        super().__init__()
        self.config = config
        self.regress_forces = self.config.get("regress_forces", True)
        self.regress_stress = self.config.get("regress_stress", True)
        scale = self.config.get("scale", "per_atom_energy_std")
        shift = self.config.get("shift", "elemwise_reference_energies")
        type_map = self.config.get("type_map", None)
        if (
            isinstance(scale, str) or isinstance(shift, str) or type_map == None
        ):  # self.config.get("stat_dir", None) is not None:
            scale = self.config.get("scale", "per_atom_energy_std")
            shift = self.config.get("shift", "elemwise_reference_energies")
            scale_stat, shift_stat, type_map, chemical_species = read_statistics(
                self.config.get("stat_dir", None),
                scale_by=scale,
                shift_by=shift,
            )

            self.config.update(
                {
                    "scale": scale if isinstance(scale, float) else scale_stat,
                    "shift": shift if isinstance(shift, float) else shift_stat,
                }
            )
            if self.config.get("type_map", None) == None:
                self.config.update(
                    {
                        "type_map": type_map,
                    }
                )

        scale, shift, type_map = (
            self.config.get("scale", 1.0),
            self.config.get("shift", 0.0),
            self.config["type_map"],
        )
        num_species = len(type_map)

        self.lammps_mliap = lammps_mliap

        num_convolution_layer = self.config.get("num_convolution_layer", 5)
        channel = self.config.get("channel", 32)
        lmax = self.config.get("lmax", 3)
        is_parity = self.config.get("is_parity", True)

        if "irreps_manual" in self.config:
            irreps_manual = self.config["irreps_manual"]
            num_convolution_layer = len(irreps_manual) - 1
        else:
            irreps_manual = generate_irreps(
                channel, lmax, num_convolution_layer, is_parity
            )

        cutoff = self.config.get("cutoff", 6.0)
        conv_type = self.config.get("conv_type", "full")
        conv_kwargs = self.config.get("conv_kwargs", {})
        sc_type = self.config.get("sc_type", "identity")
        nonlinear_type = self.config.get("nonlinear_type", "gate")
        nonlinear_kwargs = self.config.get("nonlinear_kwargs", {})

        use_bias_in_linear = False
        normalize_sph = self.config.get("normalize_sph", True)
        irreps_x = Irreps(irreps_manual[0])

        self.node_embedding = OneHotNodeEmbedding(
            num_species, irreps_x, use_bias_in_linear, type_map
        )
        self.edge_embedding = EdgeEmbedding(
            cutoff,
            self.config["cutoff_function"],
            self.config["radial_basis"],
            lmax,
            is_parity,
            normalize_sph,
        )

        irreps_filter = self.edge_embedding.spherical.irreps_out
        radial_basis_num = self.edge_embedding.basis_function.num_basis

        interaction_type = config.get("interaction_type", "equflash")
        interaction_kwargs = config.get("interaction_kwargs", {"conv_kwargs": {}})
        interaction_kwargs.get("conv_kwargs", {}).update(conv_kwargs)
        interaction_kwargs.get("conv_kwargs", {}).update(
            {"radial_basis_num": radial_basis_num}
        )
        feedforward_type = config.get("feedforward_type", "feedforward")
        feedforward_kwargs = config.get("feedforward_kwargs", {})
        interaction_cls = INTERACTION_REGISTRY[interaction_type]
        ff_cls = FF_REGISTRY[feedforward_type]
        interaction_blocks = []
        ff_blocks = []

        for t in range(num_convolution_layer):
            irreps_out = Irreps(irreps_manual[t + 1])
            interaction_block = interaction_cls(
                irreps_x=irreps_x,
                irreps_filter=irreps_filter,
                irreps_out=irreps_out,
                **interaction_kwargs,
            )

            ff_block = ff_cls(irreps_out, irreps_out, **deepcopy(feedforward_kwargs))
            interaction_blocks.append(interaction_block)

            ff_blocks.append(ff_block)

            irreps_x = irreps_out
        self.interaction_blocks = torch.nn.ModuleList(interaction_blocks)
        self.ff_blocks = torch.nn.ModuleList(ff_blocks)
        self.atomic_reduce = AtomicReduce(
            irreps_x=irreps_x,
            shift=shift,
            scale=scale,
            type_map=type_map,
            use_bias_in_linear=False,
        )
        self.force_output = ForceStressOutputFromEdge()

    @conditional_grad(torch.enable_grad())
    def forward(self, batch):

        if not self.lammps_mliap:
            batch["atomic_numbers"] = batch["atomic_numbers"].long()

        batch["edge_vec"].requires_grad_(True)
        batch = self.node_embedding(batch)
        batch = self.edge_embedding(batch)
        res = []
        for i, (interaction_block, ff_block) in enumerate(
            zip(self.interaction_blocks, self.ff_blocks)
        ):
            batch = interaction_block(batch)
            batch = ff_block(batch)

        batch = self.atomic_reduce(batch)

        out = {}
        out["energy"] = batch["total_energy"]
        out["atomic_energy"] = batch["atomic_energy"]

        if not self.lammps_mliap:
            forces, stress = self.force_output(batch)
        if self.regress_forces:
            out["forces"] = forces
        if self.regress_stress:
            out["stress"] = stress

        return out

    @torch.jit.ignore
    def no_weight_decay(self) -> set:
        no_wd_list = []
        named_parameters_list = [name for name, _ in self.named_parameters()]
        for module_name, module in self.named_modules():
            if isinstance(
                module,
                (
                    torch.nn.Linear,
                    Linear,
                    torch.nn.LayerNorm,
                    EquivariantRMSNormArraySphericalHarmonicsV2,
                    EdgeEmbedding,
                    OneHotNodeEmbedding,
                ),
            ):
                for parameter_name, _ in module.named_parameters():
                    if (
                        isinstance(module, (torch.nn.Linear, Linear))
                        and "weight" in parameter_name
                    ):
                        continue
                    global_parameter_name = module_name + "." + parameter_name
                    assert global_parameter_name in named_parameters_list
                    no_wd_list.append(global_parameter_name)

        return set(no_wd_list)
