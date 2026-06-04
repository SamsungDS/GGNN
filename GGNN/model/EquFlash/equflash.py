#############
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad


import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode


from e3nn.o3 import Irreps

from .statistics import read_statistics
from .nn.node_embedding import OneHotNodeEmbedding
from .nn.edge_embedding import EdgeEmbedding
from .nn.interaction_blocks import InteractionBlock
from .nn.atomic_reduce import AtomicReduce
from .nn.force_output import ForceStressOutputFromEdge
from .nn.activation import ACTIVATION, ACTIVATION_DICT
from .nn.feed_foward import FeedForward


import e3nn
from e3nn.o3 import Irreps, FullTensorProduct, FullyConnectedTensorProduct


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


@registry.register_model("equflash")
@compile_mode("script")
class EquFlash(nn.Module):
    def __init__(self, config: dict, dataset=None, device=None, lammps_mliap=False):
        super().__init__()
        self.config = config
        self.regress_forces = self.config.get("regress_forces", True)
        self.regress_stress = self.config.get("regress_stress", True)

        if self.config.get("type_map", None) == None:
            # if type_map not given read from statistics
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
                    "type_map": type_map,
                    "chemical_species": chemical_species,
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
        weight_nn_hidden = self.config.get("weight_nn_hidden_neurons", [64, 64])

        use_bias_in_linear = self.config.get("use_bias_in_linear", False)
        conv_denominator = self.config.get("conv_denominator", 1.0)
        train_conv_denominator = self.config.get("train_denominator", False)
        conv_type = self.config.get("conv_kwargs", {}).get("conv_type", "cueq")

        normalize_sph = self.config.get("normalize_sph", True)
        self_connection = self.config.get("self_connection", "linear")
        act_radial, act_scalar, act_gate = self.initialize_activations()
        symmetric_contraction = self.config.get("symmetric_contraction", False)
        feed_forward = self.config.get("feed_forward", False)
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

        weight_nn_layers = [radial_basis_num] + weight_nn_hidden

        interaction_blocks = []
        for t in range(num_convolution_layer):
            irreps_out = Irreps(irreps_manual[t + 1])

            interaction_block = InteractionBlock(
                irreps_x=irreps_x,
                irreps_filter=irreps_filter,
                irreps_out=irreps_out,
                weight_nn_layers=weight_nn_layers,
                conv_denominator=conv_denominator,
                act_radial=act_radial,
                train_conv_denominator=train_conv_denominator,
                act_scalar=act_scalar,
                act_gate=act_gate,
                bias_in_linear=use_bias_in_linear,
                self_connection=self_connection,
                conv_type=conv_type,
                symmetric_contraction=symmetric_contraction,
                feed_forward=feed_forward,
                use_lammps_mliap=lammps_mliap,
            )
            interaction_blocks.append(interaction_block)

            irreps_x = irreps_out
        self.interaction_blocks = torch.nn.ModuleList(interaction_blocks)
        self.atomic_reduce = AtomicReduce(
            irreps_x=irreps_x,
            shift=shift,
            scale=scale,
            type_map=type_map,
            use_bias_in_linear=use_bias_in_linear,
        )

        self.force_output = ForceStressOutputFromEdge()

    def initialize_activations(self):
        # Select radial activation (default: silu)
        act_radial = ACTIVATION[self.config.get("act_radial", "silu")]
        # Build scalar and gate activation maps
        act_scalar = {
            key: ACTIVATION_DICT[key][value]
            for key, value in self.config["act_scalar"].items()
        }
        act_gate = {
            key: ACTIVATION_DICT[key][value]
            for key, value in self.config["act_gate"].items()
        }
        return act_radial, act_scalar, act_gate

    @conditional_grad(torch.enable_grad())
    def forward(self, batch):

        if not self.lammps_mliap:
            batch["atomic_numbers"] = batch["atomic_numbers"].long()

        batch["edge_vec"].requires_grad_(True)
        batch = self.node_embedding(batch)
        batch = self.edge_embedding(batch)
        for interaction_block in self.interaction_blocks:
            batch = interaction_block(batch)
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
