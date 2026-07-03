#############
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad

import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

from e3nn.o3 import Irreps

from .statistics import read_statistics
from .nn.node_embedding import OneHotNodeEmbedding
from .nn.edge_embedding import EdgeEmbeddingField
from .nn.interaction_blocks import InteractionBlock
from .nn.atomic_reduce import AtomicReduce
from .nn.force_output import ForceStressOutputFromEdge, BECOutputfromPol, PolOutput
from .nn.activation import ACTIVATION, ACTIVATION_DICT
from .nn.feed_foward import FeedForward


import e3nn
from e3nn.o3 import Irreps, FullTensorProduct, FullyConnectedTensorProduct
from e3nn.io import CartesianTensor
from .equflash import EquFlash
from .nn.atomic_reduce import AtomicReducePol, AtomicReduceBEC



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


@registry.register_model("equflash_pol")
@compile_mode("script")
class EquFlash_pol(EquFlash):
    def __init__(self, config: dict, dataset=None, device=None, lammps_mliap=False):
        super().__init__(config)
        self.regress_bec = self.config.get("regress_bec", True)

        if "irreps_manual" in self.config:
            irreps_manual = self.config["irreps_manual"]
            num_convolution_layer = len(irreps_manual) - 1
        else:
            num_convolution_layer = self.config.get("num_convolution_layer", 5)
            channel = self.config.get("channel", 32)
            lmax = self.config.get("lmax", 3)
            is_parity = self.config.get("is_parity", True)
            irreps_manual = generate_irreps(
                channel, lmax, num_convolution_layer, is_parity
            )

        irreps_out = Irreps(irreps_manual[num_convolution_layer])

        if self.config.get("type_map", None) == None:
            # if type_map not given read from statistics
            scale = self.config.get("scale", "elemwise_reference_energies")
            shift = self.config.get("shift", "per_atom_energy_std")
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

        use_bias_in_linear = self.config.get("use_bias_in_linear", False)



        self.atomic_reduce = AtomicReducePol(
            irreps_x=irreps_out,
            shift=shift,
            scale=scale,
            type_map=type_map,
            use_bias_in_linear=use_bias_in_linear,
        )

        self.force_output = ForceStressOutputFromEdge()
        self.bec_output = BECOutputfromPol()


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
        out["polarization"] = batch['total_polarization'].double()
        out["born_charge"] = self.bec_output(batch) 
        if not self.lammps_mliap:
            forces, stress = self.force_output(batch)
        if self.regress_forces:
            out["forces"] = forces
        if self.regress_stress:
            out["stress"] = stress

        return out



@registry.register_model("equflash_bec")
@compile_mode("script")
class EquFlash_BEC(EquFlash):
    def __init__(self, config: dict, dataset=None, device=None, lammps_mliap=False):
        super().__init__(config)
        self.regress_bec = self.config.get("regress_bec", True)

        if "irreps_manual" in self.config:
            irreps_manual = self.config["irreps_manual"]
            num_convolution_layer = len(irreps_manual) - 1
        else:
            num_convolution_layer = self.config.get("num_convolution_layer", 5)
            channel = self.config.get("channel", 32)
            lmax = self.config.get("lmax", 3)
            is_parity = self.config.get("is_parity", True)
            irreps_manual = generate_irreps(
                channel, lmax, num_convolution_layer, is_parity
            )

        irreps_out = Irreps(irreps_manual[num_convolution_layer])

        if self.config.get("type_map", None) == None:
            # if type_map not given read from statistics
            scale = self.config.get("scale", "elemwise_reference_energies")
            shift = self.config.get("shift", "per_atom_energy_std")
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

        use_bias_in_linear = self.config.get("use_bias_in_linear", False)



        self.atomic_reduce = AtomicReduceBEC(
            irreps_x=irreps_out,
            shift=shift,
            scale=scale,
            type_map=type_map,
            use_bias_in_linear=use_bias_in_linear,
        )

        self.force_output = ForceStressOutputFromEdge()
        self.bec_tensor = CartesianTensor("ij")   # 1x0e+1x1e+1x2e

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
        
        batch["bec"] = torch.zeros(batch["pos"].shape[0], 9, device=batch["pos"].device, dtype=batch["pos"].dtype)
        batch = self.atomic_reduce(batch)

        out = {}
        out["energy"] = batch["total_energy"]
        out["atomic_energy"] = batch["atomic_energy"]
        out["born_charge"] = self.bec_tensor.to_cartesian(batch["bec"])

        if not self.lammps_mliap:
            forces, stress = self.force_output(batch)
        if self.regress_forces:
            out["forces"] = forces
        if self.regress_stress:
            out["stress"] = stress

        return out

@registry.register_model("equflash_field")
@compile_mode("script")
class EquFlash_field(EquFlash):
    def __init__(self, config: dict, dataset=None, device=None, lammps_mliap=False):
        super().__init__(config)

        use_bias_in_linear = self.config.get("use_bias_in_linear", False)
        weight_nn_hidden = self.config.get("weight_nn_hidden_neurons", [64, 64])
        num_convolution_layer = self.config.get("num_convolution_layer", 5)
        channel = self.config.get("channel", 32)
        lmax = self.config.get("lmax", 3)
        is_parity = self.config.get("is_parity", True)
        cutoff = self.config.get("cutoff", 6.0)
        normalize_sph = self.config.get("normalize_sph", True)

        conv_denominator = self.config.get("conv_denominator", 1.0)
        train_conv_denominator = self.config.get("train_denominator", False)
        conv_type = self.config.get("conv_kwargs", {}).get("conv_type", "cueq")

        self_connection = self.config.get("self_connection", "linear")
        act_radial, act_scalar, act_gate = self.initialize_activations()
        symmetric_contraction = self.config.get("symmetric_contraction", False)
        feed_forward = self.config.get("feed_forward", False)

 
        if self.config.get("type_map", None) == None:
            # if type_map not given read from statistics
            scale = self.config.get("scale", "elemwise_reference_energies")
            shift = self.config.get("shift", "per_atom_energy_std")
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
        
        if "irreps_manual" in self.config:
            irreps_manual = self.config["irreps_manual"]
            num_convolution_layer = len(irreps_manual) - 1
        else:
            irreps_manual = generate_irreps(
                channel, lmax, num_convolution_layer, is_parity
            )
        irreps_x = Irreps(irreps_manual[0])


        self.edge_embedding = EdgeEmbeddingField(
            cutoff,
            self.config["cutoff_function"],
            self.config["radial_basis"],
            lmax,
            is_parity,
            normalize_sph,
        )
        ############################################################
        #double edge irreps due to field encoding
        #can be moved to EdgeEmbeddingField
        irreps_filter = self.edge_embedding.spherical.irreps_out

        parts = []
        for mul, ir in irreps_filter:
            parts.extend([str(ir)] * mul)
        irreps_str =  " + ".join(parts)
        irreps_filter=Irreps(irreps_str+'+'+irreps_str)

        ##############################################


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
        self.polarization_output = PolOutput()
        self.bec_output = BECOutputfromPol()


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

        numbatch= batch["energy"].shape[0]
        batch["edge_frame"] = torch.searchsorted(batch["ptr"], batch["edge_index"][0], right=True)-1
        batch.setdefault("field", torch.zeros(numbatch, 3, device='cuda'))
        batch["field"].requires_grad_(True)
        batch["edge_field"] = batch["field"][batch["edge_frame"]]


        batch["edge_vec"].requires_grad_(True)
        batch = self.node_embedding(batch)
        batch = self.edge_embedding(batch)
        for interaction_block in self.interaction_blocks:
            batch = interaction_block(batch)
        
        batch = self.atomic_reduce(batch)

        out = {}
        out["energy"] = batch["total_energy"]
        out["atomic_energy"] = batch["atomic_energy"]

        batch["total_polarization"] = self.polarization_output(batch).double()
        out['polarization'] = batch["total_polarization"]
        out["born_charge"] = self.bec_output(batch).double()

        if not self.lammps_mliap:
            forces, stress = self.force_output(batch)
        if self.regress_forces:
            out["forces"] = forces
        if self.regress_stress:
            out["stress"] = stress
        

        return out
