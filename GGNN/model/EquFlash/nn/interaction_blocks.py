import os
from copy import deepcopy
from typing import Callable, List, Optional, Tuple
import torch

from e3nn.o3 import Irreps, FullTensorProduct
import os


from .equivariant_gate import EquivariantGate
from e3nn.o3 import Linear
from .convolution import IntegratedConv
from .symmetric_contraction import SymmetricContraction
from .feed_foward import FeedForward


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_nn_layers: List[int],
        conv_denominator: float,
        act_radial: Callable,
        train_conv_denominator: bool,
        act_scalar: Callable,
        act_gate: Callable,
        bias_in_linear: bool,
        self_connection: str = "linear",
        conv_type="e3nn",
        feed_forward: bool = False,
        symmetric_contraction: bool = False,
        use_lammps_mliap: bool = False,
    ):
        super().__init__()
        if symmetric_contraction:
            act = SymmetricContraction(irreps_out)
            irreps_act = irreps_out
        else:
            act = EquivariantGate(irreps_out, act_scalar, act_gate)
            irreps_act = act.get_gate_irreps_in()
        if self_connection == "linear":
            self.sc = Linear(
                irreps_x,
                irreps_out=irreps_act,
            )
        else:
            raise NotImplementedError()
        target_irrep = [ir for (mul, ir) in irreps_out]
        irreps_conv_out = []
        for mul, ir in FullTensorProduct(irreps_x, irreps_filter).irreps_out.simplify():
            if ir in target_irrep:
                irreps_conv_out.append((mul, ir))
        irreps_conv_out = Irreps(irreps_conv_out)

        self.self_interaction_1 = Linear(
            irreps_x,
            irreps_x,
            biases=bias_in_linear,
        )
        self.conv = IntegratedConv(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_conv_out,
            weight_layer_input_to_hidden=weight_nn_layers,
            weight_layer_act=act_radial,
            denominator=conv_denominator,
            train_denominator=train_conv_denominator,
            conv_type=conv_type,
            use_lammps_mliap=use_lammps_mliap,
        )
        self.self_interaction_2 = Linear(
            irreps_conv_out,
            irreps_act,
            biases=bias_in_linear,
        )
        self.act = act
        if feed_forward:
            self.feed_forward = FeedForward(irreps_out, act_scalar, act_gate)
        else:
            self.feed_forward = None

    def forward(self, data):
        x, edge_index, edge_embedding, edge_attr, lmp_data = (
            data["x"],
            data["edge_index"],
            data["edge_embedding"],
            data["edge_attr"],
            data.get("lmp_data", None),
        )
        sc = self.sc(x)
        x = self.self_interaction_1(x)
        out = self.conv(x, edge_index, edge_embedding, edge_attr, lmp_data)

        x = self.self_interaction_2(out)
        x = x + sc
        x = self.act(x)

        if self.feed_forward:
            x = self.feed_forward(x)
        data["x"] = x
        return data
