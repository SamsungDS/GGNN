from typing import List

import types
import torch
import itertools
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps
from e3nn.util.jit import compile_mode
import cuequivariance as cue
import cuequivariance_torch as cuet
from .util import gcd_irreps
from .activation import ShiftedSoftPlus
from ._ghost_exchange_base import LAMMPSMLIAPGhostExchangeModule


def with_cueq_conv_fusion(conv_tp: torch.nn.Module) -> torch.nn.Module:
    """Wraps a cuet.ConvTensorProduct to use conv fusion"""
    conv_tp.original_forward = conv_tp.forward
    num_segment = conv_tp.m.buffer_num_segments[0]
    num_operands = conv_tp.m.operand_extent
    conv_tp.weight_numel = num_segment * num_operands

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        tp_weights: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[1]
        receiver = edge_index[0]
        return self.original_forward(
            [tp_weights, node_feats, edge_attrs],
            {1: sender},
            {0: node_feats},
            {0: receiver},
        )[0]

    conv_tp.forward = types.MethodType(forward, conv_tp)
    return conv_tp


def cue_uvu_descriptor(irreps1, irreps2, irreps3):
    d = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kuv+ijk")
    for mul, ir in irreps1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(2, (ir.dim, mul))
    for mul, ir in irreps3:
        d.add_segment(3, (ir.dim, mul, 1))

    num_l3 = torch.zeros(len(irreps3), dtype=torch.int32)
    for (i1, (mul1, ir1)), (i2, (mul2, ir2)), (i3, (mul3, ir3)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2), enumerate(irreps3)
    ):
        if ir3 in ir1 * ir2:
            num_l3[i3] += 1
    for (i1, (mul1, ir1)), (i2, (mul2, ir2)), (i3, (mul3, ir3)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2), enumerate(irreps3)
    ):
        if ir3 in ir1 * ir2:
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(
                    None, i1, i2, i3, c=cg / num_l3[i3], dims={"u": mul1, "v": mul2}
                )
    d = d.normalize_paths_for_operand(-1)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps1.new_scalars(d.operands[0].size), cue.ir_mul),
            cue.IrrepsAndLayout(irreps1, cue.ir_mul),
            cue.IrrepsAndLayout(irreps2, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


class FullConv(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_nn_layers: List[int],
        radial_basis_num: int,
        act_radial=torch.nn.functional.silu,
        denominator: float = 1.0,
        train_denominator: bool = False,
        use_lammps_mliap: bool = False,
    ):
        super().__init__()

        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )

        # uniform channel size for irreps_x
        try:
            # uniform channel size for irreps_x
            poly = cue.descriptors.channelwise_tensor_product(
                cue.Irreps("O3", irreps_x),
                cue.Irreps("O3", irreps_filter),
                cue.Irreps("O3", irreps_out),
                simplify_irreps3=True,
            )
            self.convolution = with_cueq_conv_fusion(
                cuet.SegmentedPolynomial(
                    poly.flatten_coefficient_modes().squeeze_modes().polynomial,
                    math_dtype=torch.get_default_dtype(),
                    method="uniform_1d",
                )
            )
        except:
            # non-uniform channel size for irreps_x
            poly = cue.descriptors.channelwise_tensor_product(
                cue.Irreps("O3", irreps_x),
                cue.Irreps("O3", irreps_filter),
                cue.Irreps("O3", irreps_out),
            )
            mul_gcd = gcd_irreps(irreps_x)
            poly = poly.flatten_modes("ijk").squeeze_modes("v")
            poly = poly.apply_fn(lambda op, d: (op, d.split_mode("u", mul_gcd)))
            self.convolution = with_cueq_conv_fusion(
                cuet.SegmentedPolynomial(
                    poly.polynomial,
                    math_dtype=torch.get_default_dtype(),
                    method="uniform_1d",
                )
            )
        weight_numel = poly.inputs[0].dim
        self.weight_nn_kwargs = dict(
            hs=[radial_basis_num] + weight_nn_layers + [weight_numel], act=act_radial
        )
        irreps_mid = Irreps(poly.outputs[0].irreps.__repr__())
        self.weight_nn = FullyConnectedNet(**self.weight_nn_kwargs)
        self.irreps_in = irreps_x
        self.irreps_filter = irreps_filter
        self.irreps_out = irreps_mid.simplify()
        self._comm_size = irreps_x.dim  # used in parallel
        self.use_lammps_mliap = use_lammps_mliap
        if self.use_lammps_mliap:
            self.exchange = LAMMPSMLIAPGhostExchangeModule()

    def __repr__(self):
        return f"FullConv(\n\t{self.weight_nn},\n\tconv({self.irreps_in} x {self.irreps_filter} -> {self.irreps_out})\n)"

    def forward(self, x, edge_index, edge_embedding, edge_attr, lmp_data=None):

        weight = self.weight_nn(edge_embedding)

        # self.exchange(x, data.lmp_data)
        # note that 1 -> src 0 -> dst
        edge_src = edge_index[1]
        edge_dst = edge_index[0]
        if self.use_lammps_mliap:
            x = self.exchange(x, lmp_data)
            nlocal = lmp_data.nlocal
        else:
            nlocal = x.shape[0]
        out = self.convolution(x, edge_attr, weight, edge_index)[:nlocal]
        out = out.div(self.denominator)

        return out


class EfficientConv(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_nn_layers: List[int],
        radial_basis_num: int,
        act_radial=torch.nn.functional.silu,
        denominator: float = 1.0,
        train_denominator: bool = False,
        use_lammps_mliap: bool = False,
    ):

        super().__init__()
        self.irreps_in = irreps_x
        self.irreps_filter = irreps_filter
        self.irreps_out = irreps_out
        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )

        d = cue_uvu_descriptor(
            cue.Irreps("O3", irreps_x),
            cue.Irreps("O3", irreps_filter),
            cue.Irreps("O3", irreps_out),
        )
        weight_numel = d.inputs[0].dim
        self.convolution = cuet.SegmentedPolynomial(d.polynomial, method="uniform_1d")

        self.weight_nn_kwargs = dict(
            hs=[radial_basis_num] + weight_nn_layers + [weight_numel],
            act=act_radial,
        )
        self.weight_nn = FullyConnectedNet(**self.weight_nn_kwargs)
        self._comm_size = irreps_x.dim  # used in parallel
        self.use_lammps_mliap = use_lammps_mliap
        if self.use_lammps_mliap:
            self.exchange = LAMMPSMLIAPGhostExchangeModule()

    def forward(self, x, edge_index, edge_embedding, edge_attr, lmp_data=None):

        weight = self.weight_nn(edge_embedding)

        # self.exchange(x, data.lmp_data)
        # note that 1 -> src 0 -> dst
        edge_src = edge_index[1]
        edge_dst = edge_index[0]
        if self.use_lammps_mliap:
            x = self.exchange(x, lmp_data)
            nlocal = lmp_data.nlocal
        else:
            nlocal = x.shape[0]

        out = self.convolution(
            [weight, x, edge_attr],
            {1: edge_src},
            {0: x},
            {0: edge_dst},
        )[0][:nlocal]
        out = out.div(self.denominator)

        return out
