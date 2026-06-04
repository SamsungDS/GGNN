from itertools import chain
from typing import List

import types
import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet
from e3nn.o3 import Irreps, TensorProduct, wigner_3j
from e3nn.util.jit import compile_mode

try:
    from flashTP_e3nn import uvu_TP

    FLASHTP_AVAILABLE = True
except:
    FLASHTP_AVAILABLE = False

try:
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    CUEQ_AVAILABLE = True
except:
    CUEQ_AVAILABLE = False


from .activation import ShiftedSoftPlus
from .util import broadcast
from ._ghost_exchange_base import LAMMPSMLIAPGhostExchangeModule


def message_gather(
    node_features: torch.Tensor, edge_dst: torch.Tensor, message: torch.Tensor
):
    index = broadcast(edge_dst, message, 0)
    out_shape = (node_features.shape[0], message.shape[1])
    out = torch.zeros(out_shape, dtype=node_features.dtype, device=node_features.device)
    out.scatter_reduce_(0, index, message, reduce="sum")
    return out


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


@compile_mode("script")
class IntegratedConv(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        weight_layer_input_to_hidden: List[int],
        weight_layer_act=ShiftedSoftPlus,
        denominator: float = 1.0,
        train_denominator: bool = False,
        conv_type: str = "flashtp",
        use_lammps_mliap: bool = False,
    ):
        super().__init__()
        if conv_type.lower() not in ["flashtp", "cueq", "e3nn"]:
            assert (), "conv_type must be one of flashtp, cueq, e3nn"

        self.denominator = nn.Parameter(
            torch.FloatTensor([denominator]), requires_grad=train_denominator
        )

        instructions = []
        irreps_mid = []
        weight_numel = 0
        for i, (mul_x, ir_x) in enumerate(irreps_x):
            for j, (_, ir_filter) in enumerate(irreps_filter):
                for ir_out in ir_x * ir_filter:
                    if ir_out in irreps_out:  # here we drop l > lmax
                        k = len(irreps_mid)
                        weight_numel += mul_x * 1  # path shape
                        irreps_mid.append((mul_x, ir_out))
                        instructions.append((i, j, k, "uvu", True))

        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()  # type: ignore
        instructions = [
            (i_in1, i_in2, p[i_out], mode, train)
            for i_in1, i_in2, i_out, mode, train in instructions
        ]

        # From v0.11.x, to compatible with cuEquivariance
        self._instructions_before_sort = instructions
        instructions = sorted(instructions, key=lambda x: x[2])

        self.convolution_kwargs = dict(
            irreps_in1=irreps_x,
            irreps_in2=irreps_filter,
            irreps_out=irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        self.weight_nn_kwargs = dict(
            hs=weight_layer_input_to_hidden + [weight_numel], act=weight_layer_act
        )

        self.conv_type = conv_type.lower()

        if self.conv_type == "flashtp" and FLASHTP_AVAILABLE:
            self.convolution = uvu_TP(
                irreps_in1=irreps_x,
                irreps_in2=irreps_filter,
                irreps_out=irreps_mid,
                instructions=instructions,
                dtype=torch.float32,
                use_lammps=False,
            )

        elif self.conv_type == "cueq":
            self.transpose_in = cuet.TransposeIrrepsLayout(
                irreps_x, source=cue.mul_ir, target=cue.ir_mul
            )
            self.transpose_out = cuet.TransposeIrrepsLayout(
                irreps_mid, source=cue.ir_mul, target=cue.mul_ir
            )

            try:
                # uniform channel size for irreps_x
                self.convolution = with_cueq_conv_fusion(
                    cuet.SegmentedPolynomial(
                        cue.descriptors.channelwise_tensor_product(
                            cue.Irreps("O3", irreps_x),
                            cue.Irreps("O3", irreps_filter),
                            cue.Irreps("O3", irreps_mid),
                        )
                        .flatten_coefficient_modes()
                        .squeeze_modes()
                        .polynomial,
                        math_dtype=torch.get_default_dtype(),
                        method="uniform_1d",
                    )
                )
            except:
                # non-uniform channel size for irreps_x
                poly = cue.descriptors.channelwise_tensor_product(
                    cue.Irreps("O3", irreps_x),
                    cue.Irreps("O3", irreps_filter),
                    cue.Irreps("O3", irreps_mid),
                )
                poly = poly.flatten_modes("ijk").squeeze_modes("v")
                poly = poly.apply_fn(lambda op, d: (op, d.split_mode("u", 32)))
                self.convolution = with_cueq_conv_fusion(
                    cuet.SegmentedPolynomial(
                        poly.polynomial,
                        math_dtype=torch.get_default_dtype(),
                        method="uniform_1d",
                    )
                )
        else:
            self.convolution = TensorProduct(
                irreps_in1=irreps_x,
                irreps_in2=irreps_filter,
                irreps_out=irreps_mid,
                instructions=instructions,
                shared_weights=False,
                internal_weights=False,
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

        if self.conv_type == "flashtp":
            edge_src = edge_src.to(torch.int32)
            edge_dst = edge_dst.to(torch.int32)
            out = self.convolution(x, edge_attr, weight, edge_src, edge_dst)[:nlocal]
            out = out.div(self.denominator)
        elif self.conv_type == "e3nn":
            message = self.convolution(x[edge_src], edge_attr, weight)
            out = message_gather(x, edge_dst, message)[:nlocal]
            out = out.div(self.denominator)
        else:
            x = self.transpose_in(x)
            out = self.convolution(x, edge_attr, weight, edge_index)[:nlocal]
            out = out.div(self.denominator)
            out = self.transpose_out(out)

        return out
