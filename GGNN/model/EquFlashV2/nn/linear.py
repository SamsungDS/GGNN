from typing import NamedTuple

from opt_einsum_fx import optimize_einsums_full
import torch
from torch import fx

from e3nn.o3._irreps import Irreps
from e3nn.o3._tensor_product._codegen import _sum_tensors
from e3nn.o3._linear import Instruction
from e3nn.util import prod
from e3nn.util.codegen import CodeGenMixin


class Linear(CodeGenMixin, torch.nn.Module):
    r"""Modified e3nn.o3.linear module to support muon optimizer and ir_mul data layout"""

    weight_numel: int
    internal_weights: bool
    shared_weights: bool

    def __init__(self, irreps_in: Irreps, irreps_out: Irreps, biases=False) -> None:
        super().__init__()
        if biases == True:
            raise NotImplementedError("bias in linear layer not implemented yet")
        irreps_in = Irreps(irreps_in)
        irreps_out = Irreps(irreps_out)

        # By default, make all possible connections
        instructions = [
            (i_in, i_out)
            for i_in, (_, ir_in) in enumerate(irreps_in)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_in == ir_out
        ]

        instructions = [
            Instruction(
                i_in=i_in,
                i_out=i_out,
                path_shape=(irreps_in[i_in].mul, irreps_out[i_out].mul),
                path_weight=1,
            )
            for i_in, i_out in instructions
        ]

        def alpha(ins) -> float:
            x = sum(irreps_in[i.i_in].mul for i in instructions if i.i_out == ins.i_out)
            return 1.0 if x == 0 else x

        instructions = [
            Instruction(
                i_in=ins.i_in,
                i_out=ins.i_out,
                path_shape=ins.path_shape,
                path_weight=alpha(ins) ** (-0.5),
            )
            for ins in instructions
        ]

        for ins in instructions:
            if not ins.i_in < len(irreps_in):
                raise IndexError(f"{ins.i_in} is not a valid index for irreps_in")
            if not ins.i_out < len(irreps_out):
                raise IndexError(f"{ins.i_out} is not a valid index for irreps_out")
            if not (
                ins.i_in == -1 or irreps_in[ins.i_in].ir == irreps_out[ins.i_out].ir
            ):
                raise ValueError(
                    f"{ins.i_in} and {ins.i_out} do not have the same irrep"
                )

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.instructions = instructions

        # == Generate code ==
        graphmod, self.weight_numel = _codegen_linear(
            self.irreps_in,
            self.irreps_out,
            self.instructions,
        )
        self._codegen_register({"_compiled_main": graphmod})
        weight_og = torch.randn(self.weight_numel)
        si = 0
        ei = 0
        w_list = []
        for ins in self.instructions:
            ei = si + prod(ins.path_shape)
            w_list.append(
                torch.nn.Parameter(weight_og[si:ei].data.reshape(ins.path_shape))
            )
            si = ei
        self.weights = torch.nn.ParameterList(w_list)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"

    def forward(
        self,
        features,
    ):

        return self._compiled_main(features, *self.weights)


def _codegen_linear(
    irreps_in: Irreps,
    irreps_out: Irreps,
    instructions,
):
    graph_out = fx.Graph()
    tracer_out = fx.proxy.GraphAppendingTracer(graph_out)

    # = Function definitions =
    x = fx.Proxy(graph_out.placeholder("x", torch.Tensor), tracer_out)
    ws = [
        fx.Proxy(graph_out.placeholder(f"w{i}", torch.Tensor), tracer_out)
        for i, _ in enumerate(instructions)
    ]

    size = x.shape[:-1]
    outsize = size + (irreps_out.dim,)

    # = Short-circut for nothing to do =
    # We produce no code for empty instructions
    instructions = [ins for ins in instructions if 0 not in ins.path_shape]

    if len(instructions) == 0:
        out = x.new_zeros(outsize)

        graph_out.output(out.node, torch.Tensor)
        # Short circut
        # 0 is weight_numel
        return fx.GraphModule({}, graph_out, "linear_forward"), 0, 0

    x = x.reshape(-1, irreps_in.dim)
    batch_out = x.shape[0]

    weight_numel = sum(prod(ins.path_shape) for ins in instructions if ins.i_in != -1)

    # = extract individual input irreps =

    x_list = [
        x.narrow(-1, i.start, mul_ir.dim).reshape(
            batch_out, *(()), mul_ir.ir.dim, mul_ir.mul
        )
        for i, mul_ir in zip(irreps_in.slices(), irreps_in)
    ]

    z = ""

    flat_weight_index = 0
    flat_bias_index = 0

    out_list = []

    for ins, w in zip(instructions, ws):
        mul_ir_out = irreps_out[ins.i_out]
        mul_ir_in = irreps_in[ins.i_in]
        # Short-circut for empty irreps
        if mul_ir_in.dim == 0 or mul_ir_out.dim == 0:
            continue
        # Extract the weight from the flattened weight tensor
        path_nweight = prod(ins.path_shape)
        flat_weight_index += path_nweight

        import os

        if os.environ.get("LINEAR_MATMUL", False):
            ein_out = x_list[ins.i_in] @ w
        else:
            ein_out = torch.einsum(f"{z}uw,ziu->ziw", w, x_list[ins.i_in])
        ein_out = ins.path_weight * ein_out
        out_list += [ein_out.reshape(batch_out, *(()), mul_ir_out.dim)]

    # = Return the result =
    out = [
        _sum_tensors(
            [out for ins, out in zip(instructions, out_list) if ins.i_out == i_out],
            shape=(batch_out, *(()), mul_ir_out.dim),
            like=x,
        )
        for i_out, mul_ir_out in enumerate(irreps_out)
        if mul_ir_out.mul > 0
    ]
    if len(out) > 1:
        out = torch.cat(out, dim=-1)
    else:
        out = out[0]

    out = out.reshape(outsize)

    graph_out.output(out.node, torch.Tensor)

    # check graphs
    graph_out.lint()

    graphmod_out = fx.GraphModule({}, graph_out, "linear_forward")

    batchdim = 4
    w_list = []
    for ins in instructions:
        w_list.append(torch.zeros(ins.path_shape))
    example_inputs = (
        torch.zeros((batchdim, *(()), irreps_in.dim)),
        *w_list,
    )
    graphmod_out = optimize_einsums_full(graphmod_out, example_inputs)
    return graphmod_out, weight_numel
