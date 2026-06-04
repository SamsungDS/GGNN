import torch
from typing import Union
import itertools
import torch.nn as nn
from e3nn.o3 import Irreps
from e3nn.nn._activation import normalize2mom

import cuequivariance as cue
import cuequivariance_torch as cuet
from cuequivariance.group_theory.irreps_array.irrep_utils import into_list_of_irrep


def cue_gate_descriptor(irreps_gated, irreps_gate):
    d = cue.SegmentedTensorProduct.from_subscripts("iu,ju,ku+ijk")
    for mul, ir in irreps_gate:
        d.add_segment(
            0,
            (
                ir.dim,
                mul,
            ),
        )
    for mul, ir in irreps_gated:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps_gated:
        d.add_segment(2, (ir.dim, mul))
    for i, (mul, ir) in enumerate(irreps_gated):
        d.add_path(
            i,
            i,
            i,
            c=cue.clebsch_gordan(cue.O3(0, 1), ir, ir).squeeze(0),
            dims={"u": mul},
        )
    irreps_gated = cue.Irreps("O3", irreps_gated)
    irreps_gate = cue.Irreps("O3", irreps_gate)
    return cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(irreps_gate, cue.ir_mul),
            cue.IrrepsAndLayout(irreps_gated, cue.ir_mul),
        ],
        [cue.IrrepsAndLayout(irreps_gated, cue.ir_mul)],
        cue.SegmentedPolynomial.eval_last_operand(d),
    )


def uuu_tensor_product_with_weight(
    irreps1: cue.Irreps,
    irreps2: cue.Irreps,
    irreps3_filter=None,
    simplify_irreps3: bool = False,
) -> cue.EquivariantPolynomial:
    """ """
    G = irreps1.irrep_class

    if irreps3_filter is not None:
        irreps3_filter = into_list_of_irrep(G, irreps3_filter)

    d = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")

    for mul, ir in irreps1:
        d.add_segment(1, (ir.dim, mul))
    for mul, ir in irreps2:
        d.add_segment(2, (ir.dim, mul))

    irreps3 = []

    for (i1, (mul1, ir1)), (i2, (mul2, ir2)) in itertools.product(
        enumerate(irreps1), enumerate(irreps2)
    ):
        if ir1.dim < ir2.dim:
            continue
        for ir3 in ir1 * ir2:
            if irreps3_filter is not None and ir3 not in irreps3_filter:
                continue

            # for loop over the different solutions of the Clebsch-Gordan decomposition
            for cg in cue.clebsch_gordan(ir1, ir2, ir3):
                d.add_path(None, i1, i2, None, c=cg, dims={"u": mul1})
                irreps3.append((mul1, ir3))

    irreps3 = cue.Irreps(G, irreps3)
    irreps3, perm, inv = irreps3.sort()
    # d = d.permute_segments(0, inv)
    d = d.permute_segments(3, inv)

    d = d.normalize_paths_for_operand(-1)

    if simplify_irreps3:
        # We need to flatten the coefficients to do this simplification
        d = d.flatten_coefficient_modes()

        # Sort output segments to regroup them by irreps
        segments = [(ir, k) for _mul, ir in irreps3 for k in range(ir.dim)]
        segments = [(ir, k, sid) for sid, (ir, k) in enumerate(segments)]
        segments = sorted(segments)
        d = d.permute_segments(3, [sid for _, _, sid in segments])
        irreps3 = irreps3.simplify()
    return (
        cue.EquivariantPolynomial(
            [
                cue.IrrepsAndLayout(
                    irreps1.new_scalars(d.operands[0].size), cue.ir_mul
                ),
                cue.IrrepsAndLayout(irreps1, cue.ir_mul),
                cue.IrrepsAndLayout(irreps2, cue.ir_mul),
            ],
            [cue.IrrepsAndLayout(irreps3, cue.ir_mul)],
            cue.SegmentedPolynomial.eval_last_operand(d),
        ),
        irreps3,
    )


class Identity(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x):
        return x


class Gate(nn.Module):
    def __init__(
        self,
        irreps: Union[Irreps, str],
        act_odd=torch.nn.functional.tanh,
        act_even=torch.nn.functional.silu,
    ):
        super().__init__()
        irreps = Irreps(irreps)
        irreps_gated = []
        irreps_scalars_even = []
        irreps_scalars_odd = []
        # non scalar irreps > gated / scalar irreps > scalars
        for mul, ir in irreps:
            if ir.l > 0:
                irreps_gated.append((mul, ir))
            elif ir.p == 1:
                irreps_scalars_even.append((mul, ir))
            else:
                irreps_scalars_odd.append((mul, ir))
        self.irreps_scalars_odd = Irreps(irreps_scalars_odd)
        self.irreps_scalars_even = Irreps(irreps_scalars_even)
        self.irreps_gated = Irreps(irreps_gated)
        self.irreps_gates = Irreps([(mul, (0, 1)) for mul, _ in self.irreps_gated])
        d = cue_gate_descriptor(self.irreps_gated, self.irreps_gates)
        self.d = d
        if d.polynomial.outputs[0].num_segments == 0:
            self.gate = None
        else:
            try:
                self.gate = cuet.SegmentedPolynomial(
                    d.polynomial,
                    math_dtype=torch.get_default_dtype(),
                    method="uniform_1d",
                )
            except:
                self.gate = cuet.SegmentedPolynomial(
                    d.flatten_modes("ijk").split_mode("u", 32).polynomial,
                    math_dtype=torch.get_default_dtype(),
                    method="uniform_1d",
                )

        self.act_even = normalize2mom(act_even)
        self.act_odd = normalize2mom(act_odd)
        self.irreps_in = (
            self.irreps_scalars_odd
            + self.irreps_scalars_even
            + self.irreps_gates
            + self.irreps_gated
        ).simplify()
        self.irreps_out = irreps

    def forward(self, x):
        odd_len = self.irreps_scalars_odd.dim
        even_len = self.irreps_scalars_even.dim + self.irreps_gates.dim
        x_odd_act = x[:, :odd_len]
        x_even_act = x[:, odd_len : odd_len + even_len]
        x_gated = x[:, odd_len + even_len :]
        x_even = self.act_even(x_even_act)
        x_odd = self.act_odd(x_odd_act)
        if self.gate == None:
            x = torch.cat([x_odd, x_even[:, : self.irreps_scalars_even.dim]], axis=1)
        else:
            x_gated = self.gate([x_even[:, self.irreps_scalars_even.dim :], x_gated])[0]
            x = torch.cat(
                [x_odd, x_even[:, : self.irreps_scalars_even.dim], x_gated], axis=1
            )
        return x


class BiLinearGate(nn.Module):
    def __init__(
        self,
        irreps: Union[Irreps, str],
    ):
        super().__init__()
        irreps = Irreps(irreps)
        for mul, ir in irreps:
            if ir.l == 0 and ir.p == -1:
                raise NotImplementedError("0o feature not supported yet")
        self.irreps_target = irreps
        d, irreps_mid_out = uuu_tensor_product_with_weight(
            cue.Irreps("O3", irreps[1:]),
            cue.Irreps("O3", irreps),
            cue.Irreps("O3", irreps),
            True,
        )
        self.d = d
        self.irreps_gate = Irreps(f"{d.operands[0].dim}x0e")
        self.tpweightdim = Irreps(f"{d.operands[0].dim}x0e").dim
        self.dim_act = irreps[0].dim
        self.irreps_act = irreps[0]
        irreps_mid = (
            Irreps(f"{d.operands[0].dim}x0e")
            + irreps[:1]
            + (irreps[1:] + irreps[1:]).sort()[0].simplify()
        )
        if d.operands[3].dim > 0:
            self.tp = cuet.SegmentedPolynomial(
                d.flatten_coefficient_modes().squeeze_modes().polynomial,
                math_dtype=torch.get_default_dtype(),
                method="uniform_1d",
            )
        else:
            self.tp = None

        self.irreps_in = irreps_mid
        self.irreps_out = (
            (Irreps(irreps[:1]) + Irreps(irreps_mid_out.__repr__()))
            .sort()[0]
            .simplify()
        )

    def __repr__(self):
        return f"BiLinearGate({self.irreps_act} + {self.irreps_gate} x {self.d.inputs[1]} x {self.d.inputs[2]}->{self.irreps_out})"

    def forward(self, x):
        x_scalar = x[:, : self.irreps_gate.dim]
        x_act = x[:, self.irreps_gate.dim : self.irreps_gate.dim + self.dim_act]
        x_scalar = torch.nn.functional.silu(x_scalar)
        x_act = torch.nn.functional.silu(x_act)
        x_in = x[:, self.irreps_gate.dim + self.dim_act :]
        B = x.shape[0]
        out_dim = self.irreps_target[1:].dim
        x1 = x.new_empty(B, out_dim)
        x2 = x.new_empty(B, self.dim_act + out_dim)

        o1 = 0
        o2 = self.dim_act
        for sl, (mul, ir) in zip(self.irreps_in[2:].slices(), self.irreps_in[2:]):
            d = ir.dim
            assert mul % 2 == 0
            h = mul // 2

            # (B, mul*d) -> (B, mul, d)
            xt = x_in[:, sl].view(B, d, mul)

            a, b = xt[:, :, :h], xt[:, :, h:]  # half multiplicities
            n = h * d

            x1[:, o1 : o1 + n] = a.reshape(B, n)
            x2[:, o2 : o2 + n] = b.reshape(B, n)
            o1 += n
            o2 += n

        x2[:, : self.dim_act] = 1
        if self.tp is not None:
            x = self.tp([x_scalar, x1, x2])[0]
        else:
            x = x_scalar
        x = torch.cat([x_act, x], dim=-1)
        return x


class SymmetricContraction(nn.Module):
    """ """

    def __init__(
        self, irreps: Union[Irreps, str], contraction_degree=3, specieswise=True
    ):
        super().__init__()
        irreps = Irreps(irreps)
        layout = cue.ir_mul
        self.irreps_in = irreps
        self.irreps_out = irreps
        self.sym_contraction = cuet.SymmetricContraction(
            cue.Irreps("O3", irreps),
            cue.Irreps("O3", irreps),
            layout_in=layout,
            layout_out=layout,
            contraction_degree=contraction_degree,
            num_elements=1,
            original_mace=False,
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
        )

    def forward(self, x, y=None):
        y = torch.zeros(x.shape[0], device=x.device, dtype=torch.int)
        z = self.sym_contraction(x, y)
        return z
