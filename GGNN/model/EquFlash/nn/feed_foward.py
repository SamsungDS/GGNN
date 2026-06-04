from typing import Callable, List, Optional, Dict

import torch
import torch.nn as nn
from e3nn.nn import FullyConnectedNet, Gate, NormActivation
from e3nn.o3 import Irreps, Linear
from e3nn.util.jit import compile_mode


import torch.nn.functional as F
import cuequivariance as cue
import cuequivariance_torch as cuet

import torch


class myact(torch.nn.Module):
    def __init__(self, irreps):
        super().__init__()
        self.irreps = irreps
        gated_dim = 0
        for mul, ir in irreps:
            if ir.l != 0 or ir.p != 1:
                gated_dim += mul
        tp_cue = cue.descriptors.elementwise_tensor_product(
            cue.Irreps("O3", self.irreps), cue.Irreps("O3", self.irreps), [cue.O3(0, 1)]
        )
        tp_out_dim = tp_cue.outputs[0].dim
        self.use_tp = gated_dim > 0

        if self.use_tp:
            self.tp = cuet.EquivariantTensorProduct(tp_cue)
            self.linear = torch.nn.Linear(tp_out_dim, gated_dim, bias=False)

        self.act = torch.nn.functional.silu

    def forward(self, x):
        if self.use_tp:
            gates = self.tp(x, x)
            gates = self.act(self.linear(gates))
            sidx = 0
            sidx_gate = 0
            res = []
            for mul, ir in self.irreps:
                eidx = sidx + mul * ir.dim
                if ir.l == 0 and ir.p == 1:
                    res.append(self.act(x[:, sidx:eidx]))
                else:
                    eidx_gate = sidx_gate + mul
                    res.append(
                        (
                            gates[:, sidx_gate:eidx_gate].unsqueeze(-1)
                            * x[:, sidx:eidx].reshape(-1, mul, ir.dim)
                        ).reshape(-1, mul * ir.dim)
                    )
                    sidx_gate = eidx_gate
                sidx = eidx
            return torch.cat(res, dim=-1)
        else:
            return self.act(x)


class EquivariantLayerNormFast(nn.Module):
    def __init__(
        self,
        irreps,
        eps=1e-5,
        affine=True,
        normalization="component",
    ):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine

        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, x):
        """
        Use torch layer norm for scalar features.
        """
        node_input = x
        dim = node_input.shape[-1]

        fields = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw : (iw + mul)]
                bias = self.affine_bias[ib : (ib + mul)]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(
                    field.reshape(-1, mul * d)
                )  # [batch * sample, mul * repr]
                continue

            # For non-scalar features, use RMS value for std
            field = field.reshape(-1, mul, d)  # [batch * sample, mul, repr]

            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)  # [batch * sample, mul]
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)  # [batch * sample, mul]
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            # print(ir,field_norm.min(),end='\t')
            field_norm = 1.0 / ((field_norm + self.eps).sqrt())  # [batch * sample, mul]

            if self.affine:
                weight = self.affine_weight[None, iw : (iw + mul)]  # [1, mul]
                iw += mul
                field_norm = field_norm * weight  # [batch * sample, mul]
            field = field * field_norm.reshape(
                -1, mul, 1
            )  # [batch * sample, mul, repr]

            fields.append(field.reshape(-1, mul * d))  # [batch * sample, mul * repr]

        assert ix == dim

        output = torch.cat(fields, dim=-1)

        return output


class SeparableLayerNorm(nn.Module):
    def __init__(
        self,
        irreps,
        eps=1e-5,
        affine=True,
        normalization="component",
    ):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine
        num_scalar = sum(mul for mul, ir in self.irreps if ir.l == 0 and ir.p == 1)
        num_features = self.irreps.num_irreps

        if affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, x):
        """
        Use torch layer norm for scalar features.
        """
        node_input = x
        dim = node_input.shape[-1]

        fields = []
        norms = []
        ix = 0
        iw = 0
        ib = 0

        for (
            mul,
            ir,
        ) in (
            self.irreps
        ):  # mul is the multiplicity (number of copies) of some irrep type (ir)
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)  # [natom, mul*(2l+1)]
            ix += mul * d

            if ir.l == 0 and ir.p == 1:
                field = field - field.mean(dim=1, keepdim=True)  # [natom, mul]
                norm = (field * field).mean(axis=1, keepdim=True)
                fields.append(field)
                norms.append(norm)
            else:
                norm = (field * field).mean(axis=1, keepdim=True)
                fields.append(field)
                norms.append(norm)

        norms = torch.cat(norms, axis=-1)
        norms = (norms.mean(axis=1, keepdim=True) + self.eps).pow(-0.5)
        res = []
        for feat, (mul, ir) in zip(fields, self.irreps):
            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw : (iw + mul)]
                bias = self.affine_bias[ib : (ib + mul)]
                iw += mul
                ib += mul
                res.append(feat * norms * weight + bias)
            else:
                weight = self.affine_weight[None, iw : (iw + mul)]
                iw += mul

                feat_reshaped = feat.reshape(-1, mul, ir.dim)
                res.append(
                    (feat_reshaped * (norms * weight).unsqueeze(-1)).reshape(
                        -1, mul * ir.dim
                    )
                )

        x = torch.cat(res, dim=-1)
        return x

        return output


@compile_mode("script")
class FeedForward(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(
        self,
        irreps_in: Irreps,
        act_scalar_dict: Dict[int, Callable],
        act_gate_dict: Dict[int, Callable],
    ):
        super().__init__()

        parity_mapper = {"e": 1, "o": -1}
        act_scalar_dict = {parity_mapper[k]: v for k, v in act_scalar_dict.items()}
        act_gate_dict = {parity_mapper[k]: v for k, v in act_gate_dict.items()}

        irreps_gated_elem = []
        irreps_scalars_elem = []
        # non scalar irreps > gated / scalar irreps > scalars
        for mul, irreps in irreps_in:
            if irreps.l > 0:
                irreps_gated_elem.append((mul, irreps))
            else:
                irreps_scalars_elem.append((mul, irreps))
        irreps_scalars = Irreps(irreps_scalars_elem)
        irreps_gated = Irreps(irreps_gated_elem)

        irreps_gates_parity = 1 if "0e" in irreps_scalars else -1
        irreps_gates = Irreps(
            [(mul, (0, irreps_gates_parity)) for mul, _ in irreps_gated]
        )

        act_scalars = [act_scalar_dict[p] for _, (_, p) in irreps_scalars]
        act_gates = [act_gate_dict[p] for _, (_, p) in irreps_gates]
        gate = Gate(irreps_scalars, act_scalars, irreps_gates, act_gates, irreps_gated)
        irreps_gate = gate.irreps_in
        self.linear_1 = Linear(irreps_in, irreps_gate)
        self.act = gate
        self.linear_2 = Linear(irreps_in, irreps_in)

    def forward(self, x):
        x_res = x
        # x = self.norm(x)
        x = self.linear_1(x)
        x = self.act(x)
        x = self.linear_2(x)

        return x + x_res
