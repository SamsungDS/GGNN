import os
from copy import deepcopy
from typing import Callable, List, Optional, Tuple, Dict, Type
import torch

from e3nn.o3 import Irreps, FullTensorProduct
import os

import cuequivariance as cue
import cuequivariance_torch as cuet

from .convolution import FullConv, EfficientConv
from .normalization import EquivariantRMSNormArraySphericalHarmonicsV2
from .linear import Linear
from .nonlinears import Gate, BiLinearGate, SymmetricContraction
from .skip import SkipId
from .util import mul_irreps

NONLINEAR_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "gate": Gate,
    "bilinear_gate": BiLinearGate,
    "symmetric_contraction": SymmetricContraction,
}

CONV_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "full": FullConv,
    "efficient": EfficientConv,
}

SKIP_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "identity": SkipId,
    "linear": Linear,
}
LINEAR_REGISTRY: Dict[str, Type[torch.nn.Module]] = {
    "identity": torch.nn.Identity,
    "linear": Linear,
}


class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        bias_in_linear: bool = False,
        sc_type: str = "linear",
        conv_type="full",
        conv_kwargs: Dict = None,
        nonlinear_type: str = None,
        nonlinear_kwargs: Dict = {},
    ):
        super().__init__()

        # metaformer arch
        self.norm1 = EquivariantRMSNormArraySphericalHarmonicsV2(irreps_x)
        skip_cls = SKIP_REGISTRY[sc_type]
        self.skip_con = skip_cls(irreps_x, irreps_out)

        self.self_interaction_1 = Linear(
            irreps_x,
            irreps_x,
            biases=bias_in_linear,
        )
        conv_cls = CONV_REGISTRY[conv_type]
        conv_kwargs = deepcopy(conv_kwargs)
        self.conv = conv_cls(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out,
            **conv_kwargs,
        )
        irreps_conv_out = self.conv.irreps_out
        nonlinear_cls = NONLINEAR_REGISTRY[nonlinear_type]
        nonlinear_kwargs = deepcopy(nonlinear_kwargs)
        if "irreps_scale" in nonlinear_kwargs:
            scale = nonlinear_kwargs.pop("irreps_scale")
            nonlinear_kwargs["irreps"] = mul_irreps(scale, irreps_out)
        if "irreps" not in nonlinear_kwargs:
            nonlinear_kwargs["irreps"] = irreps_out

        self_interaction_type = nonlinear_kwargs.pop("after_nonlinear", "linear")

        self.nonlinear = nonlinear_cls(**nonlinear_kwargs)
        self.self_interaction_2 = Linear(
            irreps_conv_out, self.nonlinear.irreps_in.simplify()
        )

        if self_interaction_type == "identity" and (
            self.nonlinear.irreps_out.simplify() != irreps_out
        ):
            ValueError(f"identity requires {self.nonlinear.irreps_out} == {irreps_out}")
        self_interaction_cls = LINEAR_REGISTRY[self_interaction_type]
        self.self_interaction_3 = self_interaction_cls(
            self.nonlinear.irreps_out, irreps_out
        )

    def forward(self, data):
        x, edge_index, edge_embedding, edge_attr, lmp_data = (
            data["x"],
            data["edge_index"],
            data["edge_embedding"],
            data["edge_attr"],
            data.get("lmp_data", None),
        )
        xres = self.skip_con(x)
        x = self.norm1(x)
        x = self.self_interaction_1(x)
        out = self.conv(x, edge_index, edge_embedding, edge_attr, lmp_data)
        x = self.self_interaction_2(out)
        x = self.nonlinear(x)
        x = self.self_interaction_3(x)
        x = x + xres
        data["x"] = x

        return data


class NequIPInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        irreps_filter: Irreps,
        irreps_out: Irreps,
        bias_in_linear: bool = False,
        sc_type: str = "linear",
        conv_type="full",
        conv_kwargs: Dict = None,
        nonlinear_type: str = None,
        nonlinear_kwargs: Dict = {},
    ):
        super().__init__()

        # metaformer arch

        self.self_interaction_1 = Linear(
            irreps_x,
            irreps_x,
            biases=bias_in_linear,
        )
        conv_cls = CONV_REGISTRY[conv_type]
        conv_kwargs = deepcopy(conv_kwargs)
        self.conv = conv_cls(
            irreps_x=irreps_x,
            irreps_filter=irreps_filter,
            irreps_out=irreps_out,
            **conv_kwargs,
        )
        irreps_conv_out = self.conv.irreps_out
        nonlinear_cls = NONLINEAR_REGISTRY[nonlinear_type]
        nonlinear_kwargs = deepcopy(nonlinear_kwargs)
        if "irreps_scale" in nonlinear_kwargs:
            scale = nonlinear_kwargs.pop("irreps_scale")
            nonlinear_kwargs["irreps"] = mul_irreps(scale, irreps_out)
        if "irreps" not in nonlinear_kwargs:
            nonlinear_kwargs["irreps"] = irreps_out

        self_interaction_type = nonlinear_kwargs.pop("after_nonlinear", "linear")

        self.nonlinear = nonlinear_cls(**nonlinear_kwargs)
        skip_cls = SKIP_REGISTRY[sc_type]
        self.skip_con = skip_cls(irreps_x, self.nonlinear.irreps_in.simplify())
        self.self_interaction_2 = Linear(
            irreps_conv_out, self.nonlinear.irreps_in.simplify()
        )

    def forward(self, data):
        x, edge_index, edge_embedding, edge_attr, lmp_data = (
            data["x"],
            data["edge_index"],
            data["edge_embedding"],
            data["edge_attr"],
            data.get("lmp_data", None),
        )
        xres = self.skip_con(x)
        x = self.self_interaction_1(x)
        out = self.conv(x, edge_index, edge_embedding, edge_attr, lmp_data)
        x = self.self_interaction_2(out)
        x = x + xres
        x = self.nonlinear(x)
        data["x"] = x

        return data


class FeedForwardBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        sc_type: str = "identity",
        nonlinear_type: str = None,
        nonlinear_kwargs: Dict = {},
    ):
        super().__init__()

        # metaformer arch
        self.norm2 = EquivariantRMSNormArraySphericalHarmonicsV2(irreps_in)
        skip_cls = SKIP_REGISTRY[sc_type]
        self.skip = skip_cls(irreps_in, irreps_out)
        nonlinear_cls = NONLINEAR_REGISTRY[nonlinear_type]
        nonlinear_kwargs = deepcopy(nonlinear_kwargs)
        if "irreps_scale" in nonlinear_kwargs:
            scale = nonlinear_kwargs.pop("irreps_scale")
            nonlinear_kwargs["irreps"] = mul_irreps(scale, irreps_out)
        if "irreps" not in nonlinear_kwargs:
            nonlinear_kwargs["irreps"] = irreps_out
        self.nonlinear = nonlinear_cls(**nonlinear_kwargs)
        self.linear_1 = Linear(irreps_in, self.nonlinear.irreps_in.simplify())
        self.linear_2 = Linear(self.nonlinear.irreps_out, irreps_out)

    def forward(self, data):
        x = data["x"]
        x_res = self.skip(x)
        x = self.norm2(x)
        x = self.linear_1(x)
        x = self.nonlinear(x)
        x = self.linear_2(x)
        x = x_res + x
        data["x"] = x

        return data
