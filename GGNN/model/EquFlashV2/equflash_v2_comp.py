#############
from fairchem.core.common.registry import registry
from fairchem.core.common.utils import conditional_grad

import warnings
from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn
import numpy as np
from e3nn.util.jit import compile_mode

from torch_geometric.data import Batch
from fairchem.core.common.utils import get_pbc_distances
from fairchem.core.common import distutils
from GGNN.common.utils import generate_graph
import os
import yaml
from ase.data import atomic_numbers
from fairchem.core.common.utils import conditional_grad
from .equflash_v2 import EquFlashV2
from e3nn.o3 import Linear
from .nn.normalization import EquivariantRMSNormArraySphericalHarmonicsV2
from .nn.node_embedding import OneHotNodeEmbedding
from .nn.edge_embedding import EdgeEmbedding

from e3nn import set_optimization_defaults

import torch
from typing import Dict, Sequence, List, Optional, Any, Final


def _list_to_dict(
    keys: Sequence[str], args: List[torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return {key: arg for key, arg in zip(keys, args)}


def _list_from_dict(
    keys: Sequence[str], data: Dict[str, torch.Tensor]
) -> List[torch.Tensor]:
    return [data[key] for key in keys]


class ListInputOutputWrapper(torch.nn.Module):
    """
    Wraps a ``torch.nn.Module`` that takes and returns ``Dict[str, torch.Tensor]`` to have it take and return ``Sequence[torch.Tensor]`` for specified input and output fields.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
    ):
        super().__init__()
        self.model = model
        self.input_keys = list(input_keys)
        self.output_keys = list(output_keys)

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        inputs = _list_to_dict(self.input_keys, args)
        outputs = self.model(inputs)
        return _list_from_dict(self.output_keys, outputs)


class DictInputOutputWrapper(torch.nn.Module):
    """
    Wraps a model that takes and returns ``Sequence[torch.Tensor]`` to have it take and return ``Dict[str, torch.Tensor]`` for specified input and output fields (i.e. the opposite of ``ListInputOutputWrapper``).
    """

    def __init__(self, model, input_keys: List[str], output_keys: List[str]):
        super().__init__()
        self.model = model
        self.input_keys = input_keys
        self.output_keys = output_keys

    def forward(self, data):
        inputs = _list_from_dict(self.input_keys, data)
        with torch.inference_mode():
            outputs = self.model(inputs)
        return _list_to_dict(self.output_keys, outputs)

class ListInputOutputStateDictWrapper(ListInputOutputWrapper):
    """Like ``ListInputOutputWrapper``, but also updates the model with state dict entries before each ``forward``."""

    def __init__(
        self,
        model: torch.nn.Module,
        input_keys: Sequence[str],
        output_keys: Sequence[str],
        state_dict_keys: Sequence[str],
    ):
        super().__init__(model, input_keys, output_keys)
        self.state_dict_keys = state_dict_keys

    def forward(self, *args: torch.Tensor) -> List[torch.Tensor]:
        # won't check that `args` is of the correct length
        input_dict = _list_to_dict(self.input_keys, args[: len(self.input_keys)])
        state_dict = _list_to_dict(self.state_dict_keys, args[len(self.input_keys) :])
        # have to do it this way and not using state_dict directly for autograd reasons
        # the `.data` part is important
        # with torch.no_grad():
        #     for name, param in self.model.named_parameters():
        #         param.data.copy_(state_dict[name])
        #     for name, buffer in self.model.named_buffers():
        #         buffer.data.copy_(state_dict[name])
        output_dict = self.model(input_dict)
        return _list_from_dict(self.output_keys, output_dict)


import math
import torch
from torch.fx.experimental.proxy_tensor import make_fx
import contextlib
import difflib
import uuid
import os
from typing import List


@contextlib.contextmanager
def fx_duck_shape(enabled: bool):
    """
    For our use of `make_fx` to unfold the autograd graph, we must set the following `use_duck_shape` parameter to `False` (it's `True` by default).
    It forces dynamic batch dims (num_frames, num_atoms, num_edges) to shape specialize if the batch dim is the same as that of a static dim.
    E.g. in training, shape specialization would occur if a weight tensor has a dimension with shape (16,) and we use a batch size of 16 (so the dynamic batch dim `num_frames` is 16) because of the duck shaping.
    """
    # save previous state
    init_duck_shape = torch.fx.experimental._config.use_duck_shape
    # set mode variables
    torch.fx.experimental._config.use_duck_shape = enabled
    try:
        yield
    finally:
        # restore state
        torch.fx.experimental._config.use_duck_shape = init_duck_shape


def _nequip_make_fx(model, inputs):
    with fx_duck_shape(False):
        return make_fx(
            model,
            tracing_mode="symbolic",
            _allow_non_fake_inputs=True,
            _error_on_data_dependent_ops=True,
        )(*[i.clone() for i in inputs])


@registry.register_model("equflashv2_comp")
class CompiledEquFlashV2(nn.Module):
    """
    Wrapper of SevenNet model

    Args:
        modules: OrderedDict of nn.Modules
        cutoff: not used internally, but makes sense to have
        type_map: atomic_numbers => onehot index (see nn/node_embedding.py)
        eval_type_map: perform index mapping using type_map defaults to True
        data_key_atomic_numbers: used when eval_type_map is True
        data_key_node_feature: used when eval_type_map is True
        data_key_grad: if given, sets its requires grad True before pred
    """

    def __init__(self, config: dict, dataset=None, device=None, parallel=False):
        super().__init__()
        warnings.warn(
            "CompiledEquflash is not stable yet. If it fails during forward, use equflashv2 instead of equflashv2_comp"
        )
        # processing_dataset(config, dataset)

        self.config = config
        self.regress_forces = self.config.get("regress_forces", True)
        self.regress_stress = self.config.get("regress_stress", True)

        set_optimization_defaults(jit_script_fx=False)
        self.model = EquFlashV2(config)

        self.input_keys = [
            "atomic_numbers",
            "batch",
            "cell",
            "cell_offsets",
            "edge_index",
            "edge_vec",
            "natoms",
            "ptr",
        ]

        self.output_keys = [
            "energy",
            "forces",
            "stress",
        ]

        self.compiled_model = None

    def forward(self, batch):

        if self.compiled_model == None:

            weight_names = [n for n, _ in self.model.named_parameters()]
            buffer_names = [n for n, _ in self.model.named_buffers()]
            model_wrapper = ListInputOutputStateDictWrapper(
                model=self.model,
                input_keys=self.input_keys,
                output_keys=self.output_keys,
                state_dict_keys=weight_names + buffer_names,
            )
            test_data_list = [batch[key] for key in self.input_keys]
            weight_dict = dict(self.model.named_parameters())
            weights = [weight_dict[name] for name in weight_names]
            buffer_dict = dict(self.model.named_buffers())
            buffers = [buffer_dict[name] for name in buffer_names]
            extra_input = weights + buffers

            fx_model = _nequip_make_fx(model_wrapper, test_data_list + extra_input)
            del weights, buffers

            print("compiling!")
            self.compiled_model = torch.compile(fx_model, dynamic=True, fullgraph=True)

            # # weights, buffers
            # data_list = _list_from_dict(self.input_keys, batch)
            # print("first compile forward")
            # out_list = self.compiled_model(*(data_list + weights + buffers))
            # out_dict = _list_to_dict(self.output_keys, out_list)
        out_dict = self._compiled_forward(batch)
        out = {}
        out["energy"] = out_dict["energy"]
        if self.regress_forces:
            out["forces"] = out_dict["forces"]
        if self.regress_stress:
            out["stress"] = out_dict["stress"]
        return out

    def _compiled_forward(self, batch):
        weight_names = [n for n, _ in self.model.named_parameters()]
        buffer_names = [n for n, _ in self.model.named_buffers()]
        weight_dict = dict(self.model.named_parameters())
        weights = [weight_dict[name] for name in weight_names]
        buffer_dict = dict(self.model.named_buffers())
        buffers = [buffer_dict[name] for name in buffer_names]
        data_list = _list_from_dict(self.input_keys, batch)
        out_list = self.compiled_model(*(data_list + weights + buffers))
        out_dict = _list_to_dict(self.output_keys, out_list)
        return out_dict

    def _atomic_numbers_to_onehot(self, atomic_numbers: torch.Tensor):
        assert atomic_numbers.dtype == torch.int64
        return torch.index_select(
            input=self.z_to_onehot_tensor, dim=0, index=atomic_numbers
        )
    

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
                    OneHotNodeEmbedding
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




def convert_compiled_to_original_sevennet(ckpt):
    if "compiled_sevennet" == ckpt["config"]["model"]["name"]:
        ckpt["config"]["model"]["name"] = "sevennet"
        new_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            k1 = k.replace("model.", "")
            new_dict[k1] = v
        ckpt["state_dict"] = new_dict

    return ckpt
