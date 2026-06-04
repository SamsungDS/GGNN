"""
This file is derived from fairchem-core.
Original source: https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License. See LICENSE for details.

Modifications copyright (c) 2026 Samsung Electronics.
Licensed under CC BY-NC-SA 4.0. See LICENSE for details.
"""

from fairchem.core.common.relaxation.ase_utils import OCPCalculator
from ..trainer.trainer import Trainer

import copy
import logging
from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms
from ase.geometry import wrap_positions

from fairchem.core.common.registry import registry
from fairchem.core.common.utils import (
    load_config,
    setup_imports,
    setup_logging,
    update_config,
)
from fairchem.core.datasets import data_list_collater
from fairchem.core.models.model_registry import model_name_to_local_file
from fairchem.core.preprocessing import AtomsToGraphs
from collections import OrderedDict


def convert_compiled_ckpt(ckpt):
    if ckpt["config"]["model"]["name"].startswith("compiled_"):
        ckpt["config"]["model"]["name"] = ckpt["config"]["model"]["name"].replace(
            "compiled_", ""
        )
        new_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            if k.startswith("module.model") and "z_to_onehot_tensor" not in k:
                new_dict[k.replace(".model.", ".")] = v
        ckpt["state_dict"] = new_dict
    if "task" in ckpt["config"]:
        ckpt["config"].pop("task")
    return ckpt


if TYPE_CHECKING:
    from pathlib import Path

    from torch_geometric.data import Batch


class UCalculator(OCPCalculator):
    """ASE based calculator using an OCP model"""

    def __init__(
        self,
        config_yml: str | None = None,
        checkpoint_path: str | None = None,
        model_name: str | None = None,
        local_cache: str | None = None,
        trainer: str | None = None,
        cpu: bool = True,
        seed: int | None = None,
        only_output: list[str] | None = None,
    ) -> None:

        setup_imports()
        setup_logging()
        Calculator.__init__(self)

        if model_name is not None:
            if checkpoint_path is not None:
                raise RuntimeError(
                    "model_name and checkpoint_path were both specified, please use only one at a time"
                )
            if local_cache is None:
                raise NotImplementedError(
                    "Local cache must be set when specifying a model name"
                )
            checkpoint_path = model_name_to_local_file(
                model_name=model_name, local_cache=local_cache
            )

        # Either the config path or the checkpoint path needs to be provided
        assert config_yml or checkpoint_path is not None

        checkpoint = None
        if config_yml is not None:
            if isinstance(config_yml, str):
                config, duplicates_warning, duplicates_error = load_config(config_yml)
                if len(duplicates_warning) > 0:
                    logging.warning(
                        f"Overwritten config parameters from included configs "
                        f"(non-included parameters take precedence): {duplicates_warning}"
                    )
                if len(duplicates_error) > 0:
                    raise ValueError(
                        f"Conflicting (duplicate) parameters in simultaneously "
                        f"included configs: {duplicates_error}"
                    )
            else:
                config = config_yml
            if config["model"]["name"].startswith("compiled"):
                config["model"]["name"] = config["model"]["name"].replace(
                    "compiled_", ""
                )

            # Only keeps the train data that might have normalizer values
            if isinstance(config["dataset"], list):
                config["dataset"] = config["dataset"][0]
            elif isinstance(config["dataset"], dict):
                config["dataset"] = config["dataset"].get("train", None)
        else:
            # Loads the config from the checkpoint directly (always on CPU).
            checkpoint = torch.load(
                checkpoint_path, map_location=torch.device("cpu"), weights_only=False
            )
            checkpoint = convert_compiled_ckpt(checkpoint)
            config = checkpoint["config"]

        if trainer is not None:
            config["trainer"] = trainer
        else:
            config["trainer"] = config.get("trainer", "default")

        if "model_attributes" in config:
            config["model_attributes"]["name"] = config.pop("model")
            config["model"] = config["model_attributes"]

        # Calculate the edge indices on the fly
        config["model"]["otf_graph"] = True

        ### backwards compatability with OCP v<2.0
        config = update_config(config)

        self.config = copy.deepcopy(config)
        self.config["checkpoint"] = str(checkpoint_path)
        del config["dataset"]["src"]

        # some models that are published have configs that include tasks
        # which are not output by the model
        if only_output is not None:
            assert isinstance(
                only_output, list
            ), "only output must be a list of targets to output"
            for key in only_output:
                assert (
                    key in config["outputs"]
                ), f"{key} listed in only_outputs is not present in current model outputs {config['outputs'].keys()}"
            remove_outputs = set(config["outputs"].keys()) - set(only_output)
            for key in remove_outputs:
                config["outputs"].pop(key)

        trainer_cls = Trainer

        self.trainer = trainer_cls(
            task=config.get("task", {}),
            model=config["model"],
            dataset=[config["dataset"]],
            outputs=config["outputs"],
            loss_functions=config["loss_functions"],
            evaluation_metrics=config["evaluation_metrics"],
            optimizer=config["optim"],
            identifier="",
            slurm=config.get("slurm", {}),
            local_rank=config.get("local_rank", 0),
            is_debug=config.get("is_debug", True),
            cpu=cpu,
            amp=config.get("amp", False),
            inference_only=True,
        )

        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path=checkpoint_path, checkpoint=checkpoint)

        seed = seed if seed is not None else self.trainer.config["cmd"]["seed"]
        if seed is None:
            logging.warning(
                "No seed has been set in modelcheckpoint or OCPCalculator! Results may not be reproducible on re-run"
            )
        else:
            self.trainer.set_seed(seed)

        if "otf_graph" in config["model"]:  # temporalily impemented.
            r_edges = not config["model"]["otf_graph"]
        else:
            r_edges = not self.trainer.model.otf_graph

        self.a2g = AtomsToGraphs(
            r_energy=False,
            r_forces=False,
            r_distances=False,
            r_pbc=True,
            # r_edges=not self.trainer.model.otf_graph,  # otf graph should not be a property of the model
            r_edges=r_edges,
        )
        self.implemented_properties = list(self.config["outputs"].keys())
