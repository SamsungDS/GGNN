"""
This file is derived from fairchem-core.
Original source: https://github.com/facebookresearch/fairchem

Copyright (c) Meta Platforms, Inc. and affiliates.
Licensed under the MIT License. See LICENSE for details.

Modifications copyright (c) 2026 Samsung Electronics.
Licensed under CC BY-NC-SA 4.0. See LICENSE for details.
"""

from __future__ import annotations

import copy
import datetime
import errno
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import TYPE_CHECKING, Any
from GGNN.modules.loss import voigtMAELoss
from GGNN.modules.loss_ext import DDPLossExt
from GGNN.modules.evaluator_ext import EvaluatorExt
from GGNN.datasets import ase_arrays_dataset
from GGNN.datasets.lmdb_dataset import data_list_collater

from collections import defaultdict

import numpy as np
import numpy.typing as npt
import torch
import torch_geometric
import yaml
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data import DataLoader
from tqdm import tqdm

from fairchem.core import __version__
from fairchem.core.common import distutils, gp_utils
from fairchem.core.common.data_parallel import BalancedBatchSampler
from fairchem.core.common.logger import WandBSingletonLogger
from ..common.logger import FilesLogger
from fairchem.core.common.registry import registry
from fairchem.core.common.slurm import (
    add_timestamp_id_to_submission_pickle,
)
from fairchem.core.common.typing import assert_is_instance as aii
from fairchem.core.common.typing import none_throws
from fairchem.core.common.utils import (
    get_commit_hash,
    load_state_dict,
    match_state_dict,
    save_checkpoint,
    update_config,
)

from fairchem.core.datasets.base_dataset import create_dataset
from fairchem.core.modules.evaluator import Evaluator
from fairchem.core.modules.exponential_moving_average import ExponentialMovingAverage
from fairchem.core.modules.loss import DDPLoss
from fairchem.core.modules.normalization.element_references import (
    create_element_references,
    load_references_from_config,
)
from fairchem.core.modules.normalization.normalizer import (
    create_normalizer,
    load_normalizers_from_config,
)
from fairchem.core.modules.scaling.compat import load_scales_compat
from fairchem.core.modules.scaling.util import ensure_fitted
from fairchem.core.modules.scheduler import LRScheduler

if TYPE_CHECKING:
    from collections.abc import Sequence

from ..model.EquFlash.equflash import EquFlash
from ..model.EquFlashV2.equflash_v2 import EquFlashV2
from ..model.EquFlash.equflash_comp import CompiledEquFlash
from ..model.EquFlashV2.equflash_v2_comp import CompiledEquFlashV2
from ..model.EquFlash.equflash_pol import EquFlash_pol


from fairchem.core.common.utils import cg_change_mat, irreps_sum

if TYPE_CHECKING:
    from torch_geometric.data import Batch


@registry.register_trainer("energy")
@registry.register_trainer("forces")
@registry.register_trainer("base")
class Trainer(ABC):
    """
    Args:
        task (dict): Task configuration.
        model (dict): Model configuration.
        outputs (dict): Output property configuration.
        dataset (dict): Dataset configuration. The dataset needs to be a SinglePointLMDB dataset.
        optimizer (dict): Optimizer configuration.
        loss_functions (dict): Loss function configuration.
        evaluation_metrics (dict): Evaluation metrics configuration.
        identifier (str): Experiment identifier that is appended to log directory.
        run_dir (str, optional): Path to the run directory where logs are to be saved.
            (default: :obj:`None`)
        is_debug (bool, optional): Run in debug mode.
            (default: :obj:`False`)
        print_every (int, optional): Frequency of printing logs.
            (default: :obj:`100`)
        seed (int, optional): Random number seed.
            (default: :obj:`None`)
        logger (str, optional): Type of logger to be used.
            (default: :obj:`wandb`)
        amp (bool, optional): Run using automatic mixed precision.
            (default: :obj:`False`)
        slurm (dict): Slurm configuration. Currently just for keeping track.
            (default: :obj:`{}`)
    """

    def __init__(
        self,
        task: dict[str, str | Any],
        model: dict[str, Any],
        outputs: dict[str, str | int],
        dataset: dict[str, str | float],
        optimizer: dict[str, str | float],
        loss_functions: dict[str, str | float],
        evaluation_metrics: dict[str, str],
        identifier: str,
        # TODO: dealing with local rank is dangerous
        # T201111838 remove this and use CUDA_VISIBILE_DEVICES instead so trainers don't need to know about which devie to use
        local_rank: int,
        timestamp_id: str | None = None,
        run_dir: str | None = None,
        is_debug: bool = False,
        print_every: int = 100,
        seed: int | None = None,
        logger: (
            str | None
        ) = "files",  # = "wandb", #from common.utils.new_training_context
        amp: bool = False,
        cpu: bool = False,
        name: str = "ocp",
        slurm=None,
        gp_gpus: int | None = None,
        inference_only: bool = False,
    ):
        if slurm is None:
            slurm = {}
        self.name = name
        self.is_debug = is_debug
        self.cpu = cpu
        self.epoch = 0
        self.step = 0
        self.ema = None
        if torch.cuda.is_available() and not self.cpu:
            logging.info(f"local rank base: {local_rank}")
            self.device = torch.device(f"cuda:{local_rank}")
        else:
            self.device = torch.device("cpu")
            self.cpu = True  # handle case when `--cpu` isn't specified
            # but there are no gpu devices available

        if run_dir is None:
            run_dir = os.getcwd()

        self.timestamp_id: str
        if os.path.isdir(os.path.join(run_dir, identifier, "checkpoints")):
            timestamp_id = self._get_timestamp(self.device, identifier)
        else:
            self._get_timestamp(self.device, None)
            timestamp_id = identifier

        self.timestamp_id = none_throws(timestamp_id)

        commit_hash = get_commit_hash()

        logger_name = logger if isinstance(logger, str) else logger["name"]

        self.config = {
            "task": task,
            "trainer": name,
            "model": model,
            "outputs": outputs,
            "optim": optimizer,
            "loss_functions": loss_functions,
            "evaluation_metrics": evaluation_metrics,
            "logger": logger,
            "amp": amp,
            "gpus": distutils.get_world_size() if not self.cpu else 0,
            "cmd": {
                "identifier": identifier,
                "print_every": print_every,
                "seed": seed,
                "timestamp_id": self.timestamp_id,
                "commit": commit_hash,
                "version": __version__,
                "checkpoint_dir": os.path.join(
                    run_dir, self.timestamp_id, "checkpoints"
                ),
                "results_dir": os.path.join(run_dir, self.timestamp_id, "results"),
                "logs_dir": os.path.join(run_dir, self.timestamp_id, "logs"),
            },
            "slurm": slurm,
            "gp_gpus": gp_gpus,
        }
        # AMP Scaler
        self.scaler = torch.amp.GradScaler() if amp and not self.cpu else None

        # Fill in SLURM information in config, if applicable
        if "SLURM_JOB_ID" in os.environ and "folder" in self.config["slurm"]:
            if "SLURM_ARRAY_JOB_ID" in os.environ:
                self.config["slurm"]["job_id"] = "{}_{}".format(
                    os.environ["SLURM_ARRAY_JOB_ID"],
                    os.environ["SLURM_ARRAY_TASK_ID"],
                )
            else:
                self.config["slurm"]["job_id"] = os.environ["SLURM_JOB_ID"]
            self.config["slurm"]["folder"] = self.config["slurm"]["folder"].replace(
                "%j", self.config["slurm"]["job_id"]
            )
            if distutils.is_master():
                add_timestamp_id_to_submission_pickle(
                    self.config["slurm"]["folder"],
                    self.config["slurm"]["job_id"],
                    self.timestamp_id,
                )

        # Define datasets
        if isinstance(dataset, list):
            if len(dataset) > 0:
                self.config["dataset"] = dataset[0]
            if len(dataset) > 1:
                self.config["val_dataset"] = dataset[1]
            if len(dataset) > 2:
                self.config["test_dataset"] = dataset[2]
        elif isinstance(dataset, dict):
            # or {} in cases where "dataset": None is explicitly defined
            self.config["dataset"] = dataset.get("train", {}) or {}
            self.config["val_dataset"] = dataset.get("val", {}) or {}
            self.config["test_dataset"] = dataset.get("test", {}) or {}
        else:
            self.config["dataset"] = dataset or {}

        # add empty dicts for missing datasets
        for dataset_name in ("val_dataset", "test_dataset"):
            if dataset_name not in self.config:
                self.config[dataset_name] = {}

        if not inference_only and not is_debug and distutils.is_master():
            os.makedirs(self.config["cmd"]["checkpoint_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["results_dir"], exist_ok=True)
            os.makedirs(self.config["cmd"]["logs_dir"], exist_ok=True)

        ### backwards compatability with OCP v<2.0
        self.config = update_config(self.config)

        if distutils.is_master():
            logging.info(yaml.dump(self.config, default_flow_style=False))

        # define attributes for readability
        self.elementrefs = {}
        self.normalizers = {}
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.best_val_metric = None
        self.primary_metric = None
        self.load(inference_only)

    def train(self, disable_eval_tqdm: bool = False) -> None:
        ensure_fitted(self._unwrapped_model, warn=True)

        eval_every = self.config["optim"].get("eval_every", len(self.train_loader))
        checkpoint_every = self.config["optim"].get("checkpoint_every", eval_every)
        primary_metric = self.evaluation_metrics.get(
            "primary_metric", self.evaluator.task_primary_metric[self.name]
        )
        if not hasattr(self, "primary_metric") or self.primary_metric != primary_metric:
            self.best_val_metric = 1e9 if "mae" in primary_metric else -1.0
        else:
            primary_metric = self.primary_metric
        self.metrics = {}

        # Calculate start_epoch from step instead of loading the epoch number
        # to prevent inconsistencies due to different batch size in checkpoint.
        start_epoch = self.step // len(self.train_loader)
        rank = distutils.get_rank()

        for epoch_int in range(start_epoch, self.config["optim"]["max_epochs"]):
            skip_steps = self.step % len(self.train_loader)
            self.train_sampler.set_epoch_and_start_iteration(epoch_int, skip_steps)
            train_loader_iter = iter(self.train_loader)
            for i in range(skip_steps, len(self.train_loader)):
                self.epoch = epoch_int + (i + 1) / len(self.train_loader)
                self.step = epoch_int * len(self.train_loader) + i + 1
                if os.environ.get("PROFILE", False) and self.step == 40:
                    torch.cuda.profiler.start()
                if os.environ.get("PROFILE", False) and self.step == 50:
                    return
                self.model.train()
                # Get a batch.
                batch = next(train_loader_iter)
                # Forward, loss, backward.
                with torch.amp.autocast(
                    device_type="cuda", enabled=self.scaler is not None
                ):
                    out = self._forward(batch)
                    loss = self._compute_loss(out, batch)

                # Compute metrics.
                self.metrics = self._compute_metrics(
                    out,
                    batch,
                    self.evaluator,
                    self.metrics,
                )
                self.metrics = self.evaluator.update("loss", loss.item(), self.metrics)

                loss = self.scaler.scale(loss) if self.scaler else loss
                to_continue = self._backward(loss)
                if not to_continue and distutils.is_master():
                    logging.info(
                        f"skipping parameter update due to grad NaN on step {self.step} epoch {self.epoch:.4f}"
                    )
                # Log metrics.
                log_dict = {k: self.metrics[k]["metric"] for k in self.metrics}
                log_dict.update(
                    {
                        "lr": self.scheduler.get_lr(),
                        "epoch": self.epoch,
                        "step": self.step,
                    }
                )
                if (
                    self.step % self.config["cmd"]["print_every"] == 0
                    and distutils.is_master()
                ):
                    log_str = [f"{k}: {v:.2e}" for k, v in log_dict.items()]
                    logging.info(", ".join(log_str))
                    self.metrics = {}

                    if self.logger is not None:
                        self.logger.log(
                            log_dict,
                            step=self.step,
                            split="train",
                        )

                if (
                    checkpoint_every != -1
                    and self.step % checkpoint_every == 0
                    and checkpoint_every != len(self.train_loader)
                ):
                    self.save(
                        checkpoint_file=f"checkpoint_{self.step}.pt",
                        training_state=True,
                    )
                    self.save(
                        checkpoint_file=f"checkpoint_last.pt", training_state=True
                    )
                if self.step % 1000 == 0:
                    self.save(
                        checkpoint_file=f"checkpoint_last.pt", training_state=True
                    )
                # Evaluate on val set every `eval_every` iterations.
                if self.step % eval_every == 0:
                    if self.val_loader is not None:
                        val_metrics = self.validate(
                            split="val",
                            disable_tqdm=disable_eval_tqdm,
                        )
                        self.update_best(
                            primary_metric,
                            val_metrics,
                            disable_eval_tqdm=disable_eval_tqdm,
                        )

                    
                if self.scheduler.scheduler_type == "ReduceLROnPlateau":
                    if self.step % eval_every == 0:
                        self.scheduler.step(
                            metrics=val_metrics[primary_metric]["metric"],
                        )
                else:
                    self.scheduler.step()
                    if(self.scheduler_muon is not None):
                        self.scheduler_muon.step()
                

            torch.cuda.empty_cache()
            self.save(checkpoint_file=f"checkpoint_last.pt", training_state=True)

    @staticmethod
    def _get_timestamp(device: torch.device, prefix: str | None) -> str:
        now = datetime.datetime.now().timestamp()
        timestamp_tensor = torch.tensor(now).to(device)
        # create directories from master rank only
        distutils.broadcast(timestamp_tensor, 0)
        timestamp_str = datetime.datetime.fromtimestamp(
            timestamp_tensor.float().item()
        ).strftime("%Y-%m-%d-%H-%M-%S")
        if prefix:
            timestamp_str = prefix + "_" + timestamp_str
        return timestamp_str

    def load(self, inference_only: bool) -> None:
        self.load_seed_from_config()
        self.load_logger(inference_only)
        self.load_task()
        if inference_only is False:
            self.load_datasets()
        self.load_model()

        if inference_only is False:
            # self.load_datasets()
            self.load_references_and_normalizers()
            self.load_loss()
            self.load_optimizer()
            self.load_extras()

        if self.config["optim"].get("load_datasets_and_model_then_exit", False):
            sys.exit(0)

    @staticmethod
    def set_seed(seed) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def load_seed_from_config(self) -> None:
        # https://pytorch.org/docs/stable/notes/randomness.html
        if self.config["cmd"]["seed"] is None:
            return
        self.set_seed(self.config["cmd"]["seed"])

    def load_logger(self, inference_only=False) -> None:
        self.logger = None
        if not inference_only and not self.is_debug and distutils.is_master():
            assert self.config["logger"] is not None, "Specify logger in config"

            logger = self.config["logger"]
            logger_name = logger if isinstance(logger, str) else logger["name"]
            assert logger_name, "Specify logger name"

            if logger_name == "wandb_singleton":
                WandBSingletonLogger.init_wandb(
                    config=self.config,
                    run_id=self.config["cmd"]["timestamp_id"],
                    run_name=self.config["cmd"]["identifier"],
                    log_dir=self.config["cmd"]["logs_dir"],
                    project=self.config["logger"]["project"],
                    entity=self.config["logger"]["entity"],
                    group=self.config["logger"].get("group", ""),
                )
                self.logger = WandBSingletonLogger.get_instance()
            else:
                self.logger = registry.get_logger_class(logger_name)(self.config)
            if logger_name == "files":
                self.logger.log_model_training_info()

    def get_sampler(self, dataset, batch_size: int, shuffle: bool):
        if self.config["optim"].get("max_atoms", None):
            from GGNN.datasets.samplers import MaxAtomDistributedBatchSampler

            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()
            max_atoms = self.config["optim"].get("max_atoms", None)
            return MaxAtomDistributedBatchSampler(
                dataset, max_atoms, num_replicas, rank, 0, True, False
            )
        balancing_mode = self.config["optim"].get("load_balancing", None)
        on_error = self.config["optim"].get("load_balancing_on_error", None)
        if balancing_mode is not None:
            if on_error is None:
                on_error = "raise"
        else:
            balancing_mode = "atoms"

        if on_error is None:
            on_error = "warn_and_no_balance"

        if gp_utils.initialized():
            num_replicas = gp_utils.get_dp_world_size()
            rank = gp_utils.get_dp_rank()
        else:
            num_replicas = distutils.get_world_size()
            rank = distutils.get_rank()

        # returnMaxAtomDistributedBatchSampler(dataset,350,num_replicas,rank,0,True,False)
        return BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=num_replicas,
            rank=rank,
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            on_error=on_error,
            seed=self.config["cmd"]["seed"],
        )

    def get_dataloader(self, dataset, sampler) -> DataLoader:
        return DataLoader(
            dataset,
            collate_fn=self.collater,
            num_workers=self.config["optim"]["num_workers"],
            pin_memory=True,
            batch_sampler=sampler,
        )

    def load_datasets(self) -> None:
        if "backbone" in self.config["model"]:
            model_config = self.config["model"]["backbone"]
        else:
            model_config = self.config["model"]

        self.collater = partial(
            data_list_collater,
            cutoff=model_config["cutoff"],
            max_neighbors=model_config["max_neighbors"],
            use_pbc=model_config["use_pbc"],
            otf_graph=model_config.get("otf_graph", False),
            exclude_keys=["sid", "fid"],
        )
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

        # This is hacky and scheduled to be removed next BE week
        # move ['X_split_settings'] to ['splits'][X]
        def convert_settings_to_split_settings(config, split_name):
            config = copy.deepcopy(config)  # make sure we dont modify the original
            if f"{split_name}_split_settings" in config:
                config["splits"] = {
                    split_name: config.pop(f"{split_name}_split_settings")
                }
            return config

        # load train, val, test datasets
        if "src" in self.config["dataset"]:
            logging.info(
                f"Loading dataset: {self.config['dataset'].get('format', 'lmdb')}"
            )

            self.train_dataset = create_dataset(
                convert_settings_to_split_settings(self.config["dataset"], "train"),
                "train",
            )
            self.train_sampler = self.get_sampler(
                self.train_dataset,
                self.config["optim"]["batch_size"],
                shuffle=True,
            )
            self.train_loader = self.get_dataloader(
                self.train_dataset,
                self.train_sampler,
            )

        if (
            "first_n" in self.config["dataset"]
            or "sample_n" in self.config["dataset"]
            or "max_atom" in self.config["dataset"]
        ):
            logging.warning(
                "Dataset attributes (first_n/sample_n/max_atom) passed to all datasets! Please don't do this, its dangerous!\n"
                + "Add them under each dataset 'train_split_settings'/'val_split_settings'/'test_split_settings'"
            )

        if "src" in self.config["val_dataset"]:
            if self.config["val_dataset"].get("use_train_settings", True):
                val_config = self.config["dataset"].copy()
                val_config.update(self.config["val_dataset"])
            else:
                val_config = self.config["val_dataset"]

            self.val_dataset = create_dataset(
                convert_settings_to_split_settings(val_config, "val"), "val"
            )
            self.val_sampler = self.get_sampler(
                self.val_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )
            self.val_loader = self.get_dataloader(
                self.val_dataset,
                self.val_sampler,
            )

        if "src" in self.config["test_dataset"]:
            if self.config["test_dataset"].get("use_train_settings", True):
                test_config = self.config["dataset"].copy()
                test_config.update(self.config["test_dataset"])
            else:
                test_config = self.config["test_dataset"]

            self.test_dataset = create_dataset(
                convert_settings_to_split_settings(test_config, "test"), "test"
            )
            self.test_sampler = self.get_sampler(
                self.test_dataset,
                self.config["optim"].get(
                    "eval_batch_size", self.config["optim"]["batch_size"]
                ),
                shuffle=False,
            )

            # DEBUGGED: re-introduce ["sid", "fid"] for self.test_loader, for utrainer.predict()
            self.test_loader = DataLoader(
                self.test_dataset,
                collate_fn=partial(
                    data_list_collater,
                    cutoff=model_config["cutoff"],
                    max_neighbors=model_config["max_neighbors"],
                    use_pbc=model_config["use_pbc"],
                    otf_graph=model_config.get("otf_graph", False),
                    # exclude_keys=["sid", "fid"],    # KEEP sid, fid here
                ),
                num_workers=self.config["optim"]["num_workers"],
                pin_memory=True,
                batch_sampler=self.test_sampler,
            )

    def load_references_and_normalizers(self):
        """Load or create element references and normalizers from config"""
        # Is it troublesome that we assume any normalizer info is in train? What if there is no
        # training dataset? What happens if we just specify a test

        elementref_config = (
            self.config["dataset"].get("transforms", {}).get("element_references")
        )
        norms_config = self.config["dataset"].get("transforms", {}).get("normalizer")

        # if stats.yaml in dataset config, use stats.yaml to construct elementrefs and normalizer
        from GGNN.common.statistics import read_statistics
        from fairchem.core.modules.normalization.element_references import (
            LinearReferences,
        )
        from fairchem.core.modules.normalization.normalizer import Normalizer

        if "stat_dir" in self.config["dataset"]:
            scale, shift, type_map, chemical_specie = read_statistics(
                self.config["dataset"]["stat_dir"],
                shift_by=self.config["dataset"].get(
                    "scale_by", "elemwise_reference_energies"
                ),
                scale_by=self.config["dataset"].get("scale_by", "force_rms"),
            )
            scale = torch.tensor(scale)
            shift = torch.tensor(shift)
            if len(shift) > 1:
                self.elementrefs.update(
                    {"energy": LinearReferences(shift).to(device=self.device)}
                )
                self.normalizers.update(
                    {"energy": Normalizer(rmsd=scale).to(device=self.device)}
                )
            else:
                self.normalizers.update(
                    {
                        "energy": Normalizer(mean=shift, rmsd=scale).to(
                            device=self.device
                        )
                    }
                )
            for target in self.output_targets:
                if target != "energy":
                    self.normalizers.update(
                        {target: Normalizer(rmsd=scale).to(device=self.device)}
                    )
            return

        elementrefs, normalizers = {}, {}
        if distutils.is_master():
            if elementref_config is not None:
                # put them in a list to allow broadcasting python objects
                elementrefs = load_references_from_config(
                    elementref_config,
                    dataset=self.train_dataset,
                    seed=self.config["cmd"]["seed"],
                    checkpoint_dir=(
                        self.config["cmd"]["checkpoint_dir"]
                        if not self.is_debug
                        else None
                    ),
                )

            if norms_config is not None:
                normalizers = load_normalizers_from_config(
                    norms_config,
                    dataset=self.train_dataset,
                    seed=self.config["cmd"]["seed"],
                    checkpoint_dir=(
                        self.config["cmd"]["checkpoint_dir"]
                        if not self.is_debug
                        else None
                    ),
                    element_references=elementrefs,
                )

                # log out the values that will be used.
                for output, normalizer in normalizers.items():
                    logging.info(
                        f"Normalization values for output {output}: mean={normalizer.mean.item()}, rmsd={normalizer.rmsd.item()}."
                    )

        # put them in a list to broadcast them
        elementrefs, normalizers = [elementrefs], [normalizers]
        distutils.broadcast_object_list(
            object_list=elementrefs, src=0, device=self.device
        )
        distutils.broadcast_object_list(
            object_list=normalizers, src=0, device=self.device
        )
        # make sure element refs and normalizers are on this device
        self.elementrefs.update(
            {
                output: elementref.to(self.device)
                for output, elementref in elementrefs[0].items()
            }
        )
        self.normalizers.update(
            {
                output: normalizer.to(self.device)
                for output, normalizer in normalizers[0].items()
            }
        )

    def load_task(self):
        self.output_targets = {}
        for target_name in self.config["outputs"]:
            self.output_targets[target_name] = self.config["outputs"][target_name]
            if "decomposition" in self.config["outputs"][target_name]:
                for subtarget in self.config["outputs"][target_name]["decomposition"]:
                    self.output_targets[subtarget] = (
                        self.config["outputs"][target_name]["decomposition"]
                    )[subtarget]
                    self.output_targets[subtarget]["parent"] = target_name
                    # inherent properties if not available
                    if "level" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["level"] = self.config[
                            "outputs"
                        ][target_name].get("level", "system")
                    if "train_on_free_atoms" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["train_on_free_atoms"] = (
                            self.config["outputs"][target_name].get(
                                "train_on_free_atoms", True
                            )
                        )
                    if "eval_on_free_atoms" not in self.output_targets[subtarget]:
                        self.output_targets[subtarget]["eval_on_free_atoms"] = (
                            self.config["outputs"][target_name].get(
                                "eval_on_free_atoms", True
                            )
                        )

        # TODO: Assert that all targets, loss fn, metrics defined are consistent
        self.evaluation_metrics = self.config.get("evaluation_metrics", {})
        if "born_charge" in self.config["outputs"]:
            self.evaluator = EvaluatorExt(
                task=self.name,
                eval_metrics=self.evaluation_metrics.get(
                    "metrics", Evaluator.task_metrics.get(self.name, {})
                ),
            )
        else:
            self.evaluator = Evaluator(
                task=self.name,
                eval_metrics=self.evaluation_metrics.get(
                    "metrics", Evaluator.task_metrics.get(self.name, {})
                ),
            )

    def load_model(self) -> None:
        # Build model
        if distutils.is_master():
            logging.info(f"Loading model: {self.config['model']['name']}")

        model_config_copy = copy.deepcopy(self.config["model"])
        model_name = model_config_copy.pop("name")

        if "equflash" in model_name:
            self.model = registry.get_model_class(model_name)(
                self.config["model"],
            ).to(self.device)
        else:
            self.model = registry.get_model_class(model_name)(
                **model_config_copy,
            ).to(self.device)

        if distutils.is_master():
            print(self.model)
        if self.config.get("task", {}).get("finetune", False):
            checkpoint = self.config["task"]["finetune"]["checkpoint"]
            checkpoint = torch.load(checkpoint, map_location=self.device, weights_only=False)
                    # ckpt compatibility for compiled/non-compiled model
            ckpt_model_name = checkpoint["config"]["model"]["name"]
            trainer_model_name = self.config["model"]["name"]
        
            if trainer_model_name == "compiled_" + ckpt_model_name:
                
                from collections import OrderedDict

                new_dict = OrderedDict()
                for k, v in checkpoint["state_dict"].items():
                    if k.startswith("module.") and "z_to_onehot_tensor" not in k:
                        new_dict[k.replace("module.", "module.model.")] = v
                    elif not k.startswith("model."):
                        new_dict["model." + k] = v
                checkpoint["state_dict"] = new_dict

            elif "compiled_" + trainer_model_name == ckpt_model_name:
                from GGNN.common.calculator import convert_compiled_ckpt
                checkpoint = convert_compiled_ckpt(checkpoint)
                
            new_dict = match_state_dict(
                self.model.state_dict(), checkpoint["state_dict"]
            )
            strict = self.config.get("task", {}).get("strict_load", True)
            reset_shift = self.config.get("task", {}).get("reset_shift", False)
            reset_scale = self.config.get("task", {}).get("reset_scale", False)
            if self.config.get("task", {}).get("reset_stat", False):
                reset_shift = True
                reset_scale = True
            if reset_shift:
                new_dict = {
                    k: v
                    for k, v in new_dict.items()
                    if "rescale_atomic_energy.shift" not in k
                }
            if reset_scale:
                new_dict = {
                    k: v
                    for k, v in new_dict.items()
                    if "rescale_atomic_energy.scale" not in k
                }
            if self.config["task"]["finetune"].get("reset_head", False):
                new_dict = {
                    k: v
                    for k, v in new_dict.items()
                    if "atomic_reduce.linear_out" not in k
                }
            load_state_dict(self.model, new_dict, strict=strict)

        num_params = sum(p.numel() for p in self.model.parameters())

        if distutils.is_master():
            logging.info(
                f"Loaded {self.model.__class__.__name__} with "
                f"{num_params} parameters."
            )
        if self.logger is not None:
            # only "watch" model if user specify watch: True because logging gradients
            # spews too much data into W&B and makes the UI slow to respond
            if "watch" in self.config["logger"]:
                self.logger.watch(
                    self.model, log_freq=int(self.config["logger"]["watch"])
                )
            self.logger.log_summary({"num_params": num_params})
        if distutils.initialized():
            self.model = DistributedDataParallel(
                self.model,
                broadcast_buffers=False,
            )

    @property
    def _unwrapped_model(self):
        module = self.model
        while isinstance(module, DistributedDataParallel):
            module = module.module
        return module

    def load_checkpoint(
        self,
        checkpoint_path: str,
        checkpoint: dict | None = None,
        inference_only: bool = False,
    ) -> None:
        map_location = torch.device("cpu") if self.cpu else self.device
        if checkpoint is None:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    errno.ENOENT, "Checkpoint file not found", checkpoint_path
                )
            logging.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location=map_location, weights_only=False
            )

        # attributes that are necessary for training and validation
        if inference_only is False:
            self.epoch = checkpoint.get("epoch", 0)
            self.step = checkpoint.get("step", 0)
            self.best_val_metric = checkpoint.get("best_val_metric", None)
            self.primary_metric = checkpoint.get("primary_metric", None)

            if "optimizer" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer"])

            if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
                self.scheduler.scheduler.load_state_dict(checkpoint["scheduler"])
            if "optimizer_muon" in checkpoint:
                self.optimizer_muon.load_state_dict(checkpoint["optimizer_muon"])
                checkpoint["scheduler"]["use_beta1"] = False
                self.scheduler_muon.scheduler.load_state_dict(checkpoint["scheduler"])

        else:
            logging.info(
                "Loading checkpoint in inference-only mode, not loading keys associated with trainer state!"
            )

        if "ema" in checkpoint and checkpoint["ema"] is not None and self.ema != None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self.ema = None
        checkpoint = self.convert_checkpoint(checkpoint)

        # ckpt compatibility for compiled/non-compiled model
        ckpt_model_name = checkpoint["config"]["model"]["name"]
        trainer_model_name = self.config["model"]["name"]
      
        if trainer_model_name == "compiled_" + ckpt_model_name:
            
            from collections import OrderedDict

            new_dict = OrderedDict()
            for k, v in checkpoint["state_dict"].items():
                if k.startswith("module.") and "z_to_onehot_tensor" not in k:
                    new_dict[k.replace("module.", "module.model.")] = v
                elif not k.startswith("model."):
                    new_dict["model." + k] = v
            checkpoint["state_dict"] = new_dict

        elif "compiled_" + trainer_model_name == ckpt_model_name:
            from GGNN.common.calculator import convert_compiled_ckpt

            checkpoint = convert_compiled_ckpt(checkpoint)

        new_dict = match_state_dict(self.model.state_dict(), checkpoint["state_dict"])
        strict = self.config.get("task", {}).get("strict_load", True)

        load_state_dict(self.model, new_dict, strict=False)

        scale_dict = checkpoint.get("scale_dict", None)
        if scale_dict:
            logging.info(
                "Overwriting scaling factors with those loaded from checkpoint. "
                "If you're generating predictions with a pretrained checkpoint, this is the correct behavior. "
                "To disable this, delete `scale_dict` from the checkpoint. "
            )
            load_scales_compat(self._unwrapped_model, scale_dict)

        for key, state_dict in checkpoint["normalizers"].items():
            ### Convert old normalizer keys to new target keys
            if key == "target":
                target_key = "energy"
            elif key == "grad_target":
                target_key = "forces"
            else:
                target_key = key

            if target_key not in self.normalizers:
                self.normalizers[target_key] = create_normalizer(state_dict=state_dict)
            else:
                mkeys = self.normalizers[target_key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0

            self.normalizers[target_key].to(map_location)

        for key, state_dict in checkpoint.get("elementrefs", {}).items():
            if key not in self.elementrefs:
                self.elementrefs[key] = create_element_references(state_dict=state_dict)
            else:
                mkeys = self.elementrefs[key].load_state_dict(state_dict)
                assert len(mkeys.missing_keys) == 0
                assert len(mkeys.unexpected_keys) == 0

            self.elementrefs[key].to(map_location)

        if self.scaler and checkpoint["amp"]:
            self.scaler.load_state_dict(checkpoint["amp"])

    def load_loss(self) -> None:
        self.loss_functions = []
        for _idx, loss in enumerate(self.config["loss_functions"]):
            for target in loss:
                assert (
                    "fn" in loss[target]
                ), f"'fn' is not defined in the {target} loss config {loss[target]}."
                loss_name = loss[target].get("fn")
                assert (
                    "coefficient" in loss[target]
                ), f"'coefficient' is not defined in the {target} loss config {loss[target]}."
                coefficient = loss[target].get("coefficient")
                loss_reduction = loss[target].get("reduction")

                if target == "polarization":
                    loss_fn = DDPLossExt(loss_name, reduction=loss_reduction)
                else:
                    loss_fn = DDPLoss(loss_name, reduction=loss_reduction)
                self.loss_functions.append(
                    (target, {"fn": loss_fn, "coefficient": coefficient})
                )

    def load_optimizer(self) -> None:
        optimizer = getattr(torch.optim, self.config["optim"].get("optimizer", "AdamW"))
        optimizer_params = self.config["optim"].get("optimizer_params", {})

        weight_decay = optimizer_params.get("weight_decay", 0)
        if "weight_decay" in self.config["optim"]:
            weight_decay = self.config["optim"]["weight_decay"]
            logging.warning(
                "Using `weight_decay` from `optim` instead of `optim.optimizer_params`."
                "Please update your config to use `optim.optimizer_params.weight_decay`."
                "`optim.weight_decay` will soon be deprecated."
            )
        self.model_params_no_wd = {}
        if hasattr(self._unwrapped_model, "no_weight_decay"):
            self.model_params_no_wd = self._unwrapped_model.no_weight_decay()
        params_2d, name_2d, name_remain = [], [], []
        params_decay, params_no_decay, name_no_decay = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if any(name.endswith(skip_name) for skip_name in self.model_params_no_wd):
                params_no_decay.append(param)
                name_no_decay.append(name)
            else:
                if len(param.shape) == 2:
                    params_2d.append(param)
                    name_2d.append(name)
                else:
                    params_decay.append(param)
                    name_remain.append(name)

        if self.config["optim"].get("muon", False):
            muon_params = self.config["optim"].get("muon_params", {})
            if distutils.is_master():
                logging.info(
                    "Using muon optimizer with setting "
                    + ",\t".join([f"{k} = {v}" for k, v in muon_params.items()])
                )
                logging.info("Parameters without muon and weight decay>0 :")
                logging.info(name_remain)
            self.optimizer_muon = torch.optim.Muon(
                params=[{"params": params_2d, "weight_decay": weight_decay}],
                lr=self.config["optim"]["lr_initial"],
                **muon_params,
            )
        else:
            self.optimizer_muon = None
            params_decay = params_decay + params_2d
        if distutils.is_master():
            logging.info("Parameters without weight decay:")
            logging.info(name_no_decay)
        params_1d = []
        if len(params_no_decay) > 0:
            params_1d.append({"params": params_no_decay, "weight_decay": 0})
        if len(params_decay) > 0:
            params_1d.append({"params": params_decay, "weight_decay": weight_decay})
        self.optimizer = optimizer(
            params=params_1d,
            lr=self.config["optim"]["lr_initial"],
            **optimizer_params,
        )

    def load_extras(self) -> None:
        if self.config["optim"].get("total_iters", None) == "max_epochs":
            self.config["optim"]["total_iters"] = (
                len(self.train_loader) * self.config["optim"]["max_epochs"]
            )
            self.config["optim"]["total_steps"] = self.config["optim"]["total_iters"]

        self.scheduler = LRScheduler(self.optimizer, self.config["optim"])
        if self.optimizer_muon is not None:
            self.scheduler_muon = LRScheduler(self.optimizer_muon, self.config["optim"])
        else:
            self.scheduler_muon = None

        self.clip_grad_norm = aii(
            self.config["optim"].get("clip_grad_norm", None), (int, float)
        )
        self.ema_decay = aii(self.config["optim"].get("ema_decay"), float)
        if self.ema_decay:
            self.ema = ExponentialMovingAverage(
                self.model.parameters(),
                self.ema_decay,
            )
        else:
            self.ema = None

    def save(
        self,
        metrics=None,
        checkpoint_file: str = "checkpoint.pt",
        training_state: bool = True,
    ) -> str | None:
        if not self.is_debug and distutils.is_master():
            state = {
                "state_dict": self.model.state_dict(),
                "normalizers": {
                    key: value.state_dict() for key, value in self.normalizers.items()
                },
                "elementrefs": {
                    key: value.state_dict() for key, value in self.elementrefs.items()
                },
                "config": self.config,
                "val_metrics": metrics,
                "amp": self.scaler.state_dict() if self.scaler else None,
            }
            if training_state:
                state.update(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": (
                            self.scheduler.scheduler.state_dict()
                            if self.scheduler.scheduler_type != "Null"
                            else None
                        ),
                        "config": self.config,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.evaluation_metrics.get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                )
                if self.optimizer_muon is not None:
                    state.update(
                        {
                            "optimizer_muon": self.optimizer_muon.state_dict(),
                        }
                    )
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
            else:
                if self.ema is not None:
                    self.ema.store()
                    self.ema.copy_to()
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=checkpoint_file,
                )
                if self.ema:
                    self.ema.restore()
            return ckpt_path
        return None

    def update_best(
        self,
        primary_metric,
        val_metrics,
        disable_eval_tqdm: bool = True,
    ) -> None:
        if (
            "mae" in primary_metric
            and val_metrics[primary_metric]["metric"] < self.best_val_metric
        ) or (
            "mae" not in primary_metric
            and val_metrics[primary_metric]["metric"] > self.best_val_metric
        ):
            self.best_val_metric = val_metrics[primary_metric]["metric"]
            self.save(
                metrics=val_metrics,
                checkpoint_file="best_checkpoint.pt",
                training_state=False,
            )
            if self.test_loader is not None:
                self.predict(
                    self.test_loader,
                    results_file="predictions",
                    disable_tqdm=disable_eval_tqdm,
                )

    def _aggregate_metrics(self, metrics):
        aggregated_metrics = {}
        for k in metrics:
            aggregated_metrics[k] = {
                "total": distutils.all_reduce(
                    metrics[k]["total"], average=False, device=self.device
                ),
                "numel": distutils.all_reduce(
                    metrics[k]["numel"], average=False, device=self.device
                ),
            }
            aggregated_metrics[k]["metric"] = (
                aggregated_metrics[k]["total"] / aggregated_metrics[k]["numel"]
            )
        return aggregated_metrics

    @torch.no_grad()
    def validate(self, split: str = "val", disable_tqdm: bool = False):
        if distutils.is_master():
            logging.info("ensure fitted...")
        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master():
            logging.info(f"Evaluating on {split}.")

        self.model.eval()
        if self.ema:
            self.ema.store()
            self.ema.copy_to()

        metrics = {}
        if "born_charge" in self.config["outputs"]:
            evaluator = EvaluatorExt(
                task=self.name,
                eval_metrics=self.evaluation_metrics.get(
                    "metrics", Evaluator.task_metrics.get(self.name, {})
                ),
            )
        else:
            evaluator = Evaluator(
                task=self.name,
                eval_metrics=self.evaluation_metrics.get(
                    "metrics", Evaluator.task_metrics.get(self.name, {})
                ),
            )

        rank = distutils.get_rank()

        loader = self.val_loader if split == "val" else self.test_loader

        for _i, batch in tqdm(
            enumerate(loader),
            total=len(loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            # Forward.
            with torch.amp.autocast(
                device_type="cuda", enabled=self.scaler is not None
            ):
                batch.to(self.device)
                out = self._forward(batch)
            loss = self._compute_loss(out, batch)

            # Compute metrics.
            metrics = self._compute_metrics(out, batch, evaluator, metrics)
            metrics = evaluator.update("loss", loss.item(), metrics)

        metrics = self._aggregate_metrics(metrics)

        log_dict = {k: metrics[k]["metric"] for k in metrics}
        log_dict.update({"epoch": self.epoch})
        if distutils.is_master():
            log_str = [f"{k}: {v:.4f}" for k, v in log_dict.items()]
            logging.info(", ".join(log_str))

        # Make plots.
        if self.logger is not None:
            self.logger.log(
                log_dict,
                step=self.step,
                split=split,
            )

        if self.ema:
            self.ema.restore()

        return metrics

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()
        if self.optimizer_muon is not None:
            self.optimizer_muon.zero_grad()
        loss.backward()
        # Scale down the gradients of shared parameters
        if hasattr(self.model, "shared_parameters"):
            for p, factor in self.model.shared_parameters:
                if hasattr(p, "grad") and p.grad is not None:
                    p.grad.detach().div_(factor)
                else:
                    if not hasattr(self, "warned_shared_param_no_grad"):
                        self.warned_shared_param_no_grad = True
                        logging.warning(
                            "Some shared parameters do not have a gradient. "
                            "Please check if all shared parameters are used "
                            "and point to PyTorch parameters."
                        )
        if self.clip_grad_norm:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.clip_grad_norm,
            )
            if torch.isnan(grad_norm).any():
                self.optimizer.zero_grad()
                return False
                state = {
                    "state_dict": self.model.state_dict(),
                    "normalizers": {
                        key: value.state_dict()
                        for key, value in self.normalizers.items()
                    },
                    "elementrefs": {
                        key: value.state_dict()
                        for key, value in self.elementrefs.items()
                    },
                    "config": self.config,
                    "val_metrics": None,
                    "amp": self.scaler.state_dict() if self.scaler else None,
                }
                state.update(
                    {
                        "epoch": self.epoch,
                        "step": self.step,
                        "optimizer": self.optimizer.state_dict(),
                        "scheduler": (
                            self.scheduler.scheduler.state_dict()
                            if self.scheduler.scheduler_type != "Null"
                            else None
                        ),
                        "config": self.config,
                        "ema": self.ema.state_dict() if self.ema else None,
                        "best_val_metric": self.best_val_metric,
                        "primary_metric": self.evaluation_metrics.get(
                            "primary_metric",
                            self.evaluator.task_primary_metric[self.name],
                        ),
                    },
                )
                ckpt_path = save_checkpoint(
                    state,
                    checkpoint_dir=self.config["cmd"]["checkpoint_dir"],
                    checkpoint_file=f"ckpt_grad_explode_{self.step}_{distutils.get_rank()}.pt",
                )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
            if self.optimizer_muon is not None:
                self.optimizer_muon.step()
        if self.ema:
            self.ema.update()
        return True

    def save_results(
        self,
        predictions: dict[str, npt.NDArray],
        results_file: str | None,
        keys: Sequence[str] | None = None,
    ) -> None:
        if results_file is None:
            return
        if keys is None:
            keys = predictions.keys()

        results = distutils.gather_objects(predictions)
        distutils.synchronize()
        if distutils.is_master():
            gather_results = {
                key: list(chain(*(result[key] for result in results))) for key in keys
            }

            # Because of how distributed sampler works, some system ids
            # might be repeated to make no. of samples even across GPUs.
            _, idx = np.unique(gather_results["ids"], return_index=True)
            for k in keys:
                if "chunk_idx" in k:
                    gather_results[k] = np.cumsum([gather_results[k][i] for i in idx])[
                        :-1
                    ]
                else:
                    if f"{k}_chunk_idx" in keys or k == "forces":
                        gather_results[k] = np.concatenate(
                            [gather_results[k][i] for i in idx]
                        )
                    else:
                        gather_results[k] = np.array(
                            [gather_results[k][i] for i in idx]
                        )

            full_path = os.path.join(
                self.config["cmd"]["results_dir"], f"{self.name}_{results_file}.npz"
            )
            logging.info(f"Writing results to {full_path}")
            np.savez_compressed(full_path, **gather_results)

    ## only for sevenn checkpoint
    def convert_checkpoint(self, checkpoint):
        if "state_dict" not in checkpoint:
            checkpoint["state_dict"] = checkpoint["model_state_dict"]
            checkpoint["normalizers"] = {}

        return checkpoint


    def _denorm_preds(self, target_key: str, prediction: torch.Tensor, batch: Batch):
        """Convert model output from a batch into raw prediction by denormalizing and adding references"""
        # denorm the outputs
        if target_key in self.normalizers:
            prediction = self.normalizers[target_key](prediction)  # recent. to be fixed
            # prediction = self.normalizers[target_key].denorm(prediction)

        # recent version. to be fixed
        # # add element references
        if target_key in self.elementrefs:
            prediction = self.elementrefs[target_key](prediction, batch)

        return prediction

    def _forward(self, batch):
        out = self.model(batch.to(self.device))

        outputs = {}
        batch_size = batch.natoms.numel()
        num_atoms_in_batch = batch.natoms.sum()

        for target_key in self.output_targets:
            ### Target property is a direct output of the model
            if target_key in out:
                if isinstance(out[target_key], torch.Tensor):
                    pred = out[target_key]
                elif isinstance(out[target_key], dict):
                    # if output is a nested dictionary (in the case of hydra models), we attempt to retrieve it using the property name
                    # ie: "output_head_name.property"
                    assert (
                        "property" in self.output_targets[target_key]
                    ), f"we need to know which property to match the target to, please specify the property field in the task config, current config: {self.output_targets[target_key]}"
                    prop = self.output_targets[target_key]["property"]
                    pred = out[target_key][prop]
            # TODO clean up this logic to reconstruct a tensor from its predicted decomposition
            elif "decomposition" in self.output_targets[target_key]:
                _max_rank = 0
                for subtarget_key in self.output_targets[target_key]["decomposition"]:
                    _max_rank = max(
                        _max_rank,
                        self.output_targets[subtarget_key]["irrep_dim"],
                    )

                pred_irreps = torch.zeros(
                    (batch_size, irreps_sum(_max_rank)), device=self.device
                )

                for subtarget_key in self.output_targets[target_key]["decomposition"]:
                    irreps = self.output_targets[subtarget_key]["irrep_dim"]
                    _pred = self._denorm_preds(subtarget_key, out[subtarget_key], batch)

                    ## Fill in the corresponding irreps prediction
                    ## Reshape irrep prediction to (batch_size, irrep_dim)
                    pred_irreps[
                        :,
                        max(0, irreps_sum(irreps - 1)) : irreps_sum(irreps),
                    ] = _pred.view(batch_size, -1)

                pred = torch.einsum(
                    "ba, cb->ca",
                    cg_change_mat(_max_rank, self.device),
                    pred_irreps,
                )
            else:
                raise AttributeError(
                    f"Output target: '{target_key}', not found in model outputs: {list(out.keys())}"
                )

            ### not all models are consistent with the output shape
            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_key]["level"] == "atom":
                pred = pred.view(num_atoms_in_batch, -1)
            else:
                pred = pred.view(batch_size, -1)
            outputs[target_key] = pred

        return outputs

    def _compute_loss(self, out, batch) -> torch.Tensor:
        batch_size = batch.natoms.numel()
        fixed = batch.fixed
        mask = fixed == 0

        loss = []
        for loss_fn in self.loss_functions:
            target_name, loss_info = loss_fn

            target = batch[target_name]
            pred = out[target_name]
            natoms = batch.natoms
            natoms = torch.repeat_interleave(natoms, natoms)

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["train_on_free_atoms"]
            ):
                target = target[mask]
                pred = pred[mask]
                natoms = natoms[mask]

            num_atoms_in_batch = natoms.numel()

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            # to keep the loss coefficient weights balanced we remove linear references
            # subtract element references from target data

            if target_name in self.elementrefs:
                target = self.elementrefs[target_name].dereference(target, batch)
            # normalize the targets data
            if target_name in self.normalizers:
                target = self.normalizers[target_name].norm(target)

            extra_input = dict()
            if target_name == 'polarization':
                # MAKE SURE you use DDPLossExt(), not DDPLoss()
                extra_input["cell"] = batch["cell"]

            mult = loss_info["coefficient"]
            loss.append(
                mult
                * loss_info["fn"](
                    pred,
                    target,
                    natoms=batch.natoms,
                    **extra_input
                )
            )

        # Sanity check to make sure the compute graph is correct.
        for lc in loss:
            assert hasattr(lc, "grad_fn")
        return sum(loss)

    def _compute_metrics(self, out, batch, evaluator, metrics=None):
        if metrics is None:
            metrics = {}
        # this function changes the values in the out dictionary,
        # make a copy instead of changing them in the callers version
        out = {k: v.clone() for k, v in out.items()}

        natoms = batch.natoms
        batch_size = natoms.numel()

        ### Retrieve free atoms
        fixed = batch.fixed
        mask = fixed == 0

        s_idx = 0
        natoms_free = []
        for _natoms in natoms:
            natoms_free.append(torch.sum(mask[s_idx : s_idx + _natoms]).item())
            s_idx += _natoms
        natoms = torch.LongTensor(natoms_free).to(self.device)

        targets = {}
        for target_name in self.output_targets:
            target = batch[target_name]
            num_atoms_in_batch = batch.natoms.sum()

            if (
                self.output_targets[target_name]["level"] == "atom"
                and self.output_targets[target_name]["eval_on_free_atoms"]
            ):
                target = target[mask]
                out[target_name] = out[target_name][mask]
                num_atoms_in_batch = natoms.sum()

            ### reshape accordingly: num_atoms_in_batch, -1 or num_systems_in_batch, -1
            if self.output_targets[target_name]["level"] == "atom":
                target = target.view(num_atoms_in_batch, -1)
            else:
                target = target.view(batch_size, -1)

            out[target_name] = self._denorm_preds(target_name, out[target_name], batch)
            targets[target_name] = target

        targets["natoms"] = natoms
        out["natoms"] = natoms

        # add all other tensor properties too, but filter out the ones that are changed above
        for key in filter(
            lambda k: k not in [*list(self.output_targets.keys()), "natoms"]
            and isinstance(batch[k], torch.Tensor),
            batch.keys(),
        ):
            targets[key] = batch[key].to(self.device)
            out[key] = targets[key]

        return evaluator.eval(out, targets, prev_metrics=metrics)

    # Takes in a new data source and generates predictions on it.
    # @torch.no_grad #temporarily disabled due to Calculator. to be fixed.
    def predict(
        self,
        data_loader,
        per_image: bool = True,
        results_file: str | None = None,
        disable_tqdm: bool = False,
    ):
        if self.is_debug and per_image:
            raise FileNotFoundError("Predictions require debug mode to be turned off.")

        ensure_fitted(self._unwrapped_model, warn=True)

        if distutils.is_master() and not disable_tqdm:
            logging.info("Predicting on test.")
        assert isinstance(
            data_loader,
            (
                torch.utils.data.dataloader.DataLoader,
                torch_geometric.data.Batch,
            ),
        )
        rank = distutils.get_rank()

        if isinstance(data_loader, torch_geometric.data.Batch):
            ## temporary generate graph before prediction if input is batch.
            ## this is needed because we generate graph in collater instaed of in model forward.
            from GGNN.common.utils import generate_graph, generate_graph_nvalchemi

            if "pbc" not in data_loader:
                data_loader.pbc = torch.ones(
                    3 * (data_loader.batch.max() + 1),
                    dtype=torch.bool,
                    device=self.device,
                )
            data_loader = data_loader.to(self.device)
            edge_index, distance_vec, cell_offsets, _ = generate_graph_nvalchemi(
                data_loader,
                self.config["model"]["cutoff"],
            )
            data_loader.edge_index = edge_index
            data_loader.cell_offsets = cell_offsets
            data_loader.pbc_shift = data_loader.cell_offsets
            data_loader.edge_vec = distance_vec
            data_loader.num_atoms = data_loader.natoms
            data_loader.cell_volume = torch.linalg.det(data_loader.cell)
            data_loader = [data_loader]

        self.model.eval()
        if self.ema is not None:
            self.ema.store()
            self.ema.copy_to()

        predictions = defaultdict(list)
        for _, batch in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            position=rank,
            desc=f"device {rank}",
            disable=disable_tqdm,
        ):
            with torch.amp.autocast(
                device_type="cuda", enabled=self.scaler is not None
            ):
                out = self._forward(batch)

            for target_key in self.config["outputs"]:
                pred = self._denorm_preds(target_key, out[target_key], batch)

                if per_image:
                    ### Save outputs in desired precision, default float16
                    if (
                        self.config["outputs"][target_key].get(
                            "prediction_dtype", "float16"
                        )
                        == "float32"
                        or self.config["task"].get("prediction_dtype", "float16")
                        == "float32"
                        or self.config["task"].get("dataset", "lmdb") == "oc22_lmdb"
                    ):
                        dtype = torch.float32
                    else:
                        dtype = torch.float16

                    pred = pred.detach().cpu().to(dtype)

                    ### Split predictions into per-image predictions
                    if self.config["outputs"][target_key]["level"] == "atom":
                        batch_natoms = batch.natoms
                        batch_fixed = batch.fixed
                        per_image_pred = torch.split(pred, batch_natoms.tolist())

                        ### Save out only free atom, EvalAI does not need fixed atoms
                        _per_image_fixed = torch.split(
                            batch_fixed, batch_natoms.tolist()
                        )
                        _per_image_free_preds = [
                            _pred[(fixed == 0).tolist()].numpy()
                            for _pred, fixed in zip(per_image_pred, _per_image_fixed)
                        ]
                        _chunk_idx = np.array(
                            [free_pred.shape[0] for free_pred in _per_image_free_preds]
                        )
                        per_image_pred = _per_image_free_preds
                    ### Assumes system level properties are of the same dimension
                    else:
                        per_image_pred = pred.numpy()
                        _chunk_idx = None

                    predictions[f"{target_key}"].extend(per_image_pred)
                    ### Backwards compatibility, retain 'chunk_idx' for forces.
                    if _chunk_idx is not None:
                        if target_key == "forces":
                            predictions["chunk_idx"].extend(_chunk_idx)
                        else:
                            predictions[f"{target_key}_chunk_idx"].extend(_chunk_idx)
                else:
                    predictions[f"{target_key}"] = pred.detach()

            if not per_image:
                return predictions

            ### Get unique system identifiers
            sids = (
                batch.sid.tolist() if isinstance(batch.sid, torch.Tensor) else batch.sid
            )
            ## Support naming structure for OC20 S2EF
            if "fid" in batch:
                fids = (
                    batch.fid.tolist()
                    if isinstance(batch.fid, torch.Tensor)
                    else batch.fid
                )
                systemids = [f"{sid}_{fid}" for sid, fid in zip(sids, fids)]
            else:
                systemids = [f"{sid}" for sid in sids]

            predictions["ids"].extend(systemids)

        self.save_results(predictions, results_file)

        if self.ema:
            self.ema.restore()

        return predictions

