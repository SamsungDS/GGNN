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
import logging
from typing import TYPE_CHECKING

from submitit import AutoExecutor
from submitit.helpers import Checkpointable, DelayedSubmission
from torch.distributed.launcher.api import LaunchConfig, elastic_launch

from fairchem.core.common import distutils
from fairchem.core.common.flags import flags
from fairchem.core.common.utils import (
    build_config,
    create_grid,
    save_experiment_log,
    setup_logging,
)
from GGNN.common.utils import new_trainer_context

if TYPE_CHECKING:
    import argparse


class Runner(Checkpointable):
    def __init__(self) -> None:
        self.config = None

    def __call__(self, config: dict) -> None:
        with new_trainer_context(config=config) as ctx:
            self.config = ctx.config
            self.task = ctx.task
            self.trainer = ctx.trainer
            self.task.setup(self.trainer)
            self.task.run()

    def checkpoint(self, *args, **kwargs):
        new_runner = Runner()
        self.trainer.save(checkpoint_file="checkpoint.pt", training_state=True)
        self.config["checkpoint"] = self.task.chkpt_path
        self.config["timestamp_id"] = self.trainer.timestamp_id
        if self.trainer.logger is not None:
            self.trainer.logger.mark_preempting()
        logging.info(
            f'Checkpointing callback is triggered, checkpoint saved to: {self.config["checkpoint"]}, timestamp_id: {self.config["timestamp_id"]}'
        )
        return DelayedSubmission(new_runner, self.config)


def runner_wrapper(config: dict):
    Runner()(config)


def main(
    args: argparse.Namespace | None = None, override_args: list[str] | None = None
):
    """Run the main fairchem program."""
    setup_logging()

    if args is None:
        parser: argparse.ArgumentParser = flags.get_parser()
        args, override_args = parser.parse_known_args()

    assert (
        args.num_gpus > 0
    ), "num_gpus is used to determine number ranks, so it must be at least 1"
    config = build_config(args, override_args)

    if args.submit:  # Run on cluster
        slurm_add_params = config.get("slurm", None)  # additional slurm arguments
        configs = create_grid(config, args.sweep_yml) if args.sweep_yml else [config]

        logging.info(f"Submitting {len(configs)} jobs")
        executor = AutoExecutor(folder=args.logdir / "%j", slurm_max_num_timeout=3)
        executor.update_parameters(
            name=args.identifier,
            mem_gb=args.slurm_mem,
            timeout_min=args.slurm_timeout * 60,
            slurm_partition=args.slurm_partition,
            gpus_per_node=args.num_gpus,
            cpus_per_task=(config["optim"]["num_workers"] + 1),
            tasks_per_node=args.num_gpus,
            nodes=args.num_nodes,
            slurm_additional_parameters=slurm_add_params,
            slurm_qos=args.slurm_qos,
            slurm_account=args.slurm_account,
        )
        for config in configs:
            config["slurm"] = copy.deepcopy(executor.parameters)
            config["slurm"]["folder"] = str(executor.folder)
        jobs = executor.map_array(Runner(), configs)
        logging.info(f"Submitted jobs: {', '.join([job.job_id for job in jobs])}")
        log_file = save_experiment_log(args, jobs, configs)
        logging.info(f"Experiment log saved to: {log_file}")

    else:  # Run locally on a single node, n-processes
        runner_wrapper(config)



if __name__ == "__main__":
    main()
