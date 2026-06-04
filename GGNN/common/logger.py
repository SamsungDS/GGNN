import os
import logging
import torch
import yaml
import math
from typing import Any

from fairchem.core.common.logger import Logger
from fairchem.core.common.registry import registry

import datetime

def parse_logs(update_dict):
    ts = datetime.datetime.now().timestamp()  
    dt = datetime.datetime.fromtimestamp(ts)  
    formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
    ss = f"{formatted} "
    if "epoch" in update_dict:
        ep = update_dict["epoch"]
        ss += f"epoch {ep:.1f}"
    if "step" in update_dict:
        step = update_dict["step"]
        ss += f" (step {int(step)})"
    if ss != "":
        ss += ":"
    for key, val in update_dict.items():
        if key in ["epoch", "step"]:
            continue

        mse_metric = "mse" in key
        if mse_metric:
            key = key.replace("mse", "rmse")

        if torch.is_tensor(val):
            if mse_metric:
                val = torch.sqrt(val)
            ss += f" {key} {val.item():.5f}"
        elif isinstance(val, float):
            if key == "lr":
                ss += f" {key} {val:.2e}"
            else:
                if mse_metric:
                    val = math.sqrt(val)
                ss += f" {key} {val:.5f}"
        else:
            ss += f" {key} {val}"
    return ss


@registry.register_logger("files")
class FilesLogger(Logger):
    def __init__(self, config):
        super().__init__(config)

        logdir = self.config["cmd"]["logs_dir"] 
        self.log_path = {"train" : os.path.join(logdir, "train.log")}
        self.log_path["val"] = os.path.join(logdir, "val.log")
        self.log_path["test"] = os.path.join(logdir, "test.log")

    def watch(self, model):
        logging.warning(
            "Model gradient logging to files is not supported."
        )
        return False

    def log(self, update_dict, step=None, split=""):
        assert split in ["train", "val", "test"], f"Split {split} is not supported"
        outfile = open(self.log_path[split], 'a')

        ss = parse_logs(update_dict)
        outfile.write(ss + "\n")
        outfile.close()

    def log_plots(self, plots):
        pass

    def mark_preempting(self):
        pass

    def log_model_training_info(self, model=None):
        model_log_path = os.path.join(self.config["cmd"]["logs_dir"], "model_training_info.yml")
        outfile = open(model_log_path, 'w')
        outfile.write(yaml.dump(self.config, default_flow_style=False))
        outfile.write("\n")
        
        if model:
            outfile.write(str(model)+ "\n")
            outfile.write(f"model num of parameters: {model.num_params}\n")
        outfile.close()

    def log_final_metrics(self, table, time=None):
        log_path = os.path.join(self.config["cmd"]["logs_dir"], "final_metrics.log")
        outfile = open(log_path, 'w')
        outfile.write(str(table)+"\n")
        if time:
            outfile.write(f"train() elapsed time: {time:.1f} sec ({time/3600.0:.1f} h)\n")
        outfile.close()
    def log_summary(self, summary_dict: dict[str, Any]) -> None:
        logging.warning("log_summary for Files not supported")

    def log_artifact(self, name: str, type: str, file_location: str) -> None:
        logging.warning("log_artifact for Files not supported")