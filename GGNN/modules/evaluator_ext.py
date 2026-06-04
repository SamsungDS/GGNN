from __future__ import annotations

from typing import TYPE_CHECKING
import math

import torch

from fairchem.core.modules.evaluator import Evaluator, metrics_dict, NONE_SLICE
import fairchem.core.modules.evaluator as base_metrics

if TYPE_CHECKING:
    from collections.abc import Hashable

"""
Oct 28, 2025
This module is an extension of the metric evaluator code in fairchem-core (v1.10.0).

Here we implement:

(1) a subclass EvaluatorExt, inherited from fairchem.core.modules.evaluator.Evaluator

(2) a 'METRIC_REGISTRY' organizing all metric functions defined here and in fairchem

(3) any desired custom metric functions not already in fairchem

REF:
https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.10.0/src/fairchem/core/modules/evaluator.py
"""

# --------------------------------------
# Custom metric functions
#
# Here we implement metrics similar to custom loss functions in
# GGNN/modules/loss_ext.py
# --------------------------------------

@metrics_dict
def voigt6mae_sym(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:
    # ASSUME: target[key], prediction[key] are of shape [m, 9]. Meant for flattened symmetric cartesian tensors.
    assert prediction[key].shape[1] == 9, f"Expect pred to have shape (*,9), got {prediction[key].shape}"
    assert target[key].shape[1] == 9, f"Expect target to have shape (*,9), got {target[key].shape}"

    pred = prediction[key]
    target_mat = target[key].view(-1, 3, 3)
    target_symmat = 0.5 * (target_mat + target_mat.transpose(1,2))     # (1/2) (A + A^T)
    target_sym = target_symmat.view(-1, 9)

    return torch.abs(pred[:,[0,1,2,4,5,8]] - target_sym[:,[0,1,2,4,5,8]])

def minimum_image_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    # This code computes the difference between 'input' and 'target',
    # wrapped to primitive lattice cell.
    # Of course assumes that 'input', 'target' are 3D vectors.

    cell = cell.reshape(-1, 3, 3).to(input.dtype)
    input = input.reshape(-1, 3)
    target = target.reshape(-1, 3)

    assert cell.shape[0] == input.shape[0], (
        f"cell batch {cell.shape[0]} does not match input batch {input.shape[0]}"
    )

    raw_diff = input - target

    # map to fractional coordinates
    cell_inv = torch.linalg.inv(cell)
    frac_diff = torch.einsum(
        "bi, bij -> bj", raw_diff, cell_inv
    )

    # minimum image convention:    place values within [-0.5, +0.5)
    frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

    # map back to cartesian coordinates
    wrapped_diff = torch.einsum(
        "bi, bij -> bj", frac_diff, cell
    )

    return wrapped_diff

@metrics_dict
def folded_cell_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = minimum_image_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def folded_cell_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = minimum_image_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def folded_cell_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = minimum_image_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))

def smooth_minimum_image_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    # This code computes the difference between 'input' and 'target',
    # wrapped to primitive lattice cell, then fixed for derivative discontinuity.
    # Of course assumes that 'input', 'target' are 3D vectors.

    cell = cell.reshape(-1, 3, 3).to(input.dtype)
    input = input.reshape(-1, 3)
    target = target.reshape(-1, 3)

    assert cell.shape[0] == input.shape[0], (
        f"cell batch {cell.shape[0]} does not match input batch {input.shape[0]}"
    )

    raw_diff = input - target

    # map to fractional coordinates
    cell_inv = torch.linalg.inv(cell)
    frac_diff = torch.einsum(
        "bi, bij -> bj", raw_diff, cell_inv
    )

    # minimum image convention:    place values within [-0.5, +0.5)
    frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

    # Now we wrap the fractional coordinates in a smoothing function
    # which will ensure the smoothness of respective loss objectives.

    def smooth_wrapper(residual: torch.Tensor) -> torch.Tensor:

        term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
        term_2 = 1. - torch.cos(2. * torch.pi * residual)     # 1 - cos(2 pi x)
        exp_k = 0.5 * math.log(math.pi / 2.)                  # (1/2) ln (pi/2)    CONSTANT
        term_3 = torch.exp(exp_k * term_2)

        return (term_1 * term_3) / torch.pi

    frac_diff = smooth_wrapper(frac_diff)

    # map back to cartesian coordinates
    wrapped_diff = torch.einsum(
        "bi, bij -> bj", frac_diff, cell
    )

    return wrapped_diff

@metrics_dict
def folded_smooth_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = smooth_minimum_image_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def folded_smooth_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = smooth_minimum_image_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def folded_smooth_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = smooth_minimum_image_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))

def sinusoid_minimum_image_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    # This code computes the difference between 'input' and 'target',
    # wrapped to primitive lattice cell, then fixed for derivative discontinuity.
    # Of course assumes that 'input', 'target' are 3D vectors.

    cell = cell.reshape(-1, 3, 3).to(input.dtype)
    input = input.reshape(-1, 3)
    target = target.reshape(-1, 3)

    assert cell.shape[0] == input.shape[0], (
        f"cell batch {cell.shape[0]} does not match input batch {input.shape[0]}"
    )

    raw_diff = input - target

    # map to fractional coordinates
    cell_inv = torch.linalg.inv(cell)
    frac_diff = torch.einsum(
        "bi, bij -> bj", raw_diff, cell_inv
    )

    # minimum image convention:    place values within [-0.5, +0.5)
    frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

    # Now we wrap the fractional coordinates in a smoothing function
    # which will ensure the smoothness of respective loss objectives.

    def sine_wrapper(residual: torch.Tensor) -> torch.Tensor:

        term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
        return term_1 / torch.pi

    frac_diff = sine_wrapper(frac_diff)

    # map back to cartesian coordinates
    wrapped_diff = torch.einsum(
        "bi, bij -> bj", frac_diff, cell
    )

    return wrapped_diff

@metrics_dict
def folded_sine_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = sinusoid_minimum_image_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def folded_sine_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = sinusoid_minimum_image_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def folded_sine_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = sinusoid_minimum_image_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))

# Nov. 21, 2025
#
# Below are implementations of metrics for lattice-periodic vector targets,
# expressed in terms of fractional, not cartesian, coordinates.
#
# HELPER FUNCTIONS:
#
# frac_coord_diff:             convert cartesian differences of model prediction & label
#                              to fractional coordinates
#
# frac_wrapped_diff:           wrap fractional coordinates to [-0.5, +0.5]      (f_fold, sawtooth)
#
# frac_smooth_wrapped_diff:    a smoothed variant of wrapped fractional coordinates   (f_alt)
#
# frac_sinusoid_wrapped_diff:  a sinusoidal mapping of wrapped fractional coordinates (f_sin)
#
#
# METRICS:
#
# (1) (frac_coord_diff)              frac_mae, frac_mse, frac_l2mae     (NOT implemented as loss)
# 
# (2) (frac_wrapped_diff)            frac_folded_mae, frac_folded_mse, frac_folded_l2mae
#
# (3) (frac_smooth_wrapped_diff)     frac_smooth_mae, frac_smooth_mse, frac_smooth_l2mae
#
# (4) (frac_sinusoid_wrapped_diff)   frac_sine_mae, frac_sine_mse, frac_sine_l2mae

def frac_coord_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    cell = cell.reshape(-1, 3, 3).to(input.dtype)
    input = input.reshape(-1, 3)
    target = target.reshape(-1, 3)

    assert cell.shape[0] == input.shape[0], (
        f"cell batch {cell.shape[0]} does not match input batch {input.shape[0]}"
    )

    raw_diff = input - target

    # map to fractional coordinates
    cell_inv = torch.linalg.inv(cell)
    frac_diff = torch.einsum(
        "bi, bij -> bj", raw_diff, cell_inv
    )

    return frac_diff

def frac_wrapped_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    frac_diff = frac_coord_diff(input, target, cell)

    # minimum image convention:    place values within [-0.5, +0.5)
    frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

    # HERE WE DO NOT MAP BACK TO CARTESIAN COORDINATES!!!!
    return frac_diff

def frac_smooth_wrapped_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    frac_diff = frac_coord_diff(input, target, cell)

    # minimum image convention:    place values within [-0.5, +0.5)
    frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

    def smooth_wrapper(residual: torch.Tensor) -> torch.Tensor:

        term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
        term_2 = 1. - torch.cos(2. * torch.pi * residual)     # 1 - cos(2 pi x)
        exp_k = 0.5 * math.log(math.pi / 2.)                  # (1/2) ln (pi/2)    CONSTANT
        term_3 = torch.exp(exp_k * term_2)

        return (term_1 * term_3) / torch.pi

    frac_diff = smooth_wrapper(frac_diff)

    # HERE WE DO NOT MAP BACK TO CARTESIAN COORDINATES!!!!
    return frac_diff

def frac_sinusoid_wrapped_diff(
    input: torch.Tensor,
    target: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:

    frac_diff = frac_coord_diff(input, target, cell)

    # minimum image convention:    place values within [-0.5, +0.5)
    frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

    def sine_wrapper(residual: torch.Tensor) -> torch.Tensor:

        term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
        return term_1 / torch.pi

    frac_diff = sine_wrapper(frac_diff)

    # HERE WE DO NOT MAP BACK TO CARTESIAN COORDINATES!!!!
    return frac_diff

@metrics_dict
def frac_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_coord_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def frac_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_coord_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def frac_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_coord_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))

@metrics_dict
def frac_folded_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_wrapped_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def frac_folded_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_wrapped_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def frac_folded_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_wrapped_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))

@metrics_dict
def frac_smooth_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_smooth_wrapped_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def frac_smooth_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_smooth_wrapped_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def frac_smooth_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_smooth_wrapped_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))

@metrics_dict
def frac_sine_mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_sinusoid_wrapped_diff(prediction[key], target[key], target["cell"])
    return torch.abs(wrapped_diff)

@metrics_dict
def frac_sine_mse(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_sinusoid_wrapped_diff(prediction[key], target[key], target["cell"])
    return wrapped_diff ** 2

@metrics_dict
def frac_sine_l2mae(
    prediction: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    key: Hashable = NONE_SLICE,
) -> torch.Tensor:

    wrapped_diff = frac_sinusoid_wrapped_diff(prediction[key], target[key], target["cell"])
    return torch.sqrt((wrapped_diff ** 2).sum(dim=-1))


# --------------------------------------
# Metric registry and subclass of Evaluator
# --------------------------------------
 
METRIC_REGISTRY = {}       # maps metric name (str) -> callable function

# Support built-in metrics from fairchem-core
for name in [
    "cosine_similarity",
    "mae",
    "mse",
    "per_atom_mae",
    "per_atom_mse",
    "magnitude_error",
    "forcesx_mae",
    "forcesx_mse",
    "forcesy_mae",
    "forcesy_mse",
    "forcesz_mae",
    "forcesz_mse",
    "energy_forces_within_threshold",
    "energy_within_threshold",
    "average_distance_within_threshold",
    #"rmse"
]:
    METRIC_REGISTRY[name] = getattr(base_metrics, name)

# the function 'rmse' defined in fairchem evaluator.py is actually "l2norm"/"l2mae",
# i.e. Euclidean vector norm for 3D vectors
METRIC_REGISTRY["l2norm"] = getattr(base_metrics, "rmse")
METRIC_REGISTRY["l2mae"] = getattr(base_metrics, "rmse")


# ADD ANY CUSTOM METRIC DEFINED HERE:

METRIC_REGISTRY["voigt6mae_sym"] = voigt6mae_sym

METRIC_REGISTRY["folded_cell_mae"] = folded_cell_mae
METRIC_REGISTRY["folded_cell_mse"] = folded_cell_mse
METRIC_REGISTRY["folded_cell_l2norm"] = folded_cell_l2mae
METRIC_REGISTRY["folded_cell_l2mae"] = folded_cell_l2mae

METRIC_REGISTRY["folded_smooth_mae"] = folded_smooth_mae
METRIC_REGISTRY["folded_smooth_mse"] = folded_smooth_mse
METRIC_REGISTRY["folded_smooth_l2norm"] = folded_smooth_l2mae
METRIC_REGISTRY["folded_smooth_l2mae"] = folded_smooth_l2mae

METRIC_REGISTRY["folded_sine_mae"] = folded_sine_mae
METRIC_REGISTRY["folded_sine_mse"] = folded_sine_mse
METRIC_REGISTRY["folded_sine_l2norm"] = folded_sine_l2mae
METRIC_REGISTRY["folded_sine_l2mae"] = folded_sine_l2mae

METRIC_REGISTRY["frac_mae"] = frac_mae
METRIC_REGISTRY["frac_mse"] = frac_mse
METRIC_REGISTRY["frac_l2norm"] = frac_l2mae
METRIC_REGISTRY["frac_l2mae"] = frac_l2mae

METRIC_REGISTRY["frac_folded_mae"] = frac_folded_mae
METRIC_REGISTRY["frac_folded_mse"] = frac_folded_mse
METRIC_REGISTRY["frac_folded_l2norm"] = frac_folded_l2mae
METRIC_REGISTRY["frac_folded_l2mae"] = frac_folded_l2mae

METRIC_REGISTRY["frac_smooth_mae"] = frac_smooth_mae
METRIC_REGISTRY["frac_smooth_mse"] = frac_smooth_mse
METRIC_REGISTRY["frac_smooth_l2norm"] = frac_smooth_l2mae
METRIC_REGISTRY["frac_smooth_l2mae"] = frac_smooth_l2mae

METRIC_REGISTRY["frac_sine_mae"] = frac_sine_mae
METRIC_REGISTRY["frac_sine_mse"] = frac_sine_mse
METRIC_REGISTRY["frac_sine_l2norm"] = frac_sine_l2mae
METRIC_REGISTRY["frac_sine_l2mae"] = frac_sine_l2mae


class EvaluatorExt(Evaluator):

    def eval(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        prev_metrics: dict | None = None,
    ):
        prev_metrics = prev_metrics or {}
        metrics = prev_metrics

        for target_property in self.target_metrics:
            for fn in self.target_metrics[target_property]:

                func = METRIC_REGISTRY.get(fn)
                if func is None:
                    raise ValueError(
                        f"[EvaluatorExt] Unknown metric function: '{fn}'"
                    )

                metric_name = (
                    f"{target_property}_{fn}"
                    if target_property not in fn and target_property != "misc"
                    else fn
                )

                res = func(prediction, target, target_property)
                metrics = self.update(metric_name, res, metrics)

        return metrics
