from __future__ import annotations

import logging
from typing import Literal
import math

import torch
import torch.nn as nn

from fairchem.core.modules.loss import DDPLoss
from fairchem.core.common.registry import registry

# Similar to voigt6mae loss term, but with explicit symmetrization
# imposed to label (cartesian tensor)

@registry.register_loss("voigt6mae_sym")
class voigtMAESymLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"    

    def forward(
        self, 
        pred: torch.Tensor, 
        target: torch.Tensor, 
        natoms: torch.Tensor
    ) -> torch.Tensor:

        # ASSUME: 'target' is of shape [m, 9]
        target_mat = target.view(-1, 3, 3)
        target_symmat = 0.5 * (target_mat + target_mat.transpose(1,2))     # (1/2) (A + A^T)
        target_sym = target_symmat.view(-1, 9)

        return self.loss(pred[:,[0,1,2,4,5,8]], target_sym[:,[0,1,2,4,5,8]])

"""
Oct. 22, 2025
Here we implement MAE and MSE loss terms for
vector quantities which are CELL-PERIODIC.

What matters is not the absolute difference between
input and target, but the difference wrapped within a lattice primitive cell.

REF:
https://github.com/mir-group/allegro-pol/blob/main/allegro_pol/pol_loss.py
"""

class BaseFoldedCellLoss(nn.Module):
    # This implements the actual cell wrapping of the pred-target difference.

    def __init__(self):
        super().__init__()

    def minimum_image_diff(
        self, 
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

        # minimum image convention:    place values within [-0.5, +0.5)
        frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

        # map back to cartesian coordinates
        wrapped_diff = torch.einsum(
            "bi, bij -> bj", frac_diff, cell
        )

        return wrapped_diff


@registry.register_loss("folded_cell_mae")
class FoldedCellLossMAE(BaseFoldedCellLoss):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.minimum_image_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("folded_cell_mse")
class FoldedCellLossMSE(BaseFoldedCellLoss):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.minimum_image_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("folded_cell_l2norm")
@registry.register_loss("folded_cell_l2mae")
class FoldedCellLossL2Norm(BaseFoldedCellLoss):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        assert target.dim() == 2
        assert target.shape[1] != 1

        wrapped_diff = self.minimum_image_diff(input, target, cell)

        return torch.linalg.vector_norm(wrapped_diff, ord=2, dim=-1)


# Nov. 14, 2025
#
# An alternative kind of loss function,
# which does consider lattice periodicity but
# also makes a modification to ensure
# a smooth differentiable parameter landscape.
#
# Meant for vector-valued target which is known to be cell-periodic.

class BaseFoldedSmoothLoss(nn.Module):
    # This implements the actual modified cell wrapping of the pred-target difference.

    def __init__(self):
        super().__init__()

    def smooth_minimum_image_diff(
        self,
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

        # minimum image convention:    place values within [-0.5, +0.5)
        frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

        # Up to this point the operation is identical to BaseFoldedCellLoss.minimum_image_diff().
        # Now we wrap the fractional coordinates in a smoothing function
        # which will ensure the smoothness of respective loss objectives.

        def smooth_wrapper(residual: torch.Tensor) -> torch.Tensor:

            term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
            term_2 = 1. - torch.cos(2. * torch.pi * residual)     # 1 - cos(2 pi x)
            exp_k = 0.5 * math.log(math.pi / 2.)                  # (1/2) ln (pi/2)    CONSTANT
            term_3 = torch.exp(exp_k * term_2)

            return (term_1 * term_3) / torch.pi

        frac_diff = smooth_wrapper(frac_diff)

        # Now map back to cartesian coordinates
        wrapped_diff = torch.einsum(
            "bi, bij -> bj", frac_diff, cell
        )

        return wrapped_diff


@registry.register_loss("folded_smooth_mae")
class FoldedSmoothLossMAE(BaseFoldedSmoothLoss):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.smooth_minimum_image_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("folded_smooth_mse")
class FoldedSmoothLossMSE(BaseFoldedSmoothLoss):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.smooth_minimum_image_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("folded_smooth_l2norm")
@registry.register_loss("folded_smooth_l2mae")
class FoldedSmoothLossL2Norm(BaseFoldedSmoothLoss):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        assert target.dim() == 2
        assert target.shape[1] != 1

        wrapped_diff = self.smooth_minimum_image_diff(input, target, cell)

        return torch.linalg.vector_norm(wrapped_diff, ord=2, dim=-1)


# Nov. 14, 2025
#
# Another alternative kind of loss function,
# this time simplified to be just wrapped using sinusoidal function,
# making it easy to cross between angular branches.
#
# Meant for vector-valued target which is known to be cell-periodic.

class BaseFoldedSineLoss(nn.Module):
    # This implements the actual modified cell wrapping of the pred-target difference.

    def __init__(self):
        super().__init__()

    def sinusoid_minimum_image_diff(
        self,
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

        # minimum image convention:    place values within [-0.5, +0.5)
        frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

        # Up to this point the operation is identical to BaseFoldedCellLoss.minimum_image_diff().
        # Now we wrap the fractional coordinates in a smoothing function
        # which will ensure the smoothness of respective loss objectives.

        def sine_wrapper(residual: torch.Tensor) -> torch.Tensor:

            term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
            return term_1 / torch.pi

        frac_diff = sine_wrapper(frac_diff)

        # Now map back to cartesian coordinates
        wrapped_diff = torch.einsum(
            "bi, bij -> bj", frac_diff, cell
        )

        return wrapped_diff


@registry.register_loss("folded_sine_mae")
class FoldedSineLossMAE(BaseFoldedSineLoss):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.sinusoid_minimum_image_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("folded_sine_mse")
class FoldedSineLossMSE(BaseFoldedSineLoss):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.sinusoid_minimum_image_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("folded_sine_l2norm")
@registry.register_loss("folded_sine_l2mae")
class FoldedSineLossL2Norm(BaseFoldedSineLoss):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        assert target.dim() == 2
        assert target.shape[1] != 1

        wrapped_diff = self.sinusoid_minimum_image_diff(input, target, cell)

        return torch.linalg.vector_norm(wrapped_diff, ord=2, dim=-1)


# Nov. 21, 2025
#
# Below are implementations of losses for lattice-periodic vector targets,
# expressed in terms of fractional, not cartesian, coordinates.
#
# FindFracCoord:    convert cartesian difference of model prediction and label
#                   to fractional coordinates
#
# (not implemented because unnecessary:
#  losses acting directly on fractional coordinates without periodic wrapping)
#
#
# BaseFracFolded ( FindFracCoord ):
#     wrap fractional coordinates to [-0.5, +0.5]             (f_fold, sawtooth)
#     ->   frac_folded_mae, frac_folded_mse, frac_folded_l2mae
#
# BaseFracSmooth ( FindFracCoord ):
#     a smoothed variant of wrapped fractional coordinates    (f_alt)
#     ->   frac_smooth_mae, frac_smooth_mse, frac_smooth_l2mae
#
# BaseFracSine ( FindFracCoord ):
#     a sinusoidal mapping of wrapped fractional coordinates  (f_sin)
#     ->   frac_sine_mae, frac_sine_mse, frac_sine_l2mae

class FindFracCoord(nn.Module):

    def __init__(self):
        super().__init__()

    def frac_coord_diff(
        self, 
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


class BaseFracFolded(FindFracCoord):

    def __init__(self):
        super().__init__()

    def frac_wrapped_diff(
        self, 
        input: torch.Tensor, 
        target: torch.Tensor,
        cell: torch.Tensor,    
    ) -> torch.Tensor:

        frac_diff = self.frac_coord_diff(input, target, cell)

        # minimum image convention:    place values within [-0.5, +0.5)
        frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

        # HERE WE DO NOT MAP BACK TO CARTESIAN COORDINATES!!!!
        return frac_diff

@registry.register_loss("frac_folded_mae")
class FracFoldedLossMAE(BaseFracFolded):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.frac_wrapped_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("frac_folded_mse")
class FracFoldedLossMSE(BaseFracFolded):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.frac_wrapped_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("frac_folded_l2norm")
@registry.register_loss("frac_folded_l2mae")
class FracFoldedLossL2Norm(BaseFracFolded):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        assert target.dim() == 2
        assert target.shape[1] != 1

        wrapped_diff = self.frac_wrapped_diff(input, target, cell)

        return torch.linalg.vector_norm(wrapped_diff, ord=2, dim=-1)


class BaseFracSmooth(FindFracCoord):

    def __init__(self):
        super().__init__()

    def frac_smooth_wrapped_diff(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        cell: torch.Tensor,        
    ) -> torch.Tensor:

        frac_diff = self.frac_coord_diff(input, target, cell)

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

@registry.register_loss("frac_smooth_mae")
class FracSmoothLossMAE(BaseFracSmooth):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.frac_smooth_wrapped_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("frac_smooth_mse")
class FracSmoothLossMSE(BaseFracSmooth):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.frac_smooth_wrapped_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("frac_smooth_l2norm")
@registry.register_loss("frac_smooth_l2mae")
class FracSmoothLossL2Norm(BaseFracSmooth):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        assert target.dim() == 2
        assert target.shape[1] != 1

        wrapped_diff = self.frac_smooth_wrapped_diff(input, target, cell)

        return torch.linalg.vector_norm(wrapped_diff, ord=2, dim=-1)


class BaseFracSine(FindFracCoord):

    def __init__(self):
        super().__init__()

    def frac_sinusoid_wrapped_diff(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        cell: torch.Tensor,        
    ) -> torch.Tensor:

        frac_diff = self.frac_coord_diff(input, target, cell)

        # minimum image convention:    place values within [-0.5, +0.5)
        frac_diff = (frac_diff + 0.5) % 1.0 - 0.5

        def sine_wrapper(residual: torch.Tensor) -> torch.Tensor:

            term_1 = torch.sin(torch.pi * residual)               # sin(pi x)
            return term_1 / torch.pi

        frac_diff = sine_wrapper(frac_diff)

        # HERE WE DO NOT MAP BACK TO CARTESIAN COORDINATES!!!!
        return frac_diff

@registry.register_loss("frac_sine_mae")
class FracSineLossMAE(BaseFracSine):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.frac_sinusoid_wrapped_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("frac_sine_mse")
class FracSineLossMSE(BaseFracSine):

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        wrapped_diff = self.frac_sinusoid_wrapped_diff(input, target, cell)
        dummy_tensor = torch.zeros_like(wrapped_diff)

        return self.loss(wrapped_diff, dummy_tensor)

@registry.register_loss("frac_sine_l2norm")
@registry.register_loss("frac_sine_l2mae")
class FracSineLossL2Norm(BaseFracSine):

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        cell: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:

        assert target.dim() == 2
        assert target.shape[1] != 1

        wrapped_diff = self.frac_sinusoid_wrapped_diff(input, target, cell)

        return torch.linalg.vector_norm(wrapped_diff, ord=2, dim=-1)


#################################################################################################################

class DDPLossExt(DDPLoss):
    """
    Extended DDP-safe loss wrapper that allows additional inputs.
    Works with any loss registered through the fairchem registry, as long as it's callable via:
    `loss_fn(input, target, natoms, **kwargs)`.

    REF:   (based on fairchem-core v1.10.0)
    https://github.com/facebookresearch/fairchem/blob/fairchem_core-1.10.0/src/fairchem/core/modules/loss.py#L83
    """

    def __init__(
        self, 
        loss_name,
        reduction: Literal["mean", "sum"],
    ) -> None:
        super().__init__(loss_name=loss_name, reduction=reduction)

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
        **kwargs,     # catch-all for extra tensors like 'cell', if any
    ) -> torch.Tensor:

        # ensure torch doesn't do any unwanted broadcasting
        assert (
            input.shape == target.shape
        ), f"Mismatched shapes: {input.shape} and {target.shape}"

        # zero out nans, if any
        found_nans_or_infs = not torch.all(input.isfinite())
        if found_nans_or_infs is True:
            logging.warning("Found nans while computing loss")
            input = torch.nan_to_num(input, nan=0.0)

        loss = self.loss_fn(input, target, natoms, **kwargs)     # allows extra input
        return self._reduction(input, loss, natoms)
