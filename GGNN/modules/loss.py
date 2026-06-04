import torch
import torch.nn as nn
from fairchem.core.common.registry import registry


@registry.register_loss("voigt6mae")
class voigtMAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred[:,[0,1,2,4,5,8]], target[:,[0,1,2,4,5,8]])