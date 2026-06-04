import math
import torch



@torch.jit.script
def ShiftedSoftPlus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x) - math.log(2.0)

ACTIVATION = {
    'relu': torch.nn.functional.relu,
    'silu': torch.nn.functional.silu,
    'tanh': torch.tanh,
    'abs': torch.abs,
    'ssp': ShiftedSoftPlus,
    'sigmoid': torch.sigmoid,
    'elu': torch.nn.functional.elu,
}
ACTIVATION_FOR_EVEN = {
    'ssp': ShiftedSoftPlus,
    'silu': torch.nn.functional.silu,
}
ACTIVATION_FOR_ODD = {'tanh': torch.tanh, 'abs': torch.abs}
ACTIVATION_DICT = {'e': ACTIVATION_FOR_EVEN, 'o': ACTIVATION_FOR_ODD}