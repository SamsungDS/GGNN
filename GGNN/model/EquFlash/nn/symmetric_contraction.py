import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
import torch.nn as nn
from e3nn.o3 import Irreps, Linear
from e3nn.util.jit import compile_mode


@compile_mode("script")
class SymmetricContraction(nn.Module):
    """
    wrapper class of e3nn Linear to operate on AtomGraphData
    """

    def __init__(self, irreps_in: Irreps, contraction_degree=3):
        super().__init__()
        self.sym_contraction = cuet.SymmetricContraction(
            cue.Irreps("O3", irreps_in),
            cue.Irreps("O3", irreps_in),
            layout_in=cue.mul_ir,
            layout_out=cue.mul_ir,
            contraction_degree=contraction_degree,
            num_elements=1,
            original_mace=True,
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
        )

    def forward(self, x, y=None):
        y = torch.zeros(x.shape[0], device=x.device,dtype=torch.int)
        z = self.sym_contraction(x, y)
        return z
