import torch
from e3nn.o3 import Irreps
from math import gcd
from functools import reduce


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def gcd_irreps(irreps: Irreps):
    nums = []
    for mul, ir in irreps:
        nums.append(mul)
    if len(nums) == 0:
        return 0
    return reduce(gcd, nums)
def mul_irreps(a, irreps: Irreps):
    res=[]
    for mul, ir in irreps:
        res.append((int(mul*a),ir))
    return Irreps(res)