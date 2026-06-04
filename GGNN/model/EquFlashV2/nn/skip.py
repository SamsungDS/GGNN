import torch
import torch.nn as nn
from e3nn.o3 import Irreps


class SkipId(nn.Module):
    def __init__(self, irreps_in: Irreps, irreps_out: Irreps):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        if self.irreps_in.dim < self.irreps_out.dim:
            self.proj = ProjectUp(irreps_in, irreps_out)
        elif self.irreps_in.dim > self.irreps_out.dim:
            self.proj = ProjectDown(irreps_in, irreps_out)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x)


class ProjectUp(nn.Module):
    def __init__(self, irreps_in: Irreps, irreps_out: Irreps):
        # """
        # Works when irreps_in < irreps_out find spot that irreps_in \subset irreps_out
        # """
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        s = 0
        idxs = []
        for mul_ir in irreps_in:
            for mul_ir_target in irreps_out:
                e = s + mul_ir_target.dim
                if mul_ir == mul_ir_target:
                    idxs.append(torch.arange(s, e))
                    break
                s = e
            if s == irreps_out.dim:
                assert ()
        idx = torch.cat(idxs, dim=0)
        self.register_buffer("idx", idx, persistent=False)

    def forward(self, x):
        x_res = torch.zeros(
            (x.shape[0], self.irreps_out.dim),
            dtype=x.dtype,
            device=x.device,
        )
        x_res[:, self.idx] = x
        return x_res


class ProjectDown(nn.Module):
    def __init__(self, irreps_in: Irreps, irreps_out: Irreps):
        # """
        # Works when irreps_in > irreps_out find spot that irreps_in \subset irreps_out
        # """
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        s = 0
        idxs = []
        for mul_ir_target in irreps_out:
            for mul_ir in irreps_in:
                e = s + mul_ir.dim
                if mul_ir == mul_ir_target:
                    idxs.append(torch.arange(s, e))
                    break
                s = e
            if s == irreps_out.dim:
                assert ()
        idx = torch.cat(idxs, dim=0)
        self.register_buffer("idx", idx, persistent=False)

    def forward(self, x):
        return x[:, self.idx]
