import torch
import torch.nn as nn
from e3nn.o3 import Irreps


class EquivariantRMSNormArraySphericalHarmonicsV2(nn.Module):
    """
    1. Normalize across all m components from degrees L >= 0.
    2. Expand weights and multiply with normalized feature to prevent slicing and concatenation.
    """

    def __init__(
        self,
        irreps: Irreps,
        eps: float = 1e-5,
        affine: bool = True,
        normalization: str = "component",
        centering: bool = True,
        std_balance_degrees: bool = True,
    ):
        super().__init__()
        self.irreps = irreps
        self.lmax = max([ir.l for (mul, ir) in self.irreps])
        self.eps = eps
        self.affine = affine
        self.centering = centering
        self.std_balance_degrees = std_balance_degrees

        # for L >= 0
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.irreps.num_irreps))
            affine_weight_idx = torch.zeros(self.irreps.dim, dtype=torch.int)
            idxChannel = 0
            istart = 0
            iend = 0
            for mul, ir in self.irreps:
                iend = istart + mul * ir.dim
                idx = (
                    torch.arange(idxChannel, idxChannel + mul)
                    .repeat((ir.dim, 1))
                    .reshape(-1)
                )

                affine_weight_idx[istart:iend] = idx
                istart = iend
                idxChannel += mul
            self.register_buffer(
                "affine_weight_idx", affine_weight_idx, persistent=False
            )
            if self.centering:
                self.affine_bias = nn.Parameter(
                    torch.zeros(
                        sum([mul if ir == (0, 1) else 0 for (mul, ir) in self.irreps])
                    )
                )
                affine_bias_idx = torch.zeros(
                    self.affine_bias.shape[0], dtype=torch.int
                )
                idxChannel = 0
                istart = 0
                iend = 0
                for mul, ir in self.irreps:
                    iend = istart + mul * ir.dim
                    if ir == (0, 1):
                        affine_bias_idx[idxChannel : idxChannel + mul] = torch.arange(
                            istart, iend
                        )
                        idxChannel += mul
                    istart = iend
                self.register_buffer("affine_bias_idx", affine_bias_idx)
            else:
                self.register_parameter("affine_bias", None)
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)

        assert normalization in ["norm", "component"]
        self.normalization = normalization

        if self.std_balance_degrees:
            balance_degree_weight = torch.zeros(self.irreps.dim)
            istart = 0
            iend = 0
            for mul, ir in self.irreps:
                length = ir.dim
                iend = istart + mul * ir.dim
                balance_degree_weight[istart:iend] = 1.0 / length / mul
                istart = iend
            balance_degree_weight = balance_degree_weight / (self.lmax + 1)
            self.register_buffer("balance_degree_weight", balance_degree_weight)
        else:
            self.balance_degree_weight = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(irreps={self.irreps}, eps={self.eps}, centering={self.centering}, std_balance_degrees={self.std_balance_degrees})"

    # @torch.autocast("cuda", enabled=False)
    def forward(self, x_):
        """ """
        x = x_.clone()

        if self.centering:
            x_l0 = x[:, self.affine_bias_idx]
            x_l0_mean = x_l0.mean(dim=1, keepdim=True)  # [N, 1, 1]
            x_l0 = x_l0 - x_l0_mean
            x[:, self.affine_bias_idx] = x_l0

        # for L >= 0
        if self.normalization == "norm":
            raise NotImplementedError  # feature_norm = feature.pow(2).sum(dim=1, keepdim=True)  # [N, 1, C]
        elif self.normalization == "component":
            if self.std_balance_degrees:
                x_norm = x.pow(2)  # [N, (L_max + 1)**2, C]
                x_norm = x_norm * self.balance_degree_weight

                # torch.einsum(
                # "nic, ia -> nac", feature_norm, self.balance_degree_weight
                # )  # [N, 1, C]
            else:
                raise NotImplementedError  # feature_norm = feature.pow(2).mean(dim=1, keepdim=True)  # [N, 1, C]

        x_norm = torch.sum(x_norm, dim=1, keepdim=True)  # [N, 1, 1]
        x_norm = (x_norm + self.eps).pow(-0.5)

        if self.affine:
            weight = self.affine_weight[self.affine_weight_idx]
            x_norm = x_norm * weight

        out = x * x_norm

        if self.affine and self.centering:
            feature_l0 = out[:, self.affine_bias_idx]
            out[:, self.affine_bias_idx] = feature_l0 + self.affine_bias

        return out

