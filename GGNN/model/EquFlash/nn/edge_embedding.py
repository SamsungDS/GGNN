import math

import torch
import torch.nn as nn
from e3nn.o3 import Irreps, SphericalHarmonics
from e3nn.util.jit import compile_mode

from copy import deepcopy


class BesselBasis(nn.Module):
    """
    f : (*, 1) -> (*, bessel_basis_num)
    """

    def __init__(
        self,
        cutoff_length: float,
        bessel_basis_num: int = 8,
        trainable_coeff: bool = True,
    ):
        super().__init__()
        self.num_basis = bessel_basis_num
        self.cutoff_length = cutoff_length
        self.prefactor = 2.0 / cutoff_length
        self.coeffs = torch.FloatTensor(
            [n * math.pi / cutoff_length for n in range(1, bessel_basis_num + 1)]
        )
        if trainable_coeff:
            self.coeffs = nn.Parameter(self.coeffs)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        ur = r.unsqueeze(-1)  # to fit dimension
        return self.prefactor * torch.sin(self.coeffs * ur) / ur


class PolynomialCutoff(nn.Module):
    """
    f : (*, 1) -> (*, 1)
    https://arxiv.org/pdf/2003.03123.pdf
    """

    def __init__(
        self,
        cutoff_length: float,
        poly_cut_p_value: int = 6,
    ):
        super().__init__()
        p = poly_cut_p_value
        self.r_cut = cutoff_length
        self.p = p
        self.coeff_p0 = (p + 1.0) * (p + 2.0) / 2.0
        self.coeff_p1 = p * (p + 2.0)
        self.coeff_p2 = p * (p + 1.0) / 2.0

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r = r / self.r_cut
        return (
            1
            - self.coeff_p0 * torch.pow(r, self.p)
            + self.coeff_p1 * torch.pow(r, self.p + 1.0)
            - self.coeff_p2 * torch.pow(r, self.p + 2.0)
        )


class XPLORCutoff(nn.Module):
    """
    https://hoomd-blue.readthedocs.io/en/latest/module-md-pair.html
    """

    def __init__(
        self,
        cutoff_length: float,
        cutoff_on: float,
    ):
        super().__init__()
        self.r_on = cutoff_on
        self.r_cut = cutoff_length
        assert self.r_on < self.r_cut

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        r_sq = r * r
        r_on_sq = self.r_on * self.r_on
        r_cut_sq = self.r_cut * self.r_cut
        return torch.where(
            r < self.r_on,
            1.0,
            (r_cut_sq - r_sq) ** 2
            * (r_cut_sq + 2 * r_sq - 3 * r_on_sq)
            / (r_cut_sq - r_on_sq) ** 3,
        )


@compile_mode("script")
class SphericalEncoding(nn.Module):
    def __init__(
        self,
        lmax: int,
        parity: int = -1,
        normalization: str = "component",
        normalize: bool = True,
    ):
        super().__init__()
        self.lmax = lmax
        self.normalization = normalization
        self.irreps_in = Irreps("1x1o") if parity == -1 else Irreps("1x1e")
        self.irreps_out = Irreps.spherical_harmonics(lmax, parity)
        self.sph = SphericalHarmonics(
            self.irreps_out,
            normalize=normalize,
            normalization=normalization,
            irreps_in=self.irreps_in,
        )

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        return self.sph(r)


@compile_mode("script")
class EdgeEmbedding(nn.Module):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        cutoff,
        envelop_dct,
        rbf_dct,
        lmax,
        use_parity,
        normalize_sph=False,
    ):
        super().__init__()
        envelop_dct = deepcopy(envelop_dct)
        rbf_dct = deepcopy(rbf_dct)
        cutoff_dct = {"cutoff_length": cutoff}
        envelop_dct.update(cutoff_dct)
        envelop_name = envelop_dct.pop("cutoff_function_name")
        if envelop_name == "poly_cut":
            env = PolynomialCutoff(**envelop_dct)
        elif envelop_name == "XPLOR":
            env = XPLORCutoff(**envelop_dct)

        parity = -1 if use_parity else 1

        sph = SphericalEncoding(lmax, parity, normalize=normalize_sph)

        rbf_dct.update(cutoff_dct)
        rbf_name = rbf_dct.pop("radial_basis_name")
        if rbf_name == "bessel":
            rbf = BesselBasis(**rbf_dct)

        self.basis_function = rbf
        self.cutoff_function = env
        self.spherical = sph

    def forward(self, data):
        rvec = data["edge_vec"]
        r = torch.linalg.norm(rvec, dim=-1)
        data["edge_length"] = r
        data["edge_embedding"] = self.basis_function(r) * self.cutoff_function(
            r
        ).unsqueeze(-1)
        data["edge_attr"] = self.spherical(rvec)

        return data


@compile_mode("script")
class EdgeEmbeddingField(nn.Module):
    """
    embedding layer of |r| by
    RadialBasis(|r|)*CutOff(|r|)
    f : (N_edge) -> (N_edge, basis_num)
    """

    def __init__(
        self,
        cutoff,
        envelop_dct,
        rbf_dct,
        lmax,
        use_parity,
        normalize_sph=False,
    ):
        super().__init__()
        envelop_dct = deepcopy(envelop_dct)
        rbf_dct = deepcopy(rbf_dct)
        cutoff_dct = {"cutoff_length": cutoff}
        envelop_dct.update(cutoff_dct)
        envelop_name = envelop_dct.pop("cutoff_function_name")
        if envelop_name == "poly_cut":
            env = PolynomialCutoff(**envelop_dct)
        elif envelop_name == "XPLOR":
            env = XPLORCutoff(**envelop_dct)

        parity = -1 if use_parity else 1

        sph = SphericalEncoding(lmax, parity, normalize=normalize_sph)

        rbf_dct.update(cutoff_dct)
        rbf_name = rbf_dct.pop("radial_basis_name")
        if rbf_name == "bessel":
            rbf = BesselBasis(**rbf_dct)

        self.basis_function = rbf
        self.cutoff_function = env
        self.spherical = sph

        #############
        # for now, identical to edge sph lmax, parity
        sph_field = SphericalEncoding(lmax, parity, normalize=False)
        self.spherical_field = sph_field

    def forward(self, data):
        rvec = data["edge_vec"]
        fvec = data["edge_field"]
        r = torch.linalg.norm(rvec, dim=-1)
        data["edge_length"] = r
        data["edge_embedding"] = self.basis_function(r) * self.cutoff_function(
            r
        ).unsqueeze(-1)

        data["edge_attr"] = torch.cat(
            [self.spherical(rvec), self.spherical_field(fvec)], dim=1
        )

        return data
