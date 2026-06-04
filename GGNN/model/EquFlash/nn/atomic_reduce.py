import torch
import torch.nn as nn
from e3nn.o3 import Irreps,Linear
from typing import Union,List
from .scale import SpeciesWiseRescale

class AtomicReduce(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        shift: Union[float, List[float]],
        scale: Union[float, List[float]],
        type_map,
        use_bias_in_linear: bool,
    ):
        super().__init__()
        hidden_dim = irreps_x[0].dim // 2
        hidden_irreps = Irreps(f"{hidden_dim}x0e")
        self.linear_out1 = Linear(
            irreps_x,
            hidden_irreps,
            biases=use_bias_in_linear,
        )
        self.linear_out2 = Linear(
            hidden_irreps,
            Irreps("1x0e"),
            biases=use_bias_in_linear,
        )
        self.rescale_atomic_energy = SpeciesWiseRescale(
            shift,
            scale,
            type_map=type_map,
            train_shift=False,
            train_scale=False,
        )

    def forward(self, batch):
        x = batch["x"]
        atom_mapper = batch["atom_mapper"]
        x = self.linear_out1(x)
        x = self.linear_out2(x)
        atomic_energy = self.rescale_atomic_energy(x, atom_mapper)
        if("batch" in batch.keys()):
            output = torch.zeros(
                (batch["natoms"].shape[0],),
                dtype=atomic_energy.dtype,
                device=atomic_energy.device,
            )
            output.scatter_reduce_(0, batch["batch"], atomic_energy.squeeze(1), reduce="sum")
        else:
            output = torch.sum(atomic_energy)
        total_energy = output

        batch["total_energy"] = total_energy
        batch["atomic_energy"] = atomic_energy
        return batch


class AtomicReducePol(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        shift: Union[float, List[float]],
        scale: Union[float, List[float]],
        type_map,
        use_bias_in_linear: bool,
    ):
        super().__init__()
        hidden_dim = irreps_x[0].dim // 2
        hidden_irreps = Irreps(f"{hidden_dim}x0e+"+f"{hidden_dim}x1o")
        self.linear_out1 = Linear(
            irreps_x,
            hidden_irreps,
            biases=use_bias_in_linear,
        )
        self.linear_out2 = Linear(
            hidden_irreps,
            Irreps("1x0e+1x1o"),
            biases=use_bias_in_linear,
        )
        self.rescale_atomic_energy = SpeciesWiseRescale(
            shift,
            scale,
            type_map=type_map,
            train_shift=False,
            train_scale=False,
        )

    def forward(self, batch):
        x = batch["x"]
        atom_mapper = batch["atom_mapper"]
        x = self.linear_out1(x)
        x = self.linear_out2(x)
        atomic_energy = self.rescale_atomic_energy(x[:,0:1], atom_mapper)
        atomic_polarization = x[:,1:]
        if("batch" in batch.keys()):
            output = torch.zeros(
                (batch["natoms"].shape[0],),
                dtype=atomic_energy.dtype,
                device=atomic_energy.device,
            )
            output.scatter_reduce_(0, batch["batch"], atomic_energy.squeeze(1), reduce="sum")

            outputpol = torch.zeros(
                (batch["natoms"].shape[0],3),
                dtype=atomic_polarization.dtype,
                device=atomic_polarization.device,
            )
            outputpol.scatter_reduce_(0, batch["batch"].unsqueeze(1).expand(-1,3), atomic_polarization, reduce="sum")
        else:
            output = torch.sum(atomic_energy)
            outputpol = torch.sum(atomic_polarization, dim=0)
        total_energy = output

        batch["total_energy"] = total_energy
        batch["atomic_energy"] = atomic_energy
        batch['atomic_polarization'] = atomic_polarization
        batch['total_polarization'] = outputpol #(batch['atomic_polarization']).sum(dim=0)
        return batch


class AtomicReduceBEC(nn.Module):
    def __init__(
        self,
        irreps_x: Irreps,
        shift: Union[float, List[float]],
        scale: Union[float, List[float]],
        type_map,
        use_bias_in_linear: bool,
    ):
        super().__init__()
        hidden_dim = irreps_x[0].dim // 2
        hidden_irreps = Irreps(f"{hidden_dim*2}x0e+"+f"{hidden_dim}x1e+"+f"{hidden_dim}x2e")
        self.linear_out1 = Linear(
            irreps_x,
            hidden_irreps,
            biases=use_bias_in_linear,
        )
        self.linear_out2 = Linear(
            hidden_irreps,
            Irreps("2x0e+1x1e+1x2e"),
            biases=use_bias_in_linear,
        )
        self.rescale_atomic_energy = SpeciesWiseRescale(
            shift,
            scale,
            type_map=type_map,
            train_shift=False,
            train_scale=False,
        )

    def forward(self, batch):
        x = batch["x"]
        atom_mapper = batch["atom_mapper"]
        x = self.linear_out1(x)
        x = self.linear_out2(x)
        atomic_energy = self.rescale_atomic_energy(x[:,0:1], atom_mapper)
        if("batch" in batch.keys()):
            output = torch.zeros(
                (batch["natoms"].shape[0],),
                dtype=atomic_energy.dtype,
                device=atomic_energy.device,
            )
            output.scatter_reduce_(0, batch["batch"], atomic_energy.squeeze(1), reduce="sum")
        else:
            output = torch.sum(atomic_energy)
        total_energy = output

        batch["total_energy"] = total_energy
        batch["atomic_energy"] = atomic_energy
        batch['bec'] = x[:,1:]
        return batch


# class AtomicReducePol(nn.Module):
#     def __init__(
#         self,
#         irreps_x: Irreps,
#         shift: Union[float, List[float]],
#         scale: Union[float, List[float]],
#         type_map,
#         use_bias_in_linear: bool,
#     ):
#         super().__init__()
#         hidden_dim = irreps_x[0].dim // 2
#         hidden_irreps = Irreps(f"{hidden_dim*2}x0e+"+f"{hidden_dim}x1e+"+f"{hidden_dim}x2e")
#         self.linear_out1 = Linear(
#             irreps_x,
#             hidden_irreps,
#             biases=use_bias_in_linear,
#         )
#         self.linear_out2 = Linear(
#             hidden_irreps,
#             Irreps("2x0e+1x1e+1x2e"),
#             biases=use_bias_in_linear,
#         )
#         self.rescale_atomic_energy = SpeciesWiseRescale(
#             shift,
#             scale,
#             type_map=type_map,
#             train_shift=False,
#             train_scale=False,
#         )

#     def forward(self, batch):
#         x = batch["x"]
#         atom_mapper = batch["atom_mapper"]
#         x = self.linear_out1(x)
#         x = self.linear_out2(x)
#         atomic_energy = self.rescale_atomic_energy(x[:,0:1], atom_mapper)
#         if("batch" in batch.keys()):
#             output = torch.zeros(
#                 (batch["natoms"].shape[0],),
#                 dtype=atomic_energy.dtype,
#                 device=atomic_energy.device,
#             )
#             output.scatter_reduce_(0, batch["batch"], atomic_energy.squeeze(1), reduce="sum")
#         else:
#             output = torch.sum(atomic_energy)
#         total_energy = output

#         batch["total_energy"] = total_energy
#         batch["atomic_energy"] = atomic_energy
#         batch["bec"] = x[:,1:]
#         print(batch["total_energy"])
#         print(batch["bec"])
#         return batch