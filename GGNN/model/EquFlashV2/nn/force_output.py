import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode

from .util import broadcast

@compile_mode("script")
class ForceStressOutputFromEdge(nn.Module):
    """
    Compute stress and force from edge.
    Used in parallel torchscipt models, and training
    """

    def __init__(
        self,
    ):

        super().__init__()
        self._is_batch_data = True

    def get_grad_key(self):
        return self.key_edge

    def forward(self, batch):
        tot_num = batch["batch"].shape[0]  # an integer initialized in preprocess
        rij = batch["edge_vec"]
        energy = [(batch["total_energy"]).sum()]
        edge_idx = batch["edge_index"]
        grad = torch.autograd.grad(
            energy, [rij], create_graph=self.training, allow_unused=True
        )
        cell_volume = torch.linalg.det(batch["cell"])

        # make grad is not Optional[Tensor]
        fij = grad[0]

        if fij is not None:
            # compute force
            pf = torch.zeros(
                tot_num, 3, dtype=fij.dtype, device=fij.device
            )  # type:ignore
            nf = torch.zeros(
                tot_num, 3, dtype=fij.dtype, device=fij.device
            )  # type:ignore
            _edge_src = broadcast(edge_idx[0], fij, 0)
            _edge_dst = broadcast(edge_idx[1], fij, 0)
            pf.scatter_reduce_(0, _edge_src, fij, reduce="sum")
            nf.scatter_reduce_(0, _edge_dst, fij, reduce="sum")
            forces = pf - nf

            # compute virial
            diag = rij * fij
            s12 = rij[..., 0] * fij[..., 1]
            s23 = rij[..., 1] * fij[..., 2]
            s31 = rij[..., 2] * fij[..., 0]
            # cat last dimension
            _virial = torch.cat(
                [diag, s12.unsqueeze(-1), s23.unsqueeze(-1), s31.unsqueeze(-1)], dim=-1
            )

            _virial = (rij.unsqueeze(-1) * fij.unsqueeze(-2)).reshape(-1, 9)

            _s = torch.zeros(tot_num, 9, dtype=fij.dtype, device=fij.device)
            _edge_dst6 = broadcast(edge_idx[1], _virial, 0)
            _s.scatter_reduce_(0, _edge_dst6, _virial, reduce="sum")

            if self._is_batch_data:
                # nbatch = int(batch.max().cpu().item()) + 1
                sout = torch.zeros(
                    (batch["natoms"].shape[0], 9),
                    dtype=_virial.dtype,
                    device=_virial.device,
                )
                # _batch = broadcast(batch, _s, 0)
                sout.scatter_reduce_(
                    0, batch["batch"].unsqueeze(1).expand(-1, 9), _s, reduce="sum"
                )

            else:
                sout = torch.sum(_s, dim=0)

            sout = (
                (sout.reshape(-1, 3, 3) + sout.reshape(-1, 3, 3).transpose(1, 2)) / 2
            ).reshape(-1, 9)
            stress = sout / cell_volume.unsqueeze(-1)

        return forces, stress
