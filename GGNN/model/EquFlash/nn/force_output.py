import torch
import torch.nn as nn
from e3nn.util.jit import compile_mode


from .util import broadcast


@compile_mode("script")
class ForceOutput(nn.Module):
    """
    works when pos.requires_grad_ is True
    """

    def __init__(
        self,
    ):
        super().__init__()

    def get_grad_key(self):
        return self.key_pos

    def forward(self, batch):
        pos_tensor = [batch.pos]
        energy = [(batch.total_energy).sum()]

        # `materialize_grads` not supported in low version of pytorch
        # Also can not be deployed when using it.
        # But not using it makes problem in
        # force/stress inference in sparse systems
        # TODO: use it only in sevennet_calculator?
        grad = torch.autograd.grad(
            energy,
            pos_tensor,
            create_graph=self.training,
            allow_unused=True,
            # materialize_grads=True,
        )[0]

        # For torchscript
        if grad is not None:
            batch.forces = torch.neg(grad)
        return batch


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
        tot_num = batch["batch"].shape[
            0
        ]    # an integer initialized in preprocess
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

            stress = sout / cell_volume.unsqueeze(-1)

        return forces, stress

@compile_mode("script")
class PolOutput(nn.Module):
    """
    Compute polarization
    """

    def __init__(
        self,
    ):
        super().__init__()
        self._is_batch_data = True

    def get_grad_key(self):
        return self.key_edge

    def forward(self, batch):
        field = batch["field"]
        energy = [(batch["total_energy"]).sum()]

        grad = torch.autograd.grad(
            energy, field, create_graph=True, allow_unused=True, retain_graph=True
        )
        pol=torch.neg(grad[0])

        return pol

@compile_mode("script")
class BECOutputfromPol(nn.Module):
    """
    works when pos.requires_grad_ is True
    """

    def __init__(
        self,
    ):
        super().__init__()

    def get_grad_key(self):
        return self.key_pos

    def forward(self, batch):
        #pos_tensor = [batch.pos]
        tot_num = batch["batch"].shape[0]    # an integer initialized in preprocess
        rij = batch["edge_vec"]
        edge_idx = batch["edge_index"]
        polarization = batch["total_polarization"].sum(dim=0)

        grad0 = torch.autograd.grad(
            polarization[0],
            [rij],
            create_graph=self.training,
            allow_unused=True,
            retain_graph=True,
            # materialize_grads=True,
        )[0]
        grad1 = torch.autograd.grad(
            polarization[1],
            [rij],
            create_graph=self.training,
            allow_unused=True,
            retain_graph=True,
            # materialize_grads=True,
        )[0]
        grad2 = torch.autograd.grad(
            polarization[2],
            [rij],
            create_graph=self.training,
            allow_unused=True,
            retain_graph=True,
            # materialize_grads=True,
        )[0]

        grad = [grad0, grad1, grad2]
        bec=[]


        for i in range(3):
            becij = grad[i]

            if becij is not None:
                # compute force
                pbec = torch.zeros(
                    tot_num, 3, dtype=becij.dtype, device=becij.device
                )  # type:ignore
                nbec = torch.zeros(
                    tot_num, 3, dtype=becij.dtype, device=becij.device
                )  # type:ignore
                _edge_src = broadcast(edge_idx[0], becij, 0)
                _edge_dst = broadcast(edge_idx[1], becij, 0)
                pbec.scatter_reduce_(0, _edge_src, becij, reduce="sum")
                nbec.scatter_reduce_(0, _edge_dst, becij, reduce="sum")
                bec.append(-pbec + nbec)


        # `materialize_grads` not supported in low version of pytorch
        # Also can not be deployed when using it.
        # But not using it makes problem in
        # force/stress inference in sparse systems
        # TODO: use it only in sevennet_calculator?
        bec=torch.stack(bec, dim=1)

        # For torchscript
        if grad is not None:
            #batch.polarization = grad
            return bec