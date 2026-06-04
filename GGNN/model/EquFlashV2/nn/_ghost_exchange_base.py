import torch


class GhostExchangeModule(torch.nn.Module):
    """Base class for ghost atom exchange modules."""

    def __init__(
        self,
    ):
        super().__init__()

    def forward(
        self,
        data,
        ghost_included: bool,
    ):
        raise NotImplementedError("Subclasses must implement forward method")


class NoOpGhostExchangeModule(GhostExchangeModule):
    """Base ghost exchange module that performs a no-op."""

    def forward(
        self,
        data,
        ghost_included: bool,
    ):
        return data


# NOTE: can't use custom ops https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html#python-custom-ops-tutorial
# because of complications with `lmp_data` type and PyTorch custom ops registration system


class LAMMPSMLIAPGhostExchangeOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        node_features, lmp_data = args
        original_shape = node_features.shape
        node_features_flat = node_features.view(node_features.size(0), -1)
        out_flat = torch.empty_like(node_features_flat)
        lmp_data.forward_exchange(node_features_flat, out_flat, out_flat.size(-1))

        # save for backward
        ctx.original_shape = original_shape
        ctx.lmp_data = lmp_data

        return out_flat.view(original_shape)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output_flat = grad_output.view(grad_output.size(0), -1)
        gout_flat = torch.empty_like(grad_output_flat)
        ctx.lmp_data.reverse_exchange(grad_output_flat, gout_flat, gout_flat.size(-1))
        return gout_flat.view(ctx.original_shape), None


class LAMMPSMLIAPGhostExchangeModule(GhostExchangeModule):
    """LAMMPS ML-IAP ghost atom exchange module."""

    def forward(self, node_features, lmp_data, ghost_included=False):

        if ghost_included:
            local_node_features = torch.narrow(node_features, 0, 0, lmp_data.nlocal)
        else:
            local_node_features = node_features
        num_ghost_atoms = lmp_data.ntotal - lmp_data.nlocal
        ghost_zeros = torch.zeros(
            (num_ghost_atoms,) + node_features.shape[1:],
            dtype=node_features.dtype,
            device=node_features.device,
        )

        prepared_node_features = torch.cat((local_node_features, ghost_zeros), dim=0)

        # perform LAMMPS exchange
        x = LAMMPSMLIAPGhostExchangeOp.apply(prepared_node_features, lmp_data)

        return x
