import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.modules.batchnorm import _BatchNorm


class sync_batch_norm(Function):
    """
    A version of batch normalization that aggregates the activation statistics across all processes.

    This needs to be a custom autograd.Function, because you also need to communicate between processes
    on the backward pass (each activation affects all examples, so loss gradients from all examples affect
    the gradient for each activation).

    For a quick tutorial on torch.autograd.function, see
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, running_mean, running_std, eps: float, momentum: float):
        # Compute statistics, sync statistics, apply them to the input
        # Also, store relevant quantities to be used on the backward pass with `ctx.save_for_backward`
        input_shape = input.shape
        batch_size, num_features = input_shape[:2]
        input = input.view(batch_size, num_features, -1)
        hidden_size = input.size(2)

        # to use torch.distributed primitives only ones, use torch.distributed.all_gather to accumulate
        # all needed tensors
        num_processes = dist.get_world_size()
        # gathered_inputs = torch.zeros(num_processes * batch_size, num_features, hidden_size)
        all_inputs = [torch.zeros(batch_size, num_features, hidden_size,
                                  device=input.device, dtype=input.dtype) for _ in range(num_processes)]
        dist.all_gather(all_inputs, input)
        gathered_inputs = torch.cat(all_inputs, dim=0)
        N = gathered_inputs.size(0) * gathered_inputs.size(2)
        ctx.save_for_backward(N)

        mean = gathered_inputs.mean(dim=(0, 2), keepdim=True)
        var = (gathered_inputs.pow(2) - mean.pow(2)).mean(dim=(0, 2), keepdims=True)

        #  we will not follow pytorch and will use unbiased estimator for both forward and backward
        #  the reason for this is the following: using different formulas for inference and training
        #  is strange and seems mostly like a bug. Here we apply correction.
        var_inv = 1 / (var + 1e-9)
        ctx.save_for_backward(var_inv)

        z = (input - mean) * var_inv.sqrt()
        ctx.save_for_backward(z)

        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(momentum * mean.squeeze(0).squeeze(1))
            running_std.mul_(1 - momentum).add_(momentum * (var * N / (N - 1)).sqrt().squeeze(0).squeeze(1))

        return z.view(*input_shape)

    @staticmethod
    def backward(ctx, grad_output):
        # don't forget to return a tuple of gradients wrt all arguments of `forward`!
        N, var_inv, z = ctx.saved_tensors

        gathered_grad_output = []
        dist.all_gather(gathered_grad_output, grad_output)
        gathered_grad_output = torch.cat(gathered_grad_output, dim=0)

        grad_output_sum = gathered_grad_output.sum(dim=(0, 2, 3), keepdim=True)
        z_times_grad_output_sum = (z * gathered_grad_output).sum(dim=(0, 2, 3), keepdims=True)
        grad_input = var_inv / N * (N * gathered_grad_output - grad_output_sum - z * z_times_grad_output_sum)
        return grad_input, None, None, None, None


class SyncBatchNorm(_BatchNorm):
    """
    Applies Batch Normalization to the input (over the 0 axis), aggregating the activation statistics
    across all processes. You can assume that there are no affine operations in this layer.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__(
            num_features,
            eps,
            momentum,
            affine=False,
            track_running_stats=True,
            device=None,
            dtype=None,
        )
        self.eps = eps
        self.momentum = momentum
        self.running_mean = torch.zeros((num_features,))
        self.running_std = torch.ones((num_features,))

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() < 2:
            raise ValueError(f"expected at least 2 dimensions, got, {input.dim()}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        self._check_input_dim(input)

        if not self.training and self.track_running_stats:
            return F.batch_norm(input, self.running_mean, self.running_var)  # maybe smth else is also needed

        return sync_batch_norm.apply(input, self.running_mean, self.running_std, self.eps, self.momentum)

