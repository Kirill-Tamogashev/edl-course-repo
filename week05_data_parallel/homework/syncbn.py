import torch
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
    def forward(ctx, input, running_mean, running_var, eps: float, momentum: float):
        batch = input.size(0)
        hidden = input.size(2)
        world_size = 1 #dist.get_world_size()

        mean = input.mean(dim=(0, 2))
        # dist.all_reduce(mean)
        mean /= world_size

        squared_mean = input.pow(2).mean(dim=(0, 2))
        # dist.all_reduce(squared_mean)
        squared_mean /= world_size

        var = squared_mean - mean.pow(2)
        inverse = 1 / torch.sqrt(var + eps)
        ctx.save_for_backward(inverse)
        # print(inverse)
        
        normalized = (input - mean.view(1, -1, 1)) * inverse.view(1, -1, 1)
        ctx.save_for_backward(inverse, normalized)
        # print(normalized)
        
        bias_correction = (world_size * batch * hidden) /  (world_size * batch * hidden - 1) 
        with torch.no_grad():
            running_mean.mul_(1 - momentum).add_(mean * momentum)
            running_var.mul_(1 - momentum).add_(var * bias_correction * momentum)

        return normalized

    @staticmethod
    def backward(ctx, grad_output):
        batch = grad_output.size(0)
        hidden = grad_output.size(2)
        world_size = 1 # dist.get_world_size()
        
        N = batch * hidden * world_size
        inverse, normalized = ctx.saved_tensors
        inverse = inverse.view(1, -1, 1)
        
        grad_output_sum = grad_output.sum(dim=(0, 2))
        # dist.all_reduce(grad_output_sum)
        
        norm_mul_grad_output_sum = (normalized * grad_output).sum(dim=(0, 2))
        # dist.all_reduce(norm_mul_grad_output_sum)
        
        grad_input = inverse / N * (
            N * grad_output \
            - grad_output_sum.view(1, -1, 1) \
            - normalized * norm_mul_grad_output_sum.view(1, -1, 1)
        )

        return grad_input, None, None, None, None


def batch_norm(tensor, running_mean, running_var, eps: float = 1e-05):
    inv_std = (running_var + eps).sqrt().pow(-1)
    return (tensor - running_mean.view(1, -1, 1)) * inv_std.view(1, -1, 1)


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
        self.running_var = torch.ones((num_features,))

    def _check_input_dim(self, input: torch.Tensor) -> None:
        if input.dim() < 2:
            raise ValueError(f"expected at least 2 dimensions, got, {input.dim()}")

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # You will probably need to use `sync_batch_norm` from above
        self._check_input_dim(input)

        if not self.training and self.track_running_stats:
            return batch_norm(input, self.running_mean, self.running_var)  # maybe smth else is also needed

        return sync_batch_norm.apply(input, self.running_mean, self.running_var, self.eps, self.momentum)

