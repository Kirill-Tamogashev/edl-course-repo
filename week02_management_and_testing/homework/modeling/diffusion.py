from typing import Dict, Tuple

import torch
import torch.nn as nn

from tqdm.notebook import tqdm


class DiffusionModel(nn.Module):
    def __init__(
        self,
        eps_model: nn.Module,
        betas: Tuple[float, float],
        num_timesteps: int,
    ):
        super().__init__()
        self.eps_model = eps_model

        for name, schedule in get_schedules(betas[0], betas[1], num_timesteps).items():
            self.register_buffer(name, schedule)

        self.num_timesteps = num_timesteps
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        timestep = torch.randint(1, self.num_timesteps + 1, (x.shape[0],), device=x.device)
        eps = torch.rand_like(x)

        x_t = (
            self.sqrt_alphas_cumprod[timestep, None, None, None] * x
            + self.one_minus_alpha_over_prod[timestep, None, None, None] * eps
        )

        return self.criterion(eps, self.eps_model(x_t, timestep / self.num_timesteps))

    def sample(self, num_samples: int, size, device) -> Tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn(num_samples, *size, device=device)

        x_i = noise
        for i in tqdm(range(self.num_timesteps, 0, -1)):
            z = torch.randn(num_samples, *size, device=device) if i > 1 else 0
            time = torch.tensor(i / self.num_timesteps, device=device).repeat(num_samples, 1)

            eps = self.eps_model(x_i, time)
            x_i = self.inv_sqrt_alphas[i] * (x_i - eps * self.one_minus_alpha_over_prod[i]) + self.sqrt_betas[i] * z

        return noise, x_i


def get_schedules(beta1: float, beta2: float, num_timesteps: int, eps: float = 1e-12) -> Dict[str, torch.Tensor]:
    assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

    betas = (beta2 - beta1) * torch.arange(0, num_timesteps + 1, dtype=torch.float32) / num_timesteps + beta1
    sqrt_betas = torch.sqrt(betas)
    alphas = 1 - betas

    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)

    # Add Small correction eps to avoid div by zero
    inv_sqrt_alphas = 1 / (torch.sqrt(alphas) + eps)

    sqrt_one_minus_alpha_prod = torch.sqrt(1 - alphas_cumprod)
    # Add Small correction eps to avoid div by zero
    one_minus_alpha_over_prod = (1 - alphas) / (sqrt_one_minus_alpha_prod + eps)

    return {
        "alphas": alphas,
        "inv_sqrt_alphas": inv_sqrt_alphas,
        "sqrt_betas": sqrt_betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alpha_prod": sqrt_one_minus_alpha_prod,
        "one_minus_alpha_over_prod": one_minus_alpha_over_prod,
    }
