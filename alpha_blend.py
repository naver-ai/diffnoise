# Partially adapted from `DDPMScheduler` in `huggingface/diffusers` for beta adjustment and addjing noise.

import torch
from diffusers import DDPMScheduler

class AlphaBlendedScheduler(DDPMScheduler):
    def __init__(self, factor=1.2, *args, **kwargs):
        # Initialize the base class first
        super().__init__(*args, **kwargs)
        self._adjust_betas(factor)  # Adjust the betas by multiplying them by 1.2

    def _adjust_betas(self, factor):
        """
        Adjust beta values by a specified factor after initialization and recompute alphas and related variables.
        """
        # Scale the betas by the specified factor (e.g., 1.2)
        self.betas = self.betas ** factor

        # Recompute alphas and alphas_cumprod
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure alphas_cumprod and timestep have same device and dtype as x
        # Move the self.alphas_cumprod to device to avoid redundant CPU to GPU data movement
        # for the subsequent add_noise calls
        self.alphas_cumprod = self.alphas_cumprod.to(device=x.device)
        alphas_cumprod = self.alphas_cumprod.to(dtype=x.dtype)

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(x.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(x.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        alpha_blended_samples = sqrt_alpha_prod * x + sqrt_one_minus_alpha_prod * noise
        return alpha_blended_samples
    
    def alpha_blended(self, x, timesteps=None):
        bs = x.shape[0]
        noise = torch.randn(x.shape, device=x.device)
        samples = self.add_noise(x, noise, timesteps)
        return samples
