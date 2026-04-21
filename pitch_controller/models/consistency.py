from .base import BaseModule
from .unet import UNetPitcher
import torch

class ConsistencyPitcher(BaseModule):
    # a wrapper for the student that turns the UNet (which outputs noise) into a consistency function (that outputs clean audio)
    # clean audio = (c_skip * noisy input) + (cout * x0_prediction)

    def __init__(self, unet: UNetPitcher, sigma_data=0.5, sigma_min=0.002):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data
        self.sigma_min = sigma_min

    def forward(self, x, t, mean, f0, noise_scheduler):
        c_skip, c_out = self._consistency_dims(t, noise_scheduler)

        # UNet predicts noise (epsilon)
        epsilon_pred = self.unet(x=x, mean=mean, f0=f0, t=t)

        # -- FP32 SHIELD --
        epsilon_pred = epsilon_pred.float()
        x = x.float()
        c_skip = c_skip.float()
        c_out = c_out.float()
        # -----------------

        alpha_t = noise_scheduler.alphas_cumprod.to(x.device)[t]

        if alpha_t.dim() == 0:  # scalar
            sqrt_alpha_t = alpha_t.sqrt().view(1, 1, 1)
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt().view(1, 1, 1)
        else:  # batch [B]
            sqrt_alpha_t = alpha_t.sqrt().view(-1, 1, 1)
            sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt().view(-1, 1, 1)

        # Convert epsilon prediction to x_0 prediction
        x0_pred = (x - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
        x0_pred = torch.clamp(x0_pred, min=-2.0, max=2.0)

        # Apply consistency model parameterization
        return c_skip * x + c_out * x0_pred

    def _consistency_dims(self, t, noise_scheduler):
        # formula and constants from the original Consistency Models paper (https://arxiv.org/pdf/2303.01469)
        # Using boundary condition formulation from OpenAI implementation
        sd = self.sigma_data
        sigma_min = self.sigma_min
        alpha_t = noise_scheduler.alphas_cumprod.to(t.device)[t]
        
        # Handle both scalar and batch timesteps
        if alpha_t.dim() == 0:  # scalar
            sigma = ((1 - alpha_t) / alpha_t).sqrt().view(1, 1, 1)
        else:  # batch [B]
            sigma = ((1 - alpha_t) / alpha_t).sqrt().view(-1, 1, 1)

        # Boundary condition formulation (for distillation)
        c_skip = sd**2 / ((sigma - sigma_min)**2 + sd**2)
        c_out = (sigma - sigma_min) * sd / (sigma**2 + sd**2)**0.5

        return c_skip, c_out
        