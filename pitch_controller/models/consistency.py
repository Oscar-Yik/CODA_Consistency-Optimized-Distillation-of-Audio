from .base import BaseModule
from .unet import UNetPitcher
import torch

class ConsistencyPitcher(BaseModule):
    # a wrapper for the student that turns the UNet (which outputs noise) into a consistency function (that outputs clean audio)
    # clean audio = (c_skip * noisy input) + (cout * x0_prediction)

    def __init__(self, unet: UNetPitcher, sigma_data=0.5):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data
        self._cached_epsilon = None
        self._cached_timesteps_len = None

    def forward(self, x, t, mean, f0, noise_scheduler):
        c_skip, c_out = self._consistency_dims(t, noise_scheduler)

        # UNet predicts noise (epsilon), convert to x_0 prediction
        epsilon_pred = self.unet(x=x, mean=mean, f0=f0, t=t)
        alpha_t = noise_scheduler.alphas_cumprod.to(x.device)[t]
        sqrt_alpha_t = alpha_t.sqrt().view(-1, 1, 1)
        sqrt_one_minus_alpha_t = (1 - alpha_t).sqrt().view(-1, 1, 1)

        # Convert epsilon prediction to x_0 prediction
        x0_pred = (x - sqrt_one_minus_alpha_t * epsilon_pred) / sqrt_alpha_t
        x0_pred = torch.clamp(x0_pred, min=-2.0, max=2.0)

        # Apply consistency model parameterization
        return c_skip * x + c_out * x0_pred

    def _get_epsilon(self, noise_scheduler):
        # Cache epsilon since it only depends on the timestep schedule
        if self._cached_epsilon is None or self._cached_timesteps_len != len(noise_scheduler.timesteps):
            t_min = noise_scheduler.timesteps[-1]
            alpha_min = noise_scheduler.alphas_cumprod[t_min]
            self._cached_epsilon = ((1 - alpha_min) / alpha_min).sqrt().view(1, 1, 1)
            self._cached_timesteps_len = len(noise_scheduler.timesteps)
        return self._cached_epsilon

    def _consistency_dims(self, t, noise_scheduler):
        # formula and constants from the original Consistency Models paper (https://arxiv.org/pdf/2303.01469) "Additional Experimental Details" Section on page 25, 26
        sd = self.sigma_data
        alpha_t = noise_scheduler.alphas_cumprod.to(t.device)[t]
        sigma   = ((1 - alpha_t) / alpha_t).sqrt().view(-1, 1, 1)

        # epsilon: sigma at the smallest (least noisy) timestep in the schedule
        epsilon = self._get_epsilon(noise_scheduler)

        c_skip = sd**2 / ((sigma - epsilon)**2 + sd**2)
        c_out  = sd * (sigma - epsilon) / (sigma**2 + sd**2)**0.5

        return c_skip, c_out
        