from .base import BaseModule
from .unet import UNetPitcher


class ConsistencyPitcher(BaseModule):
    # a wrapper for the student that turns the UNet (which outputs noise) into a consistency function (that outputs clean audio)
    # clean audio = (c_skip * noisy input) + (cout * unet output)

    def __init__(self, unet: UNetPitcher, sigma_data=0.5):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def forward(self, x, t, mean, f0, noise_scheduler):
        c_skip, c_out = self._consistency_dims(t, noise_scheduler)
        return c_skip * x + c_out * self.unet(x=x, mean=mean, f0=f0, t=t)

    def _consistency_dims(self, t, noise_scheduler):
        # formula and constants from the original Consistency Models paper (https://arxiv.org/pdf/2303.01469) "Additional Experimental Details" Section on page 25, 26
        sd = self.sigma_data
        alpha_t = noise_scheduler.alphas_cumprod[t]
        sigma   = ((1 - alpha_t) / alpha_t).sqrt().view(1, 1, 1)

         # epsilon: sigma at the smallest (least noisy) timestep in the schedule
        t_min   = noise_scheduler.timesteps[-1]
        alpha_min = noise_scheduler.alphas_cumprod[t_min]
        epsilon = ((1 - alpha_min) / alpha_min).sqrt().view(1, 1, 1)

        c_skip = sd**2 / ((sigma - epsilon)**2 + sd**2)
        c_out  = sd * (sigma - epsilon) / (sigma**2 + sd**2)**0.5

        return c_skip, c_out
        