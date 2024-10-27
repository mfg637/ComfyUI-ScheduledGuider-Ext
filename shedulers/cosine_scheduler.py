import math

import numpy
import torch


class CosineScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "sigma_max": ("FLOAT", {"default": 1, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "sigma_min": ("FLOAT", {"default": 0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "period": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, steps, sigma_max, sigma_min, period):
        sigmas = numpy.zeros(steps)
        delta_s = 1 / steps
        minmax_delta = sigma_max - sigma_min
        period_value = math.pi * period * 2
        period_min = math.cos(min(math.pi, period_value))
        divider = 1 - period_min

        for step in range(steps):
            sigmas[step] = ((math.cos(step * delta_s * period_value)
                             - period_min)
                            / divider * minmax_delta + sigma_min)

        return (torch.FloatTensor(sigmas), )
