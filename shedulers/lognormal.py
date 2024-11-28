import math

import numpy
import torch


class LogNormalScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "sigma_max": ("FLOAT", {"default": 1, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "sigma_min": ("FLOAT", {"default": 0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "mean": ("FLOAT", {"default": 0.0, "min": -5000.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "standard_deviation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "a": ("FLOAT", {"default": 0.0, "min": -5000.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "b": ("FLOAT", {"default": 1.0, "min": -5000.0, "max": 5000.0, "step": 0.01, "round": False})
                 }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/CFG-schedulers"

    FUNCTION = "get_sigmas"

    def get_sigmas(
            self,
            steps: int,
            sigma_max: float,
            sigma_min: float,
            mean: float,
            standard_deviation: float,
            a: float,
            b: float
            ):
        stdev = standard_deviation
        if a > b:
            _a = a
            _b = b
            a = _b
            b = _a
        elif a == b:
            raise ValueError("Value A must not be equal to B")
        sigmas = numpy.zeros(steps)
        delta_s = (b - a) / steps
        minmax_delta = sigma_max - sigma_min

        for step in range(steps):
            x = (step + 0.5) * delta_s + a
            left_part = 1/(x * stdev * math.sqrt(2 * math.pi))
            right_part = math.exp(-1 * (math.log(x) - mean)**2/(stdev**2*2))
            sigmas[step] = left_part * right_part

        min_value = min(sigmas)
        max_value = max(sigmas)
        value_minmax_delta = max_value - min_value
        scale_coef = minmax_delta / value_minmax_delta
        for step in range(steps):
            sigmas[step] = (sigmas[step] - min_value) * scale_coef + sigma_min

        return (torch.FloatTensor(sigmas), )
