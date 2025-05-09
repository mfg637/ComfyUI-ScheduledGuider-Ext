import math
import torch
import numpy


class Arctancheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "sigma_max": ("FLOAT", {"default": 1, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "sigma_min": ("FLOAT", {"default": 0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "a": ("FLOAT", {"default": 0.0, "min": -5000.0, "max": 5000.0, "step": 0.01, "round": True}),
                 "b": ("FLOAT", {"default": 1.0, "min": -5000.0, "max": 5000.0, "step": 0.01, "round": True})
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
            a: float,
            b: float
            ):
        if a > b:
            _a = a
            _b = b
            a = _b
            b = _a
        elif a == b:
            raise ValueError("Value A must not be equal to B")
        
        a = (a * math.pi) / 2
        b = (b * math.pi) / 2
        sigmas = numpy.zeros(steps)
        delta_s = (b - a) / steps
        minmax_delta = sigma_max - sigma_min

        for step in range(steps):
            x = (step + 1) * delta_s + a
            sigmas[step] = math.atan(x)

        min_value = min(sigmas)
        max_value = max(sigmas)
        value_minmax_delta = max_value - min_value
        scale_coef = minmax_delta / value_minmax_delta
        for step in range(steps):
            sigmas[step] = (sigmas[step] - min_value) * scale_coef + sigma_min

        return (torch.FloatTensor(sigmas), )