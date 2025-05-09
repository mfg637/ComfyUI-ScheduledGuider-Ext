import numpy
import torch

def sign(num):
    return -1 if num < 0 else 1

class X_InverseScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                {"steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "sigma_max": ("FLOAT", {"default": 1, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "sigma_min": ("FLOAT", {"default": 0, "min": 0.0, "max": 5000.0, "step": 0.01, "round": False}),
                 "k": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5000.0, "step": 1.0, "round": False}),
                 "a": ("FLOAT", {"default": 0.0, "min": -5000.0, "max": 5000.0, "step": 0.1, "round": False}),
                 "b": ("FLOAT", {"default": 1.0, "min": -5000.0, "max": 5000.0, "step": 0.1, "round": False}),
                 "x_abs_min_limit": ("FLOAT", {"default": 1e-9, "min": 1e-256, "max": 2.0, "step": 0.01, "round": False}),
                 "y_abs_max_limit": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 5000.0, "step": 1.0, "round": False}),
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
            k: float,
            a: float,
            b: float,
            x_abs_min_limit: float,
            y_abs_max_limit: float
        ):
        if a > b:
            _a = a
            _b = b
            a = _b
            b = _a
        elif a == b:
            raise ValueError("Value A must not be equal to B")
        if y_abs_max_limit > x_abs_min_limit:
            y_abs_min_x = k / y_abs_max_limit
        else:
            y_abs_min_x = x_abs_min_limit
        abs_min_x = max(x_abs_min_limit, y_abs_min_x)

        sigmas = numpy.zeros(steps)
        delta_s = (b - a) / steps
        minmax_delta = sigma_max - sigma_min
        for step in range(steps):
            x = step * delta_s + a
            if abs(x) < abs_min_x:
                x = abs_min_x * sign(x)
            sigmas[step] = k / x

        min_value = min(sigmas)
        max_value = max(sigmas)
        value_minmax_delta = max_value - min_value
        scale_coef = minmax_delta / value_minmax_delta
        for step in range(steps):
            sigmas[step] = (sigmas[step] - min_value) * scale_coef + sigma_min

        return (torch.FloatTensor(sigmas), )
