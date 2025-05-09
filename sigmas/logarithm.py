import math
import torch
import numpy


def calc_logarithm(sigmas_in, base: float | None = None):
        steps = len(sigmas_in)
        sigmas_out = numpy.zeros(steps)

        if base is not None:
            for step in range(steps):
                sigmas_out[step] = math.log(sigmas_in[step], base)
        else:
            for step in range(steps):
                sigmas_out[step] = math.log(sigmas_in[step])

        return (torch.FloatTensor(sigmas_out), )


class PredefinedLogarithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "base": (["e", "10", "2"], {}),
            },
        }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, base):
        if base == "e":
            return calc_logarithm(sigmas)
        elif base == "10":
            return calc_logarithm(sigmas, 10)
        elif base == "2":
            return calc_logarithm(sigmas, 2)
        else:
            raise ValueError(f"Invalid value \"{base}\"")


class CustomBaseLogarithm:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "base": ("FLOAT", {
                    "default": 2.0,
                    "step": 0.1,
                    "min": 1.1
                }),
            },
        }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, base):
        return calc_logarithm(sigmas, base)