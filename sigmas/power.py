import math
import torch
import numpy


def to_power(sigmas_in, power: float):
        steps = len(sigmas_in)
        sigmas_out = numpy.zeros(steps)

        for step in range(steps):
            sigmas_out[step] = math.pow(sigmas_in[step], power)

        return (torch.FloatTensor(sigmas_out), )


def calc_power_by_base(sigmas_in, base: float):
        steps = len(sigmas_in)
        sigmas_out = numpy.zeros(steps)

        for step in range(steps):
            sigmas_out[step] = math.pow(base, sigmas_in[step])

        return (torch.FloatTensor(sigmas_out), )


def calc_function(sigmas_in, f):
        steps = len(sigmas_in)
        sigmas_out = numpy.zeros(steps)

        for step in range(steps):
            sigmas_out[step] = f(sigmas_in[step])

        return (torch.FloatTensor(sigmas_out), )


class PredefinedExponent:
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
            return calc_function(sigmas, math.exp)
        elif base == "10":
            return calc_power_by_base(sigmas, 10)
        elif base == "2":
            return calc_function(sigmas, math.exp2)
        else:
            raise ValueError(f"Invalid value \"{base}\"")


class CustomExponent:
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
        return calc_power_by_base(sigmas, base)


class SigmasToPower:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "power": ("FLOAT", {
                    "default": 2.0,
                    "step": 0.1,
                }),
            },
        }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, power):
        return to_power(sigmas, power)