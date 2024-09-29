import math

import comfy.samplers
import numpy
import torch


def find_clothest_index(sigma, sigma_triggers):
    index = 0
    for i, trigger_sigma in enumerate(sigma_triggers):
        if trigger_sigma >= sigma:
            index = i
        else:
            break
    return index


class Guider_SheduledCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, model, cfg_max, cfg_min, sigmas):
        self.model = model
        self.cfg_max = cfg_max
        self.cfg_min = cfg_min
        self.sigmas = sigmas
        self.sigma_max = int(max(sigmas))
        self.sigma_min = int(min(sigmas))
        self.model_sig_max = self.model.model.model_sampling.sigma_max
        self.model_sig_min = self.model.model.model_sampling.sigma_min
        self.model_sigma_triggers = numpy.zeros(len(sigmas))
        delta_percent = 1 / len(sigmas)
        for i in range(len(sigmas)):
            percent = i * delta_percent
            self.model_sigma_triggers[i] = \
                self.model.model.model_sampling.percent_to_sigma(percent)

    def set_conds(self, positive, unconditional):
        self.inner_set_conds({"positive": positive, "uncond": unconditional})

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        uncond = self.conds.get("uncond", None)
        positive_cond = self.conds.get("positive", None)

        out = comfy.samplers.calc_cond_batch(
            self.inner_model,
            [uncond, positive_cond],
            x,
            timestep,
            model_options
        )

        current_index = find_clothest_index(
            timestep, self.model_sigma_triggers
        )
        current_sigma = self.sigmas[current_index]
        current_percent = ((current_sigma - self.sigma_min) /
                           (self.sigma_max - self.sigma_min))
        current_cfg = ((self.cfg_max - self.cfg_min) * current_percent
                       + self.cfg_min)

        cfg = comfy.samplers.cfg_function(
            self.inner_model,
            out[1],
            out[0],
            current_cfg,
            x,
            timestep,
            model_options=model_options,
            cond=positive_cond,
            uncond=uncond
        )
        return cfg


class SheduledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "unconditional": ("CONDITIONING", ),
                "cfg_max": ("FLOAT", {
                    "default": 12.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "cfg_min": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                    "round": 0.01
                }),
                "sigmas": ("SIGMAS", ),
                }}

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self, model, positive, unconditional, cfg_max, cfg_min, sigmas
    ):
        guider = Guider_SheduledCFG(model)
        guider.set_conds(positive, unconditional)  # Conds
        guider.set_cfg(model, cfg_max, cfg_min, sigmas)  # Strengths
        return (guider,)
