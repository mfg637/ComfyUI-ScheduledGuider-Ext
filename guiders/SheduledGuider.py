import numpy

import comfy.samplers
from comfy_extras.nodes_perpneg import perp_neg


def find_clothest_index(sigma, sigma_triggers):
    index = 0
    for i, trigger_sigma in enumerate(sigma_triggers):
        if trigger_sigma >= sigma:
            index = i
        else:
            break
    return index


class Guider_SheduledCFG(comfy.samplers.CFGGuider):
    def set_cfg(self, model, cfg_max, cfg_min, sigmas, neg_scale=0):
        self.model = model
        self.cfg_max = cfg_max
        self.cfg_min = cfg_min
        self.sigmas = sigmas
        self.neg_scale = neg_scale
        self.sigma_max = int(max(sigmas))
        self.sigma_min = int(min(sigmas))
        self.model_sig_max = self.model.model.model_sampling.sigma_max
        self.model_sig_min = self.model.model.model_sampling.sigma_min
        self.model_sigma_triggers = numpy.zeros(len(sigmas))
        self.use_negative_as_unconditional = True
        delta_percent = 1 / len(sigmas)
        for i in range(len(sigmas)):
            percent = i * delta_percent
            self.model_sigma_triggers[i] = \
                self.model.model.model_sampling.percent_to_sigma(percent)

    def set_use_negative(self, use_neg: bool):
        self.use_negative_as_unconditional = use_neg

    def set_conds(self, positive, unconditional, negative=None):
        conds = {"positive": positive, "uncond": unconditional}
        if negative is not None:
            conds["negative"] = negative
        self.inner_set_conds(conds)

    def get_conditions(self):
        uncond = self.conds.get("uncond", None)
        positive_cond = self.conds.get("positive", None)
        negative_cond = self.conds.get("negative", None)

        return (uncond, positive_cond, negative_cond)

    def calc_predictions(self, x, timestep, model_options, conditions):
        uncond, positive_cond, negative_cond = conditions

        if negative_cond is not None:
            conditions = [uncond, positive_cond, negative_cond]
        else:
            conditions = [uncond, positive_cond]

        out = comfy.samplers.calc_cond_batch(
            self.inner_model,
            conditions,
            x,
            timestep,
            model_options
        )
        if len(out) == 3:
            return {
                "unconditional": out[0],
                "positive": out[1],
                "negative": out[2]
            }
        else:
            return {
                "unconditional": out[0],
                "positive": out[1],
                "negative": None
            }

    def calc_cfg(
        self, predictions, cfg, x, timestep, model_options, conditions
    ):
        uncond, positive_cond, negative_cond = conditions

        if predictions["negative"] is not None:
            cfg_result = perp_neg(
                x,
                predictions["positive"],
                predictions["negative"],
                predictions["unconditional"],
                self.neg_scale,
                cfg
            )

            for fn in model_options.get("sampler_post_cfg_function", []):
                if self.use_negative_as_unconditional:
                    args = {
                        "denoised": cfg_result,
                        "cond": positive_cond,
                        "uncond": negative_cond,
                        "model": self.inner_model,
                        "uncond_denoised": predictions["negative"],
                        "cond_denoised": predictions["positive"],
                        "sigma": timestep,
                        "model_options": model_options,
                        "input": x,
                        "empty_cond": uncond,
                        "empty_cond_denoised": predictions["unconditional"],
                    }
                else:
                    args = {
                        "denoised": cfg_result,
                        "cond": positive_cond,
                        "uncond": uncond,
                        "model": self.inner_model,
                        "uncond_denoised": predictions["unconditional"],
                        "cond_denoised": predictions["positive"],
                        "sigma": timestep,
                        "model_options": model_options,
                        "input": x,
                    }
                cfg_result = fn(args)

            return cfg_result
        else:
            return comfy.samplers.cfg_function(
                self.inner_model,
                predictions["positive"],
                predictions["unconditional"],
                cfg,
                x,
                timestep,
                model_options=model_options,
                cond=positive_cond,
                uncond=uncond
            )

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        conditions = self.get_conditions()
        predictions = self.calc_predictions(
            x, timestep, model_options, conditions
        )

        current_index = find_clothest_index(
            timestep, self.model_sigma_triggers
        )
        current_sigma = self.sigmas[current_index]
        current_percent = ((current_sigma - self.sigma_min) /
                           (self.sigma_max - self.sigma_min))
        current_cfg = ((self.cfg_max - self.cfg_min) * current_percent
                       + self.cfg_min)

        return self.calc_cfg(
            predictions, current_cfg, x, timestep, model_options, conditions
        )


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
        guider.set_conds(positive, unconditional)
        guider.set_cfg(model, cfg_max, cfg_min, sigmas)
        return (guider,)


class PerpNegSheduledCFGGuider:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
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
                "neg_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.01
                }),
                "sigmas": ("SIGMAS", ),
                "use_negative_as_unconditional": ("BOOLEAN", {
                    "default": True
                }),
                }}

    RETURN_TYPES = ("GUIDER",)

    FUNCTION = "get_guider"
    CATEGORY = "sampling/custom_sampling/guiders"

    def get_guider(
            self,
            model,
            positive,
            negative,
            unconditional,
            cfg_max,
            cfg_min,
            neg_scale,
            sigmas,
            use_negative_as_unconditional
    ):
        guider = Guider_SheduledCFG(model)
        guider.set_conds(positive, unconditional, negative)
        guider.set_cfg(model, cfg_max, cfg_min, sigmas, neg_scale)
        guider.set_use_negative(use_negative_as_unconditional)
        return (guider,)
