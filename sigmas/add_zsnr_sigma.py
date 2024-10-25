import torch


class AddZsnrSigma:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "sigmas": ("SIGMAS", ),
                "sigma_max": ("FLOAT", {
                    "default": 4518.763671875, "min": 0.0, "max": 1.0e38,
                }),
                },
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, sigma_max):
        return (torch.cat([torch.tensor([sigma_max]), sigmas]),)
