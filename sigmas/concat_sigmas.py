import torch


class ConcatSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "sigmas1": ("SIGMAS", ),
                "sigmas2": ("SIGMAS", ),
                },
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas1, sigmas2):
        return (torch.cat([sigmas1, sigmas2]),)
