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
        sig1_len = len(sigmas1)
        sig2_len = len(sigmas2)
        sigmas = torch.empty(sig1_len + sig2_len)
        for i in range(sig1_len):
            sigmas[i] = sigmas1[i]
        for i in range(sig2_len):
            sigmas[sig1_len + i] = sigmas2[i]
        return (sigmas,)
