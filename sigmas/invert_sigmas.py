class InvertSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"sigmas": ("SIGMAS", ),
                     }
                }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas):
        sigma_max = int(max(sigmas))
        sigma_min = int(min(sigmas))

        for i in range(len(sigmas)):
            sigmas[i] = sigma_max - sigmas[i] + sigma_min
        return (sigmas,)
