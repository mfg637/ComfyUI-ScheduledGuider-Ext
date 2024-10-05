class OffsetSigmas:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "sigmas": ("SIGMAS", ),
                "offset": ("FLOAT", {
                    "default": 1.0,
                    "step": 0.1,
                }),
            },
        }
    RETURN_TYPES = ("SIGMAS",)
    CATEGORY = "sampling/custom_sampling/sigmas"

    FUNCTION = "get_sigmas"

    def get_sigmas(self, sigmas, offset):

        for i in range(len(sigmas)):
            sigmas[i] = sigmas[i] + offset
        return (sigmas,)
