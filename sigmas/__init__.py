from . import concat_sigmas, invert_sigmas, offset_sigmas, split_by_value, power, logarithm

NODE_CLASS_MAPPINGS = {
    "InvertSigmas": invert_sigmas.InvertSigmas,
    "ConcatSigmas": concat_sigmas.ConcatSigmas,
    "OffsetSigmas": offset_sigmas.OffsetSigmas,
    "SplitSigmasByValue": split_by_value.SplitSigmasByValue,
    "SigmasToPower": power.SigmasToPower,
    "PredefinedExponent": power.PredefinedExponent,
    "CustomExponent": power.CustomExponent,
    "PredefinedLogarithm": logarithm.PredefinedLogarithm,
    "CustomBaseLogarithm": logarithm.CustomBaseLogarithm,
}