from .guiders import SheduledGuider
from .shedulers import cosine_scheduler, gaussian, lognormal
from .sigmas import concat_sigmas, invert_sigmas, offset_sigmas, split_by_value

NODE_CLASS_MAPPINGS = {
    "ScheduledCFGGuider": SheduledGuider.SheduledCFGGuider,
    "PerpNegScheduledCFGGuider": SheduledGuider.PerpNegSheduledCFGGuider,
    "CosineScheduler": cosine_scheduler.CosineScheduler,
    "GaussianScheduler": gaussian.GaussianScheduler,
    "LogNormal Scheduler": lognormal.LogNormalScheduler,
    "InvertSigmas": invert_sigmas.InvertSigmas,
    "ConcatSigmas": concat_sigmas.ConcatSigmas,
    "OffsetSigmas": offset_sigmas.OffsetSigmas,
    "SplitSigmasByValue": split_by_value.SplitSigmasByValue
}
__all__ = ['NODE_CLASS_MAPPINGS']
