from .guiders import SheduledGuider
from .shedulers import cosine_scheduler
from .sigmas import concat_sigmas, invert_sigmas, offset_sigmas

NODE_CLASS_MAPPINGS = {
    "SheduledCFGGuider": SheduledGuider.SheduledCFGGuider,
    "PerpNegSheduledCFGGuider": SheduledGuider.PerpNegSheduledCFGGuider,
    "CosineScheduler": cosine_scheduler.CosineScheduler,
    "InvertSigmas": invert_sigmas.InvertSigmas,
    "ConcatSigmas": concat_sigmas.ConcatSigmas,
    "OffsetSigmas": offset_sigmas.OffsetSigmas,
}
__all__ = ['NODE_CLASS_MAPPINGS']
