from .guiders import SheduledGuider
from .shedulers import cosine_scheduler, gaussian, lognormal
from .sigmas import add_zsnr_sigma, concat_sigmas, invert_sigmas, offset_sigmas

NODE_CLASS_MAPPINGS = {
    "ScheduledCFGGuider": SheduledGuider.SheduledCFGGuider,
    "PerpNegScheduledCFGGuider": SheduledGuider.PerpNegSheduledCFGGuider,
    "CosineScheduler": cosine_scheduler.CosineScheduler,
    "GaussianScheduler": gaussian.GaussianScheduler,
    "LogNormal Scheduler": lognormal.LogNormalScheduler,
    "Add zSNR Sigma max": add_zsnr_sigma.AddZsnrSigma,
    "InvertSigmas": invert_sigmas.InvertSigmas,
    "ConcatSigmas": concat_sigmas.ConcatSigmas,
    "OffsetSigmas": offset_sigmas.OffsetSigmas,
}
__all__ = ['NODE_CLASS_MAPPINGS']
