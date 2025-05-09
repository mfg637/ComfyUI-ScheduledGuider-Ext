from . import cosine_scheduler, gaussian, lognormal, x_inverse

NODE_CLASS_MAPPINGS = {
    "CosineScheduler": cosine_scheduler.CosineScheduler,
    "GaussianScheduler": gaussian.GaussianScheduler,
    "LogNormal Scheduler": lognormal.LogNormalScheduler,
    "k/x scheduler": x_inverse.X_InverseScheduler,
}