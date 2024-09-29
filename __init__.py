from .guiders import SheduledGuider
from .shedulers import cosine_scheduler

NODE_CLASS_MAPPINGS = {
    "SheduledCFGGuider": SheduledGuider.SheduledCFGGuider,
    "CosineScheduler": cosine_scheduler.CosineScheduler,
}
__all__ = ['NODE_CLASS_MAPPINGS']
