from .guiders import SheduledGuider
from .shedulers import NODE_CLASS_MAPPINGS as scheduler_node_mappings
from .sigmas import NODE_CLASS_MAPPINGS as sigmas_node_mappings

NODE_CLASS_MAPPINGS = {
    "ScheduledCFGGuider": SheduledGuider.SheduledCFGGuider,
    "PerpNegScheduledCFGGuider": SheduledGuider.PerpNegSheduledCFGGuider,
}
NODE_CLASS_MAPPINGS.update(scheduler_node_mappings)
NODE_CLASS_MAPPINGS.update(sigmas_node_mappings)
__all__ = ['NODE_CLASS_MAPPINGS']
