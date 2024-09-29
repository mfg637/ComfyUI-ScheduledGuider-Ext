from .guiders import SheduledGuider

NODE_CLASS_MAPPINGS = {
    "SheduledCFGGuider": SheduledGuider.SheduledCFGGuider
}
__all__ = ['NODE_CLASS_MAPPINGS']
