from .core import Fenrir
from . import interactive as _interactive

__all__ = ["Fenrir"]
__version__ = "0.1.0"

_interactive.attach_interactive_api(Fenrir)

