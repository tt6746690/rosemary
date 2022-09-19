"""We prefer a flat&simple interface for importing modules
        `from rosemary import jpt_notebook`
                    instead of
        `from rosemary.jpt import jpt_in_notebook`
    To ahieve this, we `from .modulename import *` in `__init__.py` and 
    specify explicitly names to import with `__all__` inside each module.
"""

from .jpt import *
from .meter import *
from .metrics import *
from .np import *
from .pd import *
from .plt import *
from .tree import *
from .parallel import *

try:
    from .torch_transform import *
    from .torch import *
    from .torchvision import *
except ImportError as e:
    from warnings import warn
    warn(f'Install `torch` for functionalities dependent on torch')