"""
nelpy default API

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

# from .objects import *  # NOTE: control exported symbols in objects.py

from .core import *
from .auxiliary import *

# from . import analysis
from . import filtering
from . import plotting
from . import utils
from . import utils_
# from . import io

from . version import __version__

# TODO: decide on which utils to expose directly:
# from .utils import (find_nearest_idx,
#                     find_nearest_indices)

# from .hmmutils import PoissonHMM
# from . import hmmutils

# from .plotting import plot
