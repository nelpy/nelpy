"""
nelpy.plotting
=====

This is the nelpy plotting sub-package.

nelpy.plotting provides many plot types that work directly on nelpy
objects, as well as some convenience functions to make using matplotlib
more convenient.
"""

import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

from .core import *
from .decoding import *
from .miscplot import *
from .rcmod import *
from .scalebar import add_scalebar, add_simple_scalebar
from .utils import FigureManager, suptitle, savefig
from . import colors
from . import utils

# Set default aesthetics
# setup()

__version__ = '0.0.2'  # should I maintain a separate version for this?