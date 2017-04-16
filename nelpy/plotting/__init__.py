"""
nelpy.plotting
=====

This is the nelpy plotting module.
"""

import matplotlib as mpl

_orig_rc_params = mpl.rcParams.copy()

from .core import *
from .rcmod import *
from .decoding import *
from .utils import *
from .scalebar import add_scalebar
from .miscplot import *

# Set default aesthetics
setup()

__version__ = '0.0.2'  # should I maintain a separate version for this?