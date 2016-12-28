# Capture the original matplotlib rcParams
import matplotlib as mpl
_orig_rc_params = mpl.rcParams.copy()

# Import eplotlib objects
from .utils import *
from .palettes import *

# Set default aesthetics
set()

__version__ = "0.0.1"
