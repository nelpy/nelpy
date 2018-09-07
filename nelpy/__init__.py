"""
nelpy default API

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

# from .objects import *  # NOTE: control exported symbols in objects.py

from .generalized import *
from .auxiliary import *

# from . import analysis
from . import filtering
from . import plotting
from . import utils
from . import utils_
from .utils_ import metrics
# from . import io

from .version import __version__

# for legacy support
import sys
from . import generalized
sys.modules['nelpy.core'] = generalized
sys.modules['nelpy.core._analogsignalarray'] = generalized._analogsignalarray
sys.modules['nelpy.core._analogsignalarray'].EpochSignalSlicer = generalized._analogsignalarray.IntervalSignalSlicer
sys.modules['nelpy.core._epocharray'] = generalized._intervalarray
sys.modules['nelpy.core._spiketrain'] = generalized._eventarray
sys.modules['nelpy.core._spiketrain'].EpochUnitSlicer = generalized._eventarray.IntervalSeriesSlicer
