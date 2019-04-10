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
from . import preprocessing
from . import filtering
from . import plotting
from . import utils
from . import utils_
from .utils_ import metrics
# from . import io

from .version import __version__

# for legacy support
import sys
from . import core
# sys.modules['nelpy.core'] = core
# sys.modules['nelpy.core._analogsignalarray'] = core._analogsignalarray
sys.modules['nelpy.core._analogsignalarray'].EpochSignalSlicer = core._accessors.IntervalSeriesSlicer
sys.modules['nelpy.core._analogsignalarray'].TimestampSlicer = core._analogsignalarray.AbscissaSlicer
sys.modules['nelpy.core._epocharray'] = core._intervalarray
sys.modules['nelpy.core._spiketrain'] = core._eventarray
sys.modules['nelpy.core._spiketrain'].EpochUnitSlicer = core._eventarray.IntervalSeriesSlicer
