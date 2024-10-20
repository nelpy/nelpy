"""
nelpy default API

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

from .core._intervalarray import IntervalArray, EpochArray, SpaceArray
from .core._coordinates import (
    Abscissa,
    Ordinate,
    AnalogSignalArrayAbscissa,
    AnalogSignalArrayOrdinate,
    TemporalAbscissa,
)
from .core._analogsignalarray import (
    RegularlySampledAnalogSignalArray,
    AnalogSignalArray,
    PositionArray,
    IMUSensorArray,
    MinimalExampleArray,
)
from .core._eventarray import (
    EventArray,
    BinnedEventArray,
    SpikeTrainArray,
    BinnedSpikeTrainArray,
)
from .core._valeventarray import (
    ValueEventArray,
    MarkedSpikeTrainArray,
    StatefulValueEventArray,
)

from .auxiliary._tuningcurve import (
    TuningCurve1D,
    TuningCurve2D,
    DirectionalTuningCurve1D,
)
from .auxiliary._session import Session
from .auxiliary._results import ResultsContainer, load_pkl, save_pkl

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
sys.modules["nelpy.core._analogsignalarray"].EpochSignalSlicer = (
    core._analogsignalarray.IntervalSignalSlicer
)
sys.modules["nelpy.core._analogsignalarray"].TimestampSlicer = (
    core._analogsignalarray.AbscissaSlicer
)
sys.modules["nelpy.core._epocharray"] = core._intervalarray
sys.modules["nelpy.core._spiketrain"] = core._eventarray
sys.modules["nelpy.core._spiketrain"].EpochUnitSlicer = (
    core._analogsignalarray.IntervalSignalSlicer
)
sys.modules["nelpy.core._eventarray"].ItemGetter_loc = (
    core._eventarray._accessors.ItemGetterLoc
)
sys.modules["nelpy.core._eventarray"].ItemGetter_iloc = (
    core._eventarray._accessors.ItemGetterIloc
)

__all__ = [
    "IntervalArray",
    "EpochArray",
    "SpaceArray",
    "Abscissa",
    "Ordinate",
    "AnalogSignalArrayAbscissa",
    "AnalogSignalArrayOrdinate",
    "TemporalAbscissa",
    "RegularlySampledAnalogSignalArray",
    "AnalogSignalArray",
    "PositionArray",
    "IMUSensorArray",
    "MinimalExampleArray",
    "EventArray",
    "BinnedEventArray",
    "SpikeTrainArray",
    "BinnedSpikeTrainArray",
    "ValueEventArray",
    "MarkedSpikeTrainArray",
    "StatefulValueEventArray",
    "TuningCurve1D",
    "TuningCurve2D",
    "DirectionalTuningCurve1D",
    "Session",
    "ResultsContainer",
    "load_pkl",
    "save_pkl",
    "preprocessing",
    "filtering",
    "plotting",
    "utils",
    "utils_",
    "metrics",
    "__version__",
]
