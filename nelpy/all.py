"""
nelpy full API
==============

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

from . import decoding, filtering, hmmutils, io, plotting, scoring, utils
from .core._analogsignalarray import (
    AnalogSignalArray,
    IMUSensorArray,
    MinimalExampleArray,
    PositionArray,
    RegularlySampledAnalogSignalArray,
)
from .core._coordinates import (
    Abscissa,
    AnalogSignalArrayAbscissa,
    AnalogSignalArrayOrdinate,
    Ordinate,
    TemporalAbscissa,
)
from .core._eventarray import (
    BinnedEventArray,
    BinnedSpikeTrainArray,
    EventArray,
    SpikeTrainArray,
)
from .core._intervalarray import EpochArray, IntervalArray, SpaceArray
from .core._valeventarray import (
    MarkedSpikeTrainArray,
    StatefulValueEventArray,
    ValueEventArray,
)
from .version import __version__

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
    "filtering",
    "hmmutils",
    "io",
    "decoding",
    "scoring",
    "plotting",
    "utils",
    "__version__",
]
