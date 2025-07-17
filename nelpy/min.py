"""
nelpy minimal (min) API
=======================

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

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
    "__version__",
]
