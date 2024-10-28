"""
nelpy minimal (min) API
=======================

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
