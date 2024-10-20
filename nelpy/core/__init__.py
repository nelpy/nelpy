"""
nelpy core objects

``nelpy`` is a neuroelectrophysiology object model and data analysis package.
"""

# __all__ = ['IntervalArray',
#            'EpochArray',
#            'RegularlySampledAnalogSignalArray',
#            'SpikeTrainArray',
#            'BinnedSpikeTrainArray',
#            'EventArray']
#         #    'ValueEventArray',
#         #    'StatefulEventArray']

# from ._intervalarray import *
# from ._coordinates import *
# from ._analogsignalarray import *
# from ._eventarray import *
# from ._valeventarray import *

from ._intervalarray import IntervalArray, EpochArray, SpaceArray
from ._coordinates import (
    Abscissa,
    Ordinate,
    AnalogSignalArrayAbscissa,
    AnalogSignalArrayOrdinate,
    TemporalAbscissa,
)
from ._analogsignalarray import (
    RegularlySampledAnalogSignalArray,
    AnalogSignalArray,
    PositionArray,
    IMUSensorArray,
    MinimalExampleArray,
)

from ._eventarray import (
    EventArray,
    BinnedEventArray,
    SpikeTrainArray,
    BinnedSpikeTrainArray,
)
from ._valeventarray import (
    ValueEventArray,
    MarkedSpikeTrainArray,
    StatefulValueEventArray,
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
]

""" Data container objects """

# from ._eventarray import EventArray #, ValueEventArray, StatefulEventArray

""" Data linking objects """
# from ._xxx import SignalGroup
