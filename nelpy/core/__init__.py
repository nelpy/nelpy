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

from ._analogsignalarray import (
    AnalogSignalArray,
    IMUSensorArray,
    MinimalExampleArray,
    PositionArray,
    RegularlySampledAnalogSignalArray,
)
from ._coordinates import (
    Abscissa,
    AnalogSignalArrayAbscissa,
    AnalogSignalArrayOrdinate,
    Ordinate,
    TemporalAbscissa,
)
from ._eventarray import (
    BinnedEventArray,
    BinnedSpikeTrainArray,
    EventArray,
    SpikeTrainArray,
)
from ._intervalarray import EpochArray, IntervalArray, SpaceArray
from ._valeventarray import (
    MarkedSpikeTrainArray,
    StatefulValueEventArray,
    ValueEventArray,
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
