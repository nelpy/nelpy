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

from ._intervalarray import *
from ._coordinates import *
from ._analogsignalarray import *
from ._eventarray import *

""" Data container objects """

# from ._eventarray import EventArray #, ValueEventArray, StatefulEventArray

""" Data linking objects """
# from ._xxx import SignalGroup
