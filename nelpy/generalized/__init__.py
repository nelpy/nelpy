"""
nelpy generalized objects

``nelpy`` is a neuroelectrophysiology object model and data analysis package.
"""

__all__ = ['IntervalArray',
           'UniformlySampledSignalArray',
           'SpikeTrainArray',
           'BinnedSpikeTrainArray',
           'EventArray']
        #    'ValueEventArray',
        #    'StatefulEventArray']

""" Auxiliary data objects """
from ._intervalarray import *

""" Data container objects """
from ._analogsignalarray import AnalogSignalArray
from ._spiketrain import SpikeTrainArray, BinnedSpikeTrainArray
from ._eventarray import EventArray #, ValueEventArray, StatefulEventArray

""" Data linking objects """
# from ._xxx import SignalGroup
