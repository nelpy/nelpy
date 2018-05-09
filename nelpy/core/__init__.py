"""
nelpy core objects

``nelpy`` is a neuroelectrophysiology object model and data analysis
suite.
"""

__all__ = ['EpochArray',
           'AnalogSignalArray',
           'SpikeTrainArray',
           'BinnedSpikeTrainArray',
           'EventArray']
        #    'ValueEventArray',
        #    'StatefulEventArray']

""" Auxiliary data objects """
from ._epocharray import EpochArray

""" Data container objects """
from ._analogsignalarray import AnalogSignalArray
from ._spiketrain import SpikeTrainArray, BinnedSpikeTrainArray
from ._eventarray import EventArray #, ValueEventArray, StatefulEventArray

""" Data linking objects """
# from ._xxx import SignalGroup
