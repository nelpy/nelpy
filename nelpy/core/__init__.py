"""
nelpy core objects

``nelpy`` is a neuroelectrophysiology object model and data analysis
suite.
"""

__all__ = ['EpochArray',
           'AnalogSignalArray',
           'SpikeTrainArray',
           'BinnedSpikeTrainArray']

from ._epocharray import EpochArray
from ._analogsignalarray import AnalogSignalArray
from ._spiketrain import SpikeTrainArray, BinnedSpikeTrainArray
