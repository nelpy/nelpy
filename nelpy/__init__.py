"""
nelpy
=====

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

from .objects import (EventArray,
                      EpochArray,
                      AnalogSignal,
                      AnalogSignalArray,
                      SpikeTrain,
                      SpikeTrainArray,
                      BinnedSpikeTrain,
                      BinnedSpikeTrainArray)

# TODO: decide on which utils to expose:
# from .utils import (find_nearest_idx,
#                     find_nearest_indices)

# from .hmmutils import PoissonHMM

# from .plotting import plot

__version__ = '0.0.8'
