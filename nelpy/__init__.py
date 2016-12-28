"""Nelpy is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html)."""

from .objects import (EpochArray,
                      SpikeTrain)
from .utils import (find_nearest_idx,
                    get_sort_idx,
                    add_scalebar,
                    get_counts,
                    find_nearest_indices,
                    cartesian,
                    epoch_position)

__version__ = '0.0.5'