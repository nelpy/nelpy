"""
nelpy full API
==============

``nelpy`` is a neuroelectrophysiology object model and data analysis suite
based on the python-vdmlab project (https://github.com/mvdm/vandermeerlab),
and inspired by the neuralensemble.org NEO project
(see http://neo.readthedocs.io/en/0.4.0/core.html).
"""

from .core import *  # NOTE: control exported symbols in objects.py

from . import filtering
from . import hmmutils
from . import io
from . import decoding
from . import scoring
from . import plotting
from . import utils

from . version import __version__
