"""
nelpy.io
========

This is the nelpy IO module.
"""

# TODO: add file IO utils such as mkdir, getwd, load, save, glob, etc.
# also possibly have examples and support for Jagular
# also add hdf5 support, especially for pandas

from . import hc3
from . import hc18
from . import hc11
from . import matlab
from . import neuralynx
from . import neo
from . import miniscopy
from . import brian

# from . import jagular

__all__ = ["hc3", "hc18", "hc11", "matlab", "neuralynx", "neo", "miniscopy", "brian"]

__version__ = "0.0.3"
