"""
nelpy auxiliary objects

``nelpy`` is a neuroelectrophysiology object model and data analysis
suite.
"""

# __all__ = ['TuningCurve1D',
#            'DirectionalTuningCurve1D',
#            'Session']

from ._tuningcurve import *
from ._session import Session
from ._results import *
from ._imu import IMUSensorArray
from ._position import PositionArray