"""This file contains helper classes and functions for the nelpy
plotting package.
"""

import matplotlib.artist as artist

class RasterLabelData(artist.Artist):

    def __init__(self):
        self.label_data = {}  # (k, v) = (unit_id, (unit_loc, unit_label))
        artist.Artist.__init__(self)
        self.yrange = []

    def __repr__(self):
        return "<nelpy.RasterLabelData at " + str(hex(id(self))) + ">"

    @property
    def label_data(self):
        return self._label_data

    @label_data.setter
    def label_data(self, val):
        self._label_data = val

    @property
    def yrange(self):
        return self._yrange

    @yrange.setter
    def yrange(self, val):
        self._yrange = val

from matplotlib.axes import Axes

class NelpyAxes(Axes):

    def __init__(self, **kwargs):
        Axes.__init__(self, **kwargs)

        self._empty = None

    @property
    def isempty(self):
        return self._empty

    def _as_mpl_axes(self):
        raise NotImplementedError ('converting back to pure matplotlib.axes.Axes not yet supported!')