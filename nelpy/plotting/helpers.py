"""This file contains helper classes and functions for the nelpy
plotting package.
"""

import matplotlib.artist as artist
from matplotlib.axes import Axes


class RasterLabelData(artist.Artist):
    """
    Helper class for storing label data for raster plots in nelpy.

    Attributes
    ----------
    label_data : dict
        Dictionary mapping unit_id to (unit_loc, unit_label).
    yrange : list
        List representing the y-range for the raster labels.
    """

    def __init__(self):
        """
        Initialize a RasterLabelData object.
        """
        self.label_data = {}  # (k, v) = (unit_id, (unit_loc, unit_label))
        artist.Artist.__init__(self)
        self.yrange = []

    def __repr__(self):
        """
        Return a string representation of the RasterLabelData object.
        """
        return "<nelpy.RasterLabelData at " + str(hex(id(self))) + ">"

    @property
    def label_data(self):
        """
        Get the label data dictionary.

        Returns
        -------
        dict
            Dictionary mapping unit_id to (unit_loc, unit_label).
        """
        return self._label_data

    @label_data.setter
    def label_data(self, val):
        """
        Set the label data dictionary.

        Parameters
        ----------
        val : dict
            Dictionary mapping unit_id to (unit_loc, unit_label).
        """
        self._label_data = val

    @property
    def yrange(self):
        """
        Get the y-range for the raster labels.

        Returns
        -------
        list
            The y-range list.
        """
        return self._yrange

    @yrange.setter
    def yrange(self, val):
        """
        Set the y-range for the raster labels.

        Parameters
        ----------
        val : list
            The y-range list.
        """
        self._yrange = val


class NelpyAxes(Axes):
    """
    Custom Axes class for nelpy plotting extensions.
    """

    def __init__(self, **kwargs):
        """
        Initialize a NelpyAxes object.
        """
        Axes.__init__(self, **kwargs)
        self._empty = None

    @property
    def isempty(self):
        """
        Check if the axes is empty.

        Returns
        -------
        bool or None
            True if empty, False otherwise, or None if not set.
        """
        return self._empty

    def _as_mpl_axes(self):
        """
        Not implemented: Convert back to pure matplotlib.axes.Axes.

        Raises
        ------
        NotImplementedError
            Always raised, as this is not yet supported.
        """
        raise NotImplementedError(
            "converting back to pure matplotlib.axes.Axes not yet supported!"
        )
