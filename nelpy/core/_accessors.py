"""This file contains generic accessors to handle getting data from core objects"""

import numpy as np
from .. import core

__all__ = ['IntervalSeriesSlicer', 'ItemGetterLoc', 'ItemGetterIloc']

class IntervalSeriesSlicer(object):

    """Note: 'Series' in terms of a data series. Depending on the
    object to which this slicer is attached, this can be interpreted
    differently. For example, signals for RSASAs and units for STAs"""
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """intervals (e.g. epochs), series"""
        # by default, keep all series
        seriesslice = slice(None, None, None)
        if isinstance(*args, int):
            intervalslice = args[0]
        elif isinstance(*args, core.IntervalArray):
            intervalslice = args[0]
        else:
            try:
                slices = np.s_[args][0]
                if len(slices) > 2:
                    raise IndexError("only [intervals, series] slicing is supported at this time!")
                elif len(slices) == 2:
                    intervalslice, seriesslice = slices
                else:
                    intervalslice = slices[0]
            except TypeError:
                # only interval to slice:
                intervalslice = slices

        return intervalslice, seriesslice

class ItemGetterLoc(object):
    """.loc is primarily label based (that is, series_id based)

    .loc will raise KeyError when the items are not found.

    Allowed inputs are:
        - A single label, e.g. 5 or 'a', (note that 5 is interpreted
            as a label of the index. This use is not an integer
            position along the index)
        - A list or array of labels ['a', 'b', 'c']
        - A slice object with labels 'a':'f', (note that contrary to
            usual python slices, both the start and the stop are
            included!)
    """
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, idx):
        """intervals, series"""
        intervalslice, seriesslice = self.obj._slicer[idx]

        # first convert series slice into list
        if isinstance(seriesslice, slice):
            start = seriesslice.start
            stop = seriesslice.stop
            istep = seriesslice.step
            try:
                if start is None:
                    istart = 0
                else:
                    istart = self.obj._series_ids.index(start)
            except ValueError:
                raise KeyError('series_id {} could not be found!'.format(start))
            try:
                if stop is None:
                    istop = self.obj.n_series
                else:
                    istop = self.obj._series_ids.index(stop) + 1
            except ValueError:
                raise KeyError('series_id {} could not be found!'.format(stop))
            if istep is None:
                istep = 1
            if istep < 0:
                istop -=1
                istart -=1
                istart, istop = istop, istart
            series_idx_list = list(range(istart, istop, istep))
        else:
            series_idx_list = []
            seriesslice = np.atleast_1d(seriesslice)
            for series in seriesslice:
                try:
                    uidx = self.obj.series_ids.index(series)
                except ValueError:
                    raise KeyError("series_id {} could not be found!".format(series))
                else:
                    series_idx_list.append(uidx)

        if not isinstance(series_idx_list, list):
            series_idx_list = list(series_idx_list)

        # this is mainly to make code easier to read since the _restrict
        # function prototypes say they accept intervalslice and
        # seriesslies
        seriesslice = series_idx_list

        out = self.obj.copy()
        # It is now the object's responsibility to do the
        # restriction and set its attributes properly
        out._restrict(intervalslice, seriesslice)
        out.__renew__()

        return out

class ItemGetterIloc(object):
    """.iloc is primarily integer/index based (from 0 to length-1
    of the axis).

    .iloc will raise IndexError if a requested indexer is
    out-of-bounds, except slice indexers which allow out-of-bounds
    indexing. (this conforms with python/numpy slice semantics).

    Allowed inputs are:
        - An integer e.g. 5
        - A list or array of integers [4, 3, 0]
        - A slice object with ints 1:7
    """
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, idx):
        """intervals, series"""
        intervalslice, seriesslice = self.obj._slicer[idx]

        if isinstance(seriesslice, int):
            seriesslice = [seriesslice]

        out = self.obj.copy()
        # It is now the object's responsibility to do the
        # restriction and set its attributes properly
        out._restrict(intervalslice, seriesslice)
        out.__renew__()

        return out

