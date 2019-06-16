"""This file contains generic accessors to handle getting data from core objects"""

import numpy as np
from .. import core

__all__ = ['SliceExtractor', 'ItemGetterLoc', 'ItemGetterIloc']

class SliceExtractor(object):

    def __init__(self):
        pass

    def extract(self, idx):

        # By default, keep all slices
        intervalslice = slice(None, None, None)
        seriesslice = slice(None, None, None)
        eventslice = slice(None, None, None)

        # The one case this breaks is when the idx is a tuple
        # and no other slices were requested. Otherwise,
        # something like obj[tuple, [7, 8, 9], 4] works

        # Handle special case where only one slice is provided
        if isinstance(idx, (core.IntervalArray, int, list, slice, np.ndarray)):
            intervalslice = idx
        elif not isinstance(idx, tuple):
            raise TypeError("A slice of type {} is not supported".format(idx))
        # Multidimensional cases
        elif len(idx) == 2:
            intervalslice = idx[0]
            seriesslice   = idx[1]
        elif len(idx) == 3:
            intervalslice = idx[0]
            seriesslice   = idx[1]
            eventslice    = idx[2]
        elif len(idx) > 3:
            raise ValueError("Only [interval, series, events]"
                             " indexing is supported")
        else:
            raise ValueError("Some other error occurred that we didn't handle."
                             " Please contact a developer")

        self.verify_interval_slice(intervalslice)
        self.verify_series_slice(seriesslice)
        self.verify_event_slice(eventslice)

        # not returning eventslice because haven't implemented yet, but
        # it's in the works
        return intervalslice, seriesslice

    def verify_interval_slice(self, testslice):
        if not isinstance(testslice, (int, list, tuple, slice,
                                      np.ndarray, core.IntervalArray)):
            raise TypeError("An interval slice of type {}"
                            " is not supported".format(type(testslice)))

    def verify_series_slice(self, testslice):
        if not isinstance(testslice, (int, list, tuple, slice, np.ndarray)):
            raise TypeError("A series slice of type {} is not supported"
                            .format(type(testslice)))

    def verify_event_slice(self, testslice):

        if not isinstance(testslice, (int, list, tuple, slice, np.ndarray)):
            raise TypeError("An event indexing slice of type {}"
                            " is not supported".format(type(testslice)))

        if isinstance(testslice, slice):
            # Case 1: slice(None, val, stride)
            # Case 2: slice(val, None, stride)
            # Case 3: slice(val1, val2, stride)
            # Only need to check bounds for case 3 but check stride
            # for all cases

            is_start_val = (testslice.start is not None)
            is_stop_val = (testslice.stop is not None)
            is_step_val = (testslice.step is not None)

            if is_start_val and is_stop_val:
                if testslice.stop < testslice.start:
                    raise ValueError("The stop index {} for event indexing"
                                     " must be greater than or equal to"
                                     " the start index {}"
                                     .format(testslice.stop, testslice.start))

            if is_step_val:
                if testslice.step <= 0:
                    raise ValueError("The stride for event indexing"
                                     " must be positive")

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
        self.slice_extractor = SliceExtractor()

    def __getitem__(self, idx):
        """intervals, series"""
        intervalslice, seriesslice = self.slice_extractor.extract(idx)

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
        self.slice_extractor = SliceExtractor()

    def __getitem__(self, idx):
        """intervals, series"""
        intervalslice, seriesslice = self.slice_extractor.extract(idx)

        if isinstance(seriesslice, int):
            seriesslice = [seriesslice]

        out = self.obj.copy()
        # It is now the object's responsibility to do the
        # restriction and set its attributes properly
        out._restrict(intervalslice, seriesslice)
        out.__renew__()

        return out

