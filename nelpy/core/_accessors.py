"""This file contains generic accessors to handle getting data from core objects"""

import numpy as np

from .. import core

__all__ = ["SliceExtractor", "ItemGetterLoc", "ItemGetterIloc"]


class SliceExtractor(object):
    """
    Extracts and validates slice indices for interval, series, and event dimensions.

    This class is used internally by accessor classes to parse and verify slicing/indexing
    arguments for core objects that support multi-dimensional indexing (e.g., [interval, series, event]).

    Methods
    -------
    extract(idx)
        Parses the input index and returns validated slices for interval and series dimensions.
    verify_interval_slice(testslice)
        Checks if the interval slice is of a supported type.
    verify_series_slice(testslice)
        Checks if the series slice is of a supported type.
    verify_event_slice(testslice)
        Checks if the event slice is of a supported type.

    Examples
    --------
    >>> extractor = SliceExtractor()
    >>> intervalslice, seriesslice = extractor.extract((slice(0, 2), [0, 1]))
    >>> intervalslice
    slice(0, 2, None)
    >>> seriesslice
    [0, 1]

    Notes
    -----
    - Only [interval, series, events] indexing is supported.
    - Event slice extraction is not yet implemented in the return value.
    """
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
        if isinstance(
            idx, (core.EpochArray, core.IntervalArray, int, list, slice, np.ndarray)
        ):
            intervalslice = idx
        elif not isinstance(idx, tuple):
            raise TypeError("A slice of type {} is not supported".format(idx))
        # Multidimensional cases
        elif len(idx) == 2:
            intervalslice = idx[0]
            seriesslice = idx[1]
        elif len(idx) == 3:
            intervalslice = idx[0]
            seriesslice = idx[1]
            eventslice = idx[2]
        elif len(idx) > 3:
            raise ValueError("Only [interval, series, events] indexing is supported")
        else:
            raise ValueError(
                "Some other error occurred that we didn't handle."
                " Please contact a developer"
            )

        self.verify_interval_slice(intervalslice)
        self.verify_series_slice(seriesslice)
        self.verify_event_slice(eventslice)

        # not returning eventslice because haven't implemented yet, but
        # it's in the works
        return intervalslice, seriesslice

    def verify_interval_slice(self, testslice):
        if not isinstance(
            testslice, (int, list, tuple, slice, np.ndarray, core.IntervalArray)
        ):
            raise TypeError(
                "An interval slice of type {} is not supported".format(type(testslice))
            )

    def verify_series_slice(self, testslice):
        if not isinstance(testslice, (int, list, tuple, slice, np.ndarray)):
            raise TypeError(
                "A series slice of type {} is not supported".format(type(testslice))
            )

    def verify_event_slice(self, testslice):
        if not isinstance(testslice, (int, list, tuple, slice, np.ndarray)):
            raise TypeError(
                "An event indexing slice of type {} is not supported".format(
                    type(testslice)
                )
            )

        if isinstance(testslice, slice):
            # Case 1: slice(None, val, stride)
            # Case 2: slice(val, None, stride)
            # Case 3: slice(val1, val2, stride)
            # Only need to check bounds for case 3 but check stride
            # for all cases

            is_start_val = testslice.start is not None
            is_stop_val = testslice.stop is not None
            is_step_val = testslice.step is not None

            if is_start_val and is_stop_val:
                if testslice.stop < testslice.start:
                    raise ValueError(
                        "The stop index {} for event indexing"
                        " must be greater than or equal to"
                        " the start index {}".format(testslice.stop, testslice.start)
                    )

            if is_step_val:
                if testslice.step <= 0:
                    raise ValueError("The stride for event indexing must be positive")


class ItemGetterLoc(object):
    """
    Accessor for label-based indexing, similar to pandas' `.loc`.

    Allows selection of intervals and series by label (e.g., series_id) rather than integer position.
    Raises KeyError if a label is not found.

    Parameters
    ----------
    obj : object
        The parent object supporting label-based indexing (must have `_series_ids` and `series_ids`).

    Examples
    --------
    >>> obj = ...  # some core object with series_ids
    >>> loc = ItemGetterLoc(obj)
    >>> subset = loc['seriesA']
    >>> subset = loc[['seriesA', 'seriesB']]
    >>> subset = loc['seriesA':'seriesC']

    Notes
    -----
    - Slices with labels include both the start and stop labels (unlike standard Python slices).
    - This accessor is typically available as the `.loc` property on core objects.
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
                raise KeyError("series_id {} could not be found!".format(start))
            try:
                if stop is None:
                    istop = self.obj.n_series
                else:
                    istop = self.obj._series_ids.index(stop) + 1
            except ValueError:
                raise KeyError("series_id {} could not be found!".format(stop))
            if istep is None:
                istep = 1
            if istep < 0:
                istop -= 1
                istart -= 1
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
    """
    Accessor for integer-based indexing, similar to pandas' `.iloc`.

    Allows selection of intervals and series by integer position (from 0 to length-1).
    Raises IndexError if an integer index is out of bounds (except for slices, which allow out-of-bounds indices).

    Parameters
    ----------
    obj : object
        The parent object supporting integer-based indexing.

    Examples
    --------
    >>> obj = ...  # some core object
    >>> iloc = ItemGetterIloc(obj)
    >>> subset = iloc[0]
    >>> subset = iloc[[0, 2, 4]]
    >>> subset = iloc[1:5]

    Notes
    -----
    - This accessor is typically available as the `.iloc` property on core objects.
    - Follows Python/NumPy slice semantics for out-of-bounds indices.
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
