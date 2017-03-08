"""This module contains helper functions and utilities for nelpy."""

__all__ = ['swap_cols',
           'swap_rows',
           'pairwise',
           'is_sorted',
           'linear_merge',
           'PrettyDuration',
           'get_contiguous_segments',
           'get_events_boundaries']

import numpy as np
import warnings
from itertools import tee
from collections import namedtuple
from math import floor

def swap_cols(arr, frm, to):
    """swap columns of a 2D np.array"""
    if arr.ndim > 1:
        arr[:,[frm, to]] = arr[:,[to, frm]]
    else:
        arr[frm], arr[to] = arr[to], arr[frm]

def swap_rows(arr, frm, to):
    """swap rows of a 2D np.array"""
    if arr.ndim > 1:
        arr[[frm, to],:] = arr[[to, frm],:]
    else:
        arr[frm], arr[to] = arr[to], arr[frm]

def pairwise(iterable):
    """returns a zip of all neighboring pairs.
    This is used as a helper function for is_sorted.

    Example
    -------
    >>> mylist = [2, 3, 6, 8, 7]
    >>> list(pairwise(mylist))
    [(2, 3), (3, 6), (6, 8), (8, 7)]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def is_sorted(iterable, key=lambda a, b: a <= b):
    """Check to see if iterable is monotonic increasing (sorted)."""
    return all(key(a, b) for a, b in pairwise(iterable))

def linear_merge(list1, list2):
    """Merge two SORTED lists in linear time.

    Returns a generator of the merged result.

    Examples
    --------
    >>> a = [1, 3, 5, 7]
    >>> b = [2, 4, 6, 8]
    >>> [i for i in linear_merge(a, b)]
    [1, 2, 3, 4, 5, 6, 7, 8]

    >>> [i for i in linear_merge(b, a)]
    [1, 2, 3, 4, 5, 6, 7, 8]

    >>> a = [1, 2, 2, 3]
    >>> b = [2, 2, 4, 4]
    >>> [i for i in linear_merge(a, b)]
    [1, 2, 2, 2, 2, 3, 4, 4]
    """

    # if any of the lists are empty, return the other (possibly also
    # empty) list: (this is necessary because having either list1 or
    # list2 be empty makes this quite a bit more complicated...)
    if isinstance(list1, (list, np.ndarray)):
        if len(list1) == 0:
            list2 = iter(list2)
            while True:
                yield next(list2)
    if isinstance(list2, (list, np.ndarray)):
        if len(list2) == 0:
            list1 = iter(list1)
            while True:
                yield next(list1)

    list1 = iter(list1)
    list2 = iter(list2)

    value1 = next(list1)
    value2 = next(list2)

    # We'll normally exit this loop from a next() call raising
    # StopIteration, which is how a generator function exits anyway.
    while True:
        if value1 <= value2:
            # Yield the lower value.
            yield value1
            try:
                # Grab the next value from list1.
                value1 = next(list1)
            except StopIteration:
                # list1 is empty.  Yield the last value we received from list2, then
                # yield the rest of list2.
                yield value2
                while True:
                    yield next(list2)
        else:
            yield value2
            try:
                value2 = next(list2)

            except StopIteration:
                # list2 is empty.
                yield value1
                while True:
                    yield next(list1)

def get_contiguous_segments(data,step=None, sort=False):
    """Compute contiguous segments (seperated by step) in a list.

    WARNING! This function assumes that a sorted list is passed.
    If this is not the case (or if it is uncertain), use sort=True
    to force the list to be sorted first.

    Returns an array of size (n_segments, 2), with each row
    being of the form ([start, stop]) inclusive.
    """
    from itertools import groupby
    from operator import itemgetter

    if step is None:
        step = 1
    if sort:
        data = np.sort(data)  # below groupby algorithm assumes sorted list
    if np.any(np.diff(data) < step):
        warnings.warn("some steps in the data are smaller than the requested step size.")

    bdries = []

    for k, g in groupby(enumerate(data), lambda ix: (round(100*step*ix[0] - 100*ix[1])//10)):
        f = itemgetter(1)
        gen = (f(x) for x in g)
        start = next(gen)
        stop = start
        for stop in gen:
            pass
        bdries.append([start, stop])

    return np.asarray(bdries)

class PrettyDuration(float):
    """Time duration with pretty print"""

    def __init__(self, seconds):
        self.duration = seconds

    def __str__(self):
        return self.time_string(self.duration)

    @staticmethod
    def to_dhms(seconds):
        """convert seconds into hh:mm:ss:ms"""
        ms = seconds % 1; ms = round(ms*1000)
        seconds = floor(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        Time = namedtuple('Time', 'dd hh mm ss ms')
        time = Time(dd=d, hh=h, mm=m, ss=s, ms=ms)
        return time

    @staticmethod
    def time_string(seconds):
        """returns a formatted time string."""
        dd, hh, mm, ss, s = PrettyDuration.to_dhms(seconds)
        if s > 0:
            sstr = ".{}".format(int(s))
        else:
            sstr = ""
        if dd > 0:
            daystr = "{:01d} days ".format(dd)
        else:
            daystr = ""
        if hh > 0:
            timestr = daystr + "{:01d}:{:02d}:{:02d}{} hours".format(hh, mm, ss, sstr)
        elif mm > 0:
            timestr = daystr + "{:01d}:{:02d}{} minutes".format(mm, ss, sstr)
        elif ss > 0:
            timestr = daystr + "{:01d}{} seconds".format(ss, sstr)
        else:
            timestr = daystr +"{} milliseconds".format(s)
        return timestr


def find_threshold_crossing_events(x, threshold):
    """Find threshold crossing events.
    """
    from itertools import groupby
    from operator import itemgetter

    above_threshold = np.where(x > threshold, 1, 0)
    eventlist = []
    eventmax = []
    for k,v in groupby(enumerate(above_threshold),key=itemgetter(1)):
        if k:
            v = list(v)
            eventlist.append([v[0][0],v[-1][0]])
            try :
                eventmax.append(x[v[0][0]:(v[-1][0]+1)].max())
            except :
                print(v, x[v[0][0]:v[-1][0]])
    eventmax = np.asarray(eventmax)
    eventlist = np.asarray(eventlist)
    return eventlist, eventmax

def get_events_boundaries(x, PrimaryThreshold=None, SecondaryThreshold=None):
    """get event boundaries such that event.max >= PrimaryThreshold
    and the event extent is defined by SecondaryThreshold.

    Note that when PrimaryThreshold==SecondaryThreshold, then this is a
    simple threshold crossing algorithm.

    returns bounds, maxes, events
        where bounds <==> SecondaryThreshold to SecondaryThreshold
              maxes  <==> maximum value during each event
              events <==> PrimaryThreshold to PrimaryThreshold
    """

    if PrimaryThreshold is None: # by default, threshold is 3 SDs above mean of x
        PrimaryThreshold = np.mean(x) + 3*np.std(x)

    if SecondaryThreshold is None: # by default, revert back to mean of x
        SecondaryThreshold = np.mean(x) # + 0*np.std(x)

    events, _ = find_threshold_crossing_events(x, PrimaryThreshold)

    if len(events) == 0:
        bounds, maxes, events = [], [], []
        warnings.warn("no events satisfied criteria")
        return bounds, maxes, events

    # Find periods where value is > SecondaryThreshold; note that the previous periods should be within these!
    assert SecondaryThreshold <= PrimaryThreshold, "Secondary Threshold by definition should include more data than Primary Threshold"

    bounds, broader_maxes = find_threshold_crossing_events(x, SecondaryThreshold)

    # Find corresponding big windows for potential events
    #  Specifically, look for closest left edge that is just smaller
    outer_boundary_indices = np.searchsorted(bounds[:,0], events[:,0])
    #  searchsorted finds the index after, so subtract one to get index before
    outer_boundary_indices = outer_boundary_indices - 1

    # Find extended boundaries for events by pairing to larger windows
    #   (Note that there may be repeats if the larger window contains multiple > 3SD sections)
    bounds = bounds[outer_boundary_indices,:]
    maxes = broader_maxes[outer_boundary_indices]

    # Now, since all that we care about are the larger windows, so we should get rid of repeats
    _, unique_idx = np.unique(bounds[:,0], return_index=True)
    bounds = bounds[unique_idx,:] # SecondaryThreshold to SecondaryThreshold
    maxes = maxes[unique_idx]     # maximum value during event
    events = events[unique_idx,:] # PrimaryThreshold to PrimaryThreshold

    return bounds, maxes, events

########################################################################
# uncurated below this line!
########################################################################

def find_nearest_idx(array, val):
    """Finds nearest index in array to value.

    Parameters
    ----------
    array : np.array
    val : float

    Returns
    -------
    Index into array that is closest to val

    """
    return (np.abs(array-val)).argmin()


def find_nearest_indices(array, vals):
    """Finds nearest index in array to value.

    Parameters
    ----------
    array : np.array
        This is the array you wish to index into.
    vals : np.array
        This is the array that you are getting your indices from.

    Returns
    -------
    Indices into array that is closest to vals.

    Notes
    -----
    Wrapper around find_nearest_idx().

    """
    return np.array([find_nearest_idx(array, val) for val in vals], dtype=int)

def get_sort_idx(tuning_curves):
    """Finds indices to sort neurons by max firing in tuning curve.

    Parameters
    ----------
    tuning_curves : list of lists
        Where each inner list is the tuning curves for an individual
        neuron.

    Returns
    -------
    sorted_idx : list
        List of integers that correspond to the neuron in sorted order.

    """
    tc_max_loc = []
    for i, neuron_tc in enumerate(tuning_curves):
        tc_max_loc.append((i, np.where(neuron_tc == np.max(neuron_tc))[0][0]))
    sorted_by_tc = sorted(tc_max_loc, key=lambda x: x[1])

    sorted_idx = []
    for idx in sorted_by_tc:
        sorted_idx.append(idx[0])

    return sorted_idx

def cartesian(xcenters, ycenters):
    """Finds every combination of elements in two arrays.

    Parameters
    ----------
    xcenters : np.array
    ycenters : np.array

    Returns
    -------
    cartesian : np.array
        With shape(n_sample, 2).

    """
    return np.transpose([np.tile(xcenters, len(ycenters)), np.repeat(ycenters, len(xcenters))])


def epoch_position(position, epoch):
    """Finds positions associated with epoch times

    Parameters
    ----------
    position : vdmlab.Position
    epoch : vdmlab.Epoch

    Returns
    -------
    epoch_position : vdmlab.Position

    """
    return position.time_slices(epoch.starts, epoch.stops)
