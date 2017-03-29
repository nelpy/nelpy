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
from scipy.signal import hilbert
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter
from numpy import log, ceil
import copy

from . import objects # so that objects.AnalogSignalArray is exposed

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

def rmse(predictions, targets):
    """Calculate the root mean squared error of an array of predictions.

    Parameters
    ----------
    predictions : array_like
        Array of predicted values.
    targets : array_like
        Array of target values.

    Returns
    -------
    rmse: float
        Root mean squared error of the predictions wrt the targets.
    """
    predictions = np.asanyarray(predictions)
    targets = np.asanyarray(targets)
    rmse = np.sqrt(((predictions - targets) ** 2).mean())
    return rmse

def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

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

class PrettyInt(int):
    """Prints integers in a more readable format"""

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return '{:,}'.format(self.val)

    def __repr__(self):
        return '{:,}'.format(self.val)

class PrettyDuration(float):
    """Time duration with pretty print"""

    def __init__(self, seconds):
        self.duration = seconds

    def __str__(self):
        return self.time_string(self.duration)

    def __repr__(self):
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
            if mm == 0:
                # in this case, represent milliseconds in terms of
                # seconds (i.e. a decimal)
                sstr = str(s/1000).lstrip('0')
            else:
                # for all other cases, milliseconds will be represented
                # as an integer
                sstr = ":{:03d}".format(int(s))
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

def shrinkMatColsTo(mat, numCols):
    """ Docstring goes here
    Shrinks a NxM1 matrix down to an NxM2 matrix, where M2 <= M1"""
    import scipy.ndimage
    numCells = mat.shape[0]
    numColsMat = mat.shape[1]
    a = np.zeros((numCells, numCols))
    for row in np.arange(numCells):
        niurou = scipy.ndimage.interpolation.zoom(input=mat[row,:], zoom=(numCols/numColsMat), order = 1)
        a[row,:] = niurou
    return a

def find_threshold_crossing_events(x, threshold, *, mode='above'):
    """Find threshold crossing events.

    Parameters
    ----------
    x :
    threshold :
    mode : string, optional in ['above', 'below']; default 'above'
        event triggering above, or below threshold
    """
    from itertools import groupby
    from operator import itemgetter

    if mode == 'below':
        cross_threshold = np.where(x < threshold, 1, 0)
    elif mode == 'above':
        cross_threshold = np.where(x > threshold, 1, 0)
    else:
        raise NotImplementedError(
            "mode {} not understood for find_threshold_crossing_events".format(str(mode)))
    eventlist = []
    eventmax = []
    for k,v in groupby(enumerate(cross_threshold),key=itemgetter(1)):
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

def get_events_boundaries(x, *, PrimaryThreshold=None,
                          SecondaryThreshold=None,
                          minThresholdLength=None, minLength=None,
                          maxLength=None, ds=None, mode='above'):
    """get event boundaries such that event.max >= PrimaryThreshold
    and the event extent is defined by SecondaryThreshold.

    Note that when PrimaryThreshold==SecondaryThreshold, then this is a
    simple threshold crossing algorithm.

    NB. minLength and maxLength are applied to the SecondaryThreshold
        events, whereas minThresholdLength is applied to the
        PrimaryThreshold events.

    Parameters
    ----------
    x :
    PrimaryThreshold : float
    SecondaryThreshold : float
    minThresholdLength : float
    minLength : float
    maxLength : float
    ds : float
    mode : string, optional in ['above', 'below']; default 'above'
        event triggering above, or below threshold

    Returns
    -------
    returns bounds, maxes, events
        where bounds <==> SecondaryThreshold to SecondaryThreshold
              maxes  <==> maximum value during each event
              events <==> PrimaryThreshold to PrimaryThreshold
    """

    x = x.squeeze()
    if x.ndim > 1:
        raise TypeError("multidimensional arrays not supported!")

    if PrimaryThreshold is None: # by default, threshold is 3 SDs above mean of x
        PrimaryThreshold = np.mean(x) + 3*np.std(x)

    if SecondaryThreshold is None: # by default, revert back to mean of x
        SecondaryThreshold = np.mean(x) # + 0*np.std(x)

    events, _ = \
        find_threshold_crossing_events(x=x,
                                       threshold=PrimaryThreshold,
                                       mode=mode)

    # apply minThresholdLength criterion:
    if minThresholdLength is not None and len(events) > 0:
        durations = (events[:,1] - events[:,0] + 1) * ds
        events = events[[durations >= minThresholdLength]]

    if len(events) == 0:
        bounds, maxes, events = [], [], []
        warnings.warn("no events satisfied criteria")
        return bounds, maxes, events

    # Find periods where value is > SecondaryThreshold; note that the previous periods should be within these!
    if mode == 'above':
        assert SecondaryThreshold <= PrimaryThreshold, \
            "Secondary Threshold by definition should include more data than Primary Threshold"
    elif mode == 'below':
        assert SecondaryThreshold >= PrimaryThreshold, \
            "Secondary Threshold by definition should include more data than Primary Threshold"
    else:
        raise NotImplementedError(
            "mode {} not understood for find_threshold_crossing_events".format(str(mode)))

    bounds, broader_maxes = \
        find_threshold_crossing_events(x=x,
                                       threshold=SecondaryThreshold,
                                       mode=mode)

    # Find corresponding big windows for potential events
    #  Specifically, look for closest left edge that is just smaller
    outer_boundary_indices = np.searchsorted(bounds[:,0], events[:,0])
    #  searchsorted finds the index after, so subtract one to get index before
    outer_boundary_indices = outer_boundary_indices - 1

    # Find extended boundaries for events by pairing to larger windows
    #   (Note that there may be repeats if the larger window contains multiple > 3SD sections)
    bounds = bounds[outer_boundary_indices,:]
    maxes = broader_maxes[outer_boundary_indices]

    if minLength is not None and len(events) > 0:
        durations = (bounds[:,1] - bounds[:,0] + 1) * ds
        # TODO: refactor [durations <= maxLength] but be careful about edge cases
        bounds = bounds[[durations >= minLength]]
        maxes = maxes[[durations >= minLength]]
        events = events[[durations >= minLength]]

    if maxLength is not None and len(events) > 0:
        durations = (bounds[:,1] - bounds[:,0] + 1) * ds
        # TODO: refactor [durations <= maxLength] but be careful about edge cases
        bounds = bounds[[durations <= maxLength]]
        maxes = maxes[[durations <= maxLength]]
        events = events[[durations <= maxLength]]

    if len(events) == 0:
        bounds, maxes, events = [], [], []
        warnings.warn("no events satisfied criteria")
        return bounds, maxes, events

    # Now, since all that we care about are the larger windows, so we should get rid of repeats
    _, unique_idx = np.unique(bounds[:,0], return_index=True)
    bounds = bounds[unique_idx,:] # SecondaryThreshold to SecondaryThreshold
    maxes = maxes[unique_idx]     # maximum value during event
    events = events[unique_idx,:] # PrimaryThreshold to PrimaryThreshold

    return bounds, maxes, events

def signal_envelope1D(data, *, sigma=None, fs=None):
    """Docstring goes here

    sigma = 0 means no smoothing (default 4 ms)
    """

    if sigma is None:
        sigma = 0.004   # 4 ms standard deviation
    if fs is None:
        if isinstance(data, (np.ndarray, list)):
            raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, objects.AnalogSignalArray):
            fs = data.fs

    if isinstance(data, (np.ndarray, list)):
        # Compute number of samples to compute fast FFTs
        padlen = nextfastpower(len(data)) - len(data)
        # Pad data
        paddeddata = np.pad(data, (0, padlen), 'constant')
        # Use hilbert transform to get an envelope
        envelope = np.absolute(hilbert(paddeddata))
        # Truncate results back to original length
        envelope = envelope[:len(data)]
        if sigma:
            # Smooth envelope with a gaussian (sigma = 4 ms default)
            EnvelopeSmoothingSD = sigma*fs
            smoothed_envelope = gaussian_filter1d(envelope, EnvelopeSmoothingSD, mode='constant')
            envelope = smoothed_envelope
    elif isinstance(data, objects.AnalogSignalArray):
        # Compute number of samples to compute fast FFTs:
        padlen = nextfastpower(len(data.ydata)) - len(data.ydata)
        # Pad data
        paddeddata = np.pad(data.ydata, (0, padlen), 'constant')
        # Use hilbert transform to get an envelope
        envelope = np.absolute(hilbert(paddeddata))
        # Truncate results back to original length
        envelope = envelope[:len(data.ydata)]
        if sigma:
            # Smooth envelope with a gaussian (sigma = 4 ms default)
            EnvelopeSmoothingSD = sigma*fs
            smoothed_envelope = gaussian_filter1d(envelope, EnvelopeSmoothingSD, mode='constant')
            envelope = smoothed_envelope
        newasa = data.copy()
        newasa._ydata = envelope
        return newasa
    return envelope

def nextpower(n, base=2.0):
    """Return the next integral power of two greater than the given number.
    Specifically, return m such that
        m >= n
        m == 2**x
    where x is an integer. Use base argument to specify a base other than 2.
    This is useful for ensuring fast FFT sizes.

    From https://gist.github.com/bhawkins/4479607 (Brian Hawkins)
    """
    x = base**ceil (log (n) / log (base))
    if type(n) == np.ndarray:
        return np.asarray (x, dtype=int)
    else:
        return int (x)

def nextfastpower(n):
    """Return the next integral power of small factors greater than the given
    number.  Specifically, return m such that
        m >= n
        m == 2**x * 3**y * 5**z
    where x, y, and z are integers.
    This is useful for ensuring fast FFT sizes.

    From https://gist.github.com/bhawkins/4479607 (Brian Hawkins)
    """
    if n < 7:
        return max (n, 1)
    # x, y, and z are all bounded from above by the formula of nextpower.
    # Compute all possible combinations for powers of 3 and 5.
    # (Not too many for reasonable FFT sizes.)
    def power_series (x, base):
        nmax = ceil (log (x) / log (base))
        return np.logspace (0.0, nmax, num=nmax+1, base=base)
    n35 = np.outer (power_series (n, 3.0), power_series (n, 5.0))
    n35 = n35[n35<=n]
    # Lump the powers of 3 and 5 together and solve for the powers of 2.
    n2 = nextpower (n / n35)
    return int (min (n2 * n35))

def smooth_AnalogSignalArray(asa, *, fs=None, sigma=None, bw=None, in_place=False):
    """Smooths a regularly sampled AnalogSignalArray with a Gaussian kernel.

    Smoothing is applied in time, and the same smoothing is applied to each
    signal in the AnalogSignalArray.

    Smoothing is applied within each epoch.

    Parameters
    ----------
    asa : AnalogSignalArray
    fs : float, optional
        Sampling rate (in Hz) of AnalogSignalArray. If not provided, it will
        be obtained from asa.fs
    sigma : float, optional
        Standard deviation of Gaussian kernel, in seconds. Default is 0.05 (50 ms)
    bw : float, optional
        Bandwidth outside of which the filter value will be zero. Default is 4.0
    inplace : bool
        If True the data will be replaced with the smoothed data.
        Default is False.

    Returns
    -------
    out : AnalogSignalArray
        An AnalogSignalArray with smoothed data is returned.
    """

    if fs is None:
        fs = asa.fs
    if fs is None:
        raise ValueError("fs must either be specified, or must be contained in the AnalogSignalArray!")
    if sigma is None:
        sigma = 0.05 # 50 ms default

    sigma = sigma * fs
    bw = 4 # bandwidth of filter (outside of this bandwidth, the filter is zero)

    if not inplace:
        out = copy.deepcopy(asa)
    else:
        out = asa

    cum_lengths = np.insert(np.cumsum(asa.lengths), 0, 0)

    # now smooth each epoch separately
    for idx in range(asa.n_epochs):
        out._ydata[:,cum_lengths[idx]:cum_lengths[idx+1]] = gaussian_filter(asa._ydata[:,cum_lengths[idx]:cum_lengths[idx+1]], sigma=(0,sigma), truncate=bw)

    return out

def dxdt_AnalogSignalArray(asa, *, fs=None, smooth=False, rectify=True, sigma=None, bw=None):
    """Numerical differentiation of a regularly sampled AnalogSignalArray.

    Optionally also smooths result with a Gaussian kernel.

    Smoothing is applied in time, and the same smoothing is applied to each
    signal in the AnalogSignalArray.

    Differentiation, (and if requested, smoothing) is applied within each epoch.

    Parameters
    ----------
    asa : AnalogSignalArray
    fs : float, optional
        Sampling rate (in Hz) of AnalogSignalArray. If not provided, it will
        be obtained from asa.fs
    smooth : bool, optional
        If true, result will be smoothed. Default is False
    rectify : bool, optional
        If True, absolute value of derivative is computed. Default is True.
    sigma : float, optional
        Standard deviation of Gaussian kernel, in seconds. Default is 0.05
        (50 ms).
    bw : float, optional
        Bandwidth outside of which the filter value will be zero. Default is 4.0

    Returns
    -------
    out : AnalogSignalArray
        An AnalogSignalArray with derivative data (in units per second) is returned.
    """

    if fs is None:
        fs = asa.fs
    if fs is None:
        raise ValueError("fs must either be specified, or must be contained in the AnalogSignalArray!")
    if sigma is None:
        sigma = 0.05 # 50 ms default

    out = copy.deepcopy(asa)

    cum_lengths = np.insert(np.cumsum(asa.lengths), 0, 0)

    # now obtain the derivative for each epoch separately
    for idx in range(asa.n_epochs):
        out._ydata[:,cum_lengths[idx]:cum_lengths[idx+1]] = np.gradient(asa._ydata[:,cum_lengths[idx]:cum_lengths[idx+1]], axis=1)

    out._ydata = out._ydata * fs

    if rectify:
        out._ydata = np.abs(out._ydata)

    if smooth:
        out = smooth_AnalogSignalArray(out, fs=fs, sigma=sigma, bw=bw)

    return out

def spiketrain_union(st1, st2):
    """Join two spiketrains together.

    WARNING! This function should be improved a lot!
    """
    assert st1.n_units == st2.n_units
    support = st1.support.join(st2.support)

    newdata = []
    for unit in range(st1.n_units):
        newdata.append(np.append(st1.time[unit], st2.time[unit]))

    return objects.SpikeTrainArray(newdata, support=support, fs=1)

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
