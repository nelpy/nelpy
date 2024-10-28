"""This module contains helper functions and utilities for nelpy."""

__all__ = [
    "spatial_information",
    "frange",
    "swap_cols",
    "swap_rows",
    "pairwise",
    "is_sorted",
    "linear_merge",
    "PrettyDuration",
    "ddt_asa",
    "get_contiguous_segments",
    "get_events_boundaries",
    "get_threshold_crossing_epochs",
    "_bst_get_bins",
]

import numpy as np
import logging
from itertools import tee
from collections import namedtuple
from math import floor
from scipy.signal import hilbert
import scipy.ndimage.filters  # import gaussian_filter1d, gaussian_filter
from numpy import log, ceil
import copy

from . import core  # so that core.RegularlySampledAnalogSignalArray is exposed

from .utils_.decorators import keyword_deprecation

try:
    from scipy.fft import next_fast_len  # scipy 1.*
except ImportError:
    from scipy.fftpack import next_fast_len  # scipy 0.*

# def sub2ind(array_shape, rows, cols):
#     ind = rows*array_shape[1] + cols
#     ind[ind < 0] = -1
#     ind[ind >= array_shape[0]*array_shape[1]] = -1
#     return ind

# def ind2sub(array_shape, ind):
#     # see also np.unravel_index(ind, array.shape)
#     ind[ind < 0] = -1
#     ind[ind >= array_shape[0]*array_shape[1]] = -1
#     rows = (ind.astype('int') / array_shape[1])
#     cols = ind % array_shape[1]
#     return (rows, cols)


def ragged_array(arr):
    """Takes a list of arrays, and returns a ragged array.

    See https://github.com/numpy/numpy/issues/12468
    """
    n_elem = len(arr)
    out = np.array(n_elem * [None])
    for ii in range(out.shape[0]):
        out[ii] = arr[ii]
    return out


def asa_indices_within_epochs(asa, intervalarray):
    """Return indices of ASA within epochs.

    [[start, stop]
         ...
     [start, stop]]

    so that data can be associated with asa._data[:,start:stop] for each epoch.
    """
    indices = []
    intervalarray = intervalarray[asa.support]
    for interval in intervalarray.merge().data:
        a_start = interval[0]
        a_stop = interval[1]
        frm, to = np.searchsorted(asa._abscissa_vals, (a_start, a_stop))
        indices.append((frm, to))
    indices = np.array(indices, ndmin=2)

    return indices


def frange(start, stop, step):
    """arange with floating point step"""
    # TODO: this function is not very general; we can extend it to work
    # for reverse (stop < start), empty, and default args, etc.
    # there are also many edge cases where this is weird.
    # see https://stackoverflow.com/questions/7267226/range-for-floats
    # for better alternatives.
    num_steps = int(np.floor((stop - start) / step))
    return np.linspace(start, stop, num=num_steps, endpoint=False)


def spatial_information(ratemap, Pi=1):
    """Compute the spatial information and firing sparsity...

    The specificity index examines the amount of information
    (in bits) that a single spike conveys about the animal's
    location (i.e., how well cell firing predicts the animal's
    location).The spatial information content of cell discharge was
    calculated using the formula:
        information content = \Sum P_i(R_i/R)log_2(R_i/R)
    where i is the bin number, P_i, is the probability for occupancy
    of bin i, R_i, is the mean firing rate for bin i, and R is the
    overall mean firing rate.

    In order to account for the effects of low firing rates (with
    fewer spikes there is a tendency toward higher information
    content) or random bursts of firing, the spike firing
    time-series was randomly offset in time from the rat location
    time-series, and the information content was calculated. A
    distribution of the information content based on 100 such random
    shifts was obtained and was used to compute a standardized score
    (Zscore) of information content for that cell. While the
    distribution is not composed of independent samples, it was
    nominally normally distributed, and a Z value of 2.29 was chosen
    as a cut-off for significance (the equivalent of a one-tailed
    t-test with P = 0.01 under a normal distribution).

    Reference(s)
    ------------
    Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
        and Skaggs, W. E. (1994). "Spatial information content and
        reliability of hippocampal CA1 neurons: effects of visual
        input", Hippocampus, 4(4), 410-421.

    Parameters
    ----------
    ratemap : array of shape (n_units, n_bins)
        Rate map in Hz.

    Pi : numpy.ndarray
        A probability distribution over the bins of the rate map. If not
        provided, it is assumed to be uniform.
    Returns
    -------
    si : array of shape (n_units,)
        spatial information (in bits per spike)
    """
    # convert Pi to probability distribution
    Pi[np.isnan(Pi) | (Pi == 0)] = np.finfo(float).eps
    Pi = Pi / (np.sum(Pi) + np.finfo(float).eps)

    ratemap = copy.deepcopy(ratemap)
    # ensure that the ratemap always has nonzero firing rates,
    # otherwise the spatial information might return NaNs:
    ratemap[np.isnan(ratemap) | (ratemap == 0)] = np.finfo(float).eps

    if len(ratemap.shape) == 3:
        # we have 2D tuning curve, (n_units, n_x, n_y)
        R = (ratemap * Pi).sum(axis=1).sum(axis=1)
        Ri = np.transpose(ratemap, (2, 1, 0))
        si = np.sum(np.sum((Pi * ((Ri / R) * np.log2(Ri / R)).T), axis=1), axis=1)
    elif len(ratemap.shape) == 2:
        # we have 1D tuning curve, (n_units, n_x)
        R = (ratemap * Pi).sum(axis=1)  # mean firing rate
        Ri = ratemap.T
        si = np.sum((Pi * ((Ri / R) * np.log2(Ri / R)).T), axis=1)
    else:
        raise TypeError("rate map shape not supported / understood!")

    return si


def information_rate(ratemap, Pi=1):
    """
    This computes the spatial information rate of cell spikes given variable x in
    bits/second.

    Parameters
    ----------
    ratemap : numpy.ndarray
        A firing rate map, any number of dimensions.
    Pi : numpy.ndarray
        A probability distribution over the bins of the rate map. If not
        provided, it is assumed to be uniform.

    Returns
    -------
    ir : numpy.ndarray
        The information rate of the cell, in bits/second.

    """
    # convert Pi to probability distribution
    Pi[np.isnan(Pi) | (Pi == 0)] = np.finfo(float).eps
    Pi = Pi / (np.sum(Pi) + np.finfo(float).eps)

    if len(ratemap.shape) == 3:
        # we have 2D tuning curve, (n_units, n_x, n_y)
        R = (ratemap * Pi).sum(axis=1).sum(axis=1)
    elif len(ratemap.shape) == 2:
        R = (ratemap * Pi).sum(axis=1)
    else:
        raise TypeError("rate map shape not supported / understood!")
    return spatial_information(ratemap, Pi) * R


def spatial_selectivity(ratemap, Pi=1):
    """
    The selectivity measure max(rate)/mean(rate)  of the cell. The more
    tightly concentrated the cell's activity, the higher the selectivity.
    A cell with no spatial tuning at all will have a selectivity of 1.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.
    Pi : numpy.ndarray
        A probability distribution of the occupancy of each bin in the
        rate map. If not provided, the occupancy will be assumed to be uniform.

    Returns
    -------
    out : float
        selectivity
    """
    # convert Pi to probability distribution
    Pi[np.isnan(Pi) | (Pi == 0)] = np.finfo(float).eps
    Pi = Pi / (np.sum(Pi) + np.finfo(float).eps)

    if len(ratemap.shape) == 3:
        # we have 2D tuning curve, (n_units, n_x, n_y)
        R = (ratemap * Pi).sum(axis=1).sum(axis=1)
        max_rate = np.max(ratemap, axis=1).max(axis=1)
    elif len(ratemap.shape) == 2:
        R = (ratemap * Pi).sum(axis=1)
        max_rate = np.max(ratemap, axis=1)
    else:
        raise TypeError("rate map shape not supported / understood!")
    return max_rate / R


def spatial_sparsity(ratemap, Pi=1):
    """Compute the firing sparsity...
    Compute sparsity of a rate map, The sparsity  measure is an adaptation
    to space. The adaptation measures the fraction of the environment  in which
    a cell is  active. A sparsity of, 0.1 means that the place field of the
    cell occupies 1/10 of the area the subject traverses

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.
    Pi : numpy.ndarray
        A probability distribution over the bins of the rate map. If not
        provided, it is assumed to be uniform.
    Returns
    -------
    out : float
        sparsity

    References
    ----------
    .. [2] Skaggs, W. E., McNaughton, B. L., Wilson, M., & Barnes, C. (1996).
    Theta phase precession in hippocampal neuronal populations and the
    compression of temporal sequences. Hippocampus, 6, 149-172.

    Parameters
    ----------

    ratemap : array of shape (n_units, n_bins)
        Rate map in Hz.
    Pi : array of shape (n_bins,)
        Occupancy of the animal.
    Returns
    -------
    sparsity: array of shape (n_units,)
        sparsity (in percent) for each unit
    """

    Pi[np.isnan(Pi) | (Pi == 0)] = np.finfo(float).eps
    Pi = Pi / (np.sum(Pi) + np.finfo(float).eps)

    ratemap = copy.deepcopy(ratemap)
    # ensure that the ratemap always has nonzero firing rates,
    # otherwise the spatial information might return NaNs:
    ratemap[np.isnan(ratemap) | (ratemap == 0)] = np.finfo(float).eps

    if len(ratemap.shape) == 3:
        # we have 2D tuning curve, (n_units, n_x, n_y)
        R = (ratemap * Pi).sum(axis=1).sum(axis=1)
        avg_sqr_rate = np.sum(ratemap**2 * Pi, axis=1).sum(axis=1)

    elif len(ratemap.shape) == 2:
        R = (ratemap * Pi).sum(axis=1)
        avg_sqr_rate = np.sum(ratemap**2 * Pi, axis=1)
    else:
        raise TypeError("rate map shape not supported / understood!")

    return R**2 / avg_sqr_rate


def _bst_get_bins_inside_interval(interval, ds, w=1):
    """(np.array) Return bin edges entirely contained inside an interval.

    Bin edges always start at interval.start, and continue for as many
    bins as would fit entirely inside the interval.

    NOTE 1: there are (n+1) bin edges associated with n bins.

    WARNING: if an interval is smaller than ds, then no bin will be
            associated with the particular interval.

    NOTE 2: nelpy uses half-open intervals [a,b), but if the bin
            width divides b-a, then the bins will cover the entire
            range. For example, if interval = [0,2) and ds = 1, then
            bins = [0,1,2], even though [0,2] is not contained in
            [0,2). There might be numerical precision deviations from this?

    Parameters
    ----------
    interval : EpochArray
        EpochArray containing a single interval with a start, and stop
    ds : float
        Time bin width, in seconds.
    w : number of bins to use in a sliding window mode. Default is 1 (no sliding window).
        For example, 40 ms bins, with a stride of 5 ms, can be achieved by using (ds=0.005, w=8)
        For now, w has to be an integer, and therefore 5 second bins, with a stride of 2 seconds
        are not supported within this framework.

    Returns
    -------
    bins : array
        Bin edges in an array of shape (n+1,) where n is the number
        of bins
    centers : array
        Bin centers in an array of shape (n,) where n is the number
        of bins
    """

    if interval.length < ds:
        return None, None

    n_bins = int(np.floor(interval.length / ds))  # number of bins

    # linspace is better than arange for non-integral steps
    bins = np.linspace(interval.start, interval.start + n_bins * ds, n_bins + 1)

    if w > 1:
        wn_bins = np.max((1, n_bins - w + 1))
        wn_bins = bins[: wn_bins + 1] + w / 2 * ds - ds / 2
        bins = wn_bins

    centers = bins[:-1] + (ds / 2)

    return bins, centers


def _bst_get_bins(intervalArray, ds, w=1):
    """
    Docstring goes here. TBD. For use with bins that are contained
    wholly inside the intervals.

    """
    b = []  # bin list
    c = []  # centers list
    left_edges = []
    right_edges = []
    counter = 0
    for interval in intervalArray:
        bins, centers = _bst_get_bins_inside_interval(interval=interval, ds=ds, w=w)
        if bins is not None:
            left_edges.append(counter)
            counter += len(centers) - 1
            right_edges.append(counter)
            counter += 1
            b.extend(bins.tolist())
            c.extend(centers.tolist())
    bins = np.array(b)
    bin_centers = np.array(c)

    le = np.array(left_edges)
    le = le[:, np.newaxis]
    re = np.array(right_edges)
    re = re[:, np.newaxis]
    binned_support = np.hstack((le, re))
    lengths = np.atleast_1d((binned_support[:, 1] - binned_support[:, 0] + 1).squeeze())
    support_starts = bins[np.insert(np.cumsum(lengths + 1), 0, 0)[:-1]]
    support_stops = bins[np.insert(np.cumsum(lengths + 1) - 1, 0, 0)[1:]]
    supportdata = np.vstack([support_starts, support_stops]).T
    support = type(intervalArray)(supportdata)  # set support to TRUE bin support

    return bins, bin_centers, binned_support, support


@keyword_deprecation(replace_x_with_y={"bw": "truncate"})
def get_mua(st, ds=None, sigma=None, truncate=None, _fast=True):
    """Compute the multiunit activity (MUA) from a spike train.

    Parameters
    ----------
    st : SpikeTrainArray
        SpikeTrainArray containing one or more units.
     -- OR --
    st : BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing multiunit activity.
    ds : float, optional
        Time step in which to bin spikes. Default is 1 ms.
    sigma : float, optional
        Standard deviation (in seconds) of Gaussian smoothing kernel.
        Default is 10 ms. If sigma==0 then no smoothing is applied.
    truncate : float, optional
        Bandwidth of the Gaussian filter. Default is 6.

    Returns
    -------
    mua : AnalogSignalArray
        AnalogSignalArray with MUA.
    """

    if ds is None:
        ds = 0.001  # 1 ms bin size
    if sigma is None:
        sigma = 0.01  # 10 ms standard deviation
    if truncate is None:
        truncate = 6

    if isinstance(st, core.EventArray):
        # bin spikes, so that we can count the spikes
        mua_binned = st.bin(ds=ds).flatten()
    elif isinstance(st, core.BinnedEventArray):
        mua_binned = st.flatten()
        ds = mua_binned.ds
    else:
        raise TypeError("st has to be one of (SpikeTrainArray, BinnedSpikeTrainArray)")

    # make sure data type is float, so that smoothing works, and convert to rate
    mua_binned._data = mua_binned._data.astype(float) / ds

    # TODO: now that we can simply cast from BST to ASA and back, the following logic could be simplified:
    # put mua rate inside an AnalogSignalArray
    if _fast:
        mua = core.AnalogSignalArray([], empty=True)
        mua._data = mua_binned.data
        mua._abscissa_vals = mua_binned.bin_centers
        mua._abscissa.support = mua_binned.support
    else:
        mua = core.AnalogSignalArray(
            mua_binned.data, timestamps=mua_binned.bin_centers, fs=1 / ds
        )

    mua._fs = 1 / ds

    if (sigma != 0) and (truncate > 0):
        mua = gaussian_filter(mua, sigma=sigma, truncate=truncate)

    return mua


def is_odd(n):
    """Returns True if n is odd, and False if n is even.
    Assumes integer.
    """
    return bool(n & 1)


def swap_cols(arr, frm, to):
    """swap columns of a 2D np.array"""
    if arr.ndim > 1:
        arr[:, [frm, to]] = arr[:, [to, frm]]
    else:
        arr[frm], arr[to] = arr[to], arr[frm]


def swap_rows(arr, frm, to):
    """swap rows of a 2D np.array"""
    if arr.ndim > 1:
        arr[[frm, to], :] = arr[[to, frm], :]
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


def argsort(seq):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)


def is_sorted_general(iterable, key=lambda a, b: a <= b):
    """Check to see if iterable is monotonic increasing (sorted)."""
    return all(key(a, b) for a, b in pairwise(iterable))


def is_sorted(x, chunk_size=None):
    """Returns True if iterable is monotonic increasing (sorted).

    NOTE: intended for 1D array, list or tuple. Will not work on
    more than 1D

    This function works in-core with memory footrpint XXX.
    chunk_size = 100000 is probably a good choice.
    """

    if not isinstance(x, (tuple, list, np.ndarray)):
        raise TypeError("Unsupported type {}".format(type(x)))

    x = np.atleast_1d(np.array(x).squeeze())
    if x.ndim > 1:
        raise ValueError("Input x must be 1-dimensional")

    if chunk_size is None:
        chunk_size = 500000
    stop = x.size
    for chunk_start in range(0, stop, chunk_size):
        chunk_stop = int(min(stop, chunk_start + chunk_size + 1))
        chunk = x[chunk_start:chunk_stop]
        if not np.all(chunk[:-1] <= chunk[1:]):
            return False
    return True


def linear_merge(list1, list2):
    """Merge two SORTED lists in linear time.

    UPDATED TO WORK WITH PYTHON 3.7+ (see https://stackoverflow.com/questions/51700960/runtimeerror-generator-raised-stopiteration-every-time-i-try-to-run-app)

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
                try:
                    yield next(list2)
                except StopIteration:
                    return
    if isinstance(list2, (list, np.ndarray)):
        if len(list2) == 0:
            list1 = iter(list1)
            while True:
                try:
                    yield next(list1)
                except StopIteration:
                    return

    list1 = iter(list1)
    list2 = iter(list2)

    value1 = next(list1)
    value2 = next(list2)

    # We'll normally exit this loop from a next() call raising
    # StopIteration, which is how a generator function exits anyway.
    while True:
        if value1 <= value2:
            # Yield the lower value.
            try:
                yield value1
            except StopIteration:
                return
            try:
                # Grab the next value from list1.
                value1 = next(list1)
            except StopIteration:
                # list1 is empty.  Yield the last value we received from list2, then
                # yield the rest of list2.
                try:
                    yield value2
                except StopIteration:
                    return
                while True:
                    try:
                        yield next(list2)
                    except StopIteration:
                        return
        else:
            try:
                yield value2
            except StopIteration:
                return
            try:
                value2 = next(list2)

            except StopIteration:
                # list2 is empty.
                try:
                    yield value1
                except StopIteration:
                    return
                while True:
                    try:
                        yield next(list1)
                    except StopIteration:
                        return


def get_mua_events(
    mua,
    fs=None,
    minLength=None,
    maxLength=None,
    PrimaryThreshold=None,
    minThresholdLength=None,
    SecondaryThreshold=None,
):
    """Determine MUA/PBEs from multiunit activity.

    MUA : multiunit activity
    PBE : population burst event

    Parameters
    ----------
    mua : AnalogSignalArray
        AnalogSignalArray with one signal, namely the multiunit firing rate [in Hz].
    fs : float, optional
        Sampling frequency of mua, in Hz. If not specified, it will be inferred from
        mua.fs
    minLength : float, optional
    maxLength : float, optional
    PrimaryThreshold : float, optional
    SecondaryThreshold : float, optional
    minThresholdLength : float, optional

    Returns
    -------
    mua_epochs : EpochArray
        EpochArray containing all the MUA events / PBEs.

    Example
    -------
    mua = get_mua(spiketrain)
    mua_epochs = get_mua_events(mua)
    PBEs = get_PBEs(spiketrain, min_active=5)
         = get_PBEs(get_mua_events(get_mua(*)), spiketrain, min_active=5)
    """

    if fs is None:
        fs = mua.fs
    if fs is None:
        raise ValueError("fs must either be specified, or must be contained in mua!")

    if PrimaryThreshold is None:
        PrimaryThreshold = mua.mean() + 3 * mua.std()
    if SecondaryThreshold is None:
        SecondaryThreshold = mua.mean()
    if minLength is None:
        minLength = 0.050  # 50 ms minimum event duration
    if maxLength is None:
        maxLength = 0.750  # 750 ms maximum event duration
    if minThresholdLength is None:
        minThresholdLength = 0.0

    # determine MUA event bounds:
    mua_bounds_idx, maxes, _ = get_events_boundaries(
        x=mua.data,
        PrimaryThreshold=PrimaryThreshold,
        SecondaryThreshold=SecondaryThreshold,
        minThresholdLength=minThresholdLength,
        minLength=minLength,
        maxLength=maxLength,
        ds=1 / fs,
    )

    if len(mua_bounds_idx) == 0:
        logging.warning("no mua events detected")
        return core.EpochArray(empty=True)

    # store MUA bounds in an EpochArray
    mua_epochs = core.EpochArray(mua.time[mua_bounds_idx])

    return mua_epochs


@keyword_deprecation(replace_x_with_y={"bw": "truncate"})
def get_PBEs(
    data,
    fs=None,
    ds=None,
    sigma=None,
    truncate=None,
    unsorted_id=0,
    min_active=None,
    minLength=None,
    maxLength=None,
    PrimaryThreshold=None,
    minThresholdLength=None,
    SecondaryThreshold=None,
):
    """Determine PBEs from multiunit activity or spike trains.

    Definitions
    -----------
    MUA : multiunit activity
    PBE : population burst event

    Summary
    -------
    This function can be used to identify PBE epochs from spike trains, binned
    spike trains, or multiunit activity (in the form of an AnalogSignalArray).

    It is recommended to either pass in a SpikeTrainArray or a
    BinnedSpikeTrainArray, so that a `min_active` number of sorted units can be
    set.

    It is also recommended that the unsorted units (but not noise artifacts!)
    should be included in the spike train that is used to estimate the PBEs. By
    default, unit_id=0 is assumed to be unsorted, but this can be changed, or if
    no unsorted units are present, you can set unsorted_id=None. Equivalently,
    if min_active=0, then no restriction will apply, and the unsorted_id will
    have no effect on the final PBE epochs.

    Examples
    --------
    PBE_epochs = get_PBEs(mua_asa)
    PBE_epochs = get_PBEs(spiketrain, min_active=5)
    PBE_epochs = get_PBEs(binnedspiketrain, min_active=5)

    Parameters
    ----------
    data : AnalogSignalArray
        AnalogSignalArray with one signal, namely the multiunit firing rate [in Hz].
     -- OR --
    data : SpikeTrainArray
        SpikeTrainArray with multiple units, including unsorted unit(s), but
        excluding any noise artifects.
     -- OR --
    data : BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing multiunit activity.
    fs : float, optional
        Sampling frequency of mua, in Hz. If not specified, it will be inferred
        from data.
    ds : float, optional
        Time step in which to bin spikes. Default is 1 ms.
    sigma : float, optional
        Standard deviation (in seconds) of Gaussian smoothing kernel.
        Default is 10 ms. If sigma==0 then no smoothing is applied.
    truncate : float, optional
        Bandwidth of the Gaussian filter. Default is 6.
    unsorted_id : int, optional
        unit_id of the unsorted unit. Default is 0. If no unsorted unit is
        present, then set unsorted_id = None
    min_active : int, optional
        Minimum number of active units per event, excluding unsorted unit.
        Default is 5.
    minLength : float, optional
        Minimum event duration in seconds. Default is 50 ms.
    maxLength : float, optional
        Maximum event duration in seconds. Default is 750 ms.
    PrimaryThreshold : float, optional
        Primary threshold to exceed. Default is mean() + 3*std()
    SecondaryThreshold : float, optional
        Secondary threshold to fall back to. Default is mean().
    minThresholdLength : float, optional
        Minimum duration to stay above PrimaryThreshold. Default is 0 ms.

    Returns
    -------
    PBE_epochs : EpochArray
        EpochArray containing all the PBEs.

    Future improvements
    -------------------
    As of now, it is possible, but not easy to specify the Primary and Secondary
    thresholds for event detection. A slight change in API might be needed to
    make this specification more flexible.
    """

    if sigma is None:
        sigma = 0.01  # 10 ms standard deviation
    if truncate is None:
        truncate = 6

    if isinstance(data, core.AnalogSignalArray):
        # if we have only mua, then we cannot set (ds, unsorted_id, min_active)
        if ds is not None:
            raise ValueError(
                "if data is an AnalogSignalArray then ds cannot be specified!"
            )
        if unsorted_id:
            raise ValueError(
                "if data is an AnalogSignalArray then unsorted_id cannot be specified!"
            )
        if min_active is not None:
            raise ValueError(
                "if data is an AnalogSignalArray then min_active cannot be specified!"
            )
        mua = data
        mua._data = mua._data.astype(float)
        if (sigma != 0) and (truncate > 0):
            mua = gaussian_filter(mua, sigma=sigma, truncate=truncate)

    elif isinstance(data, (core.EventArray, core.BinnedEventArray)):
        # set default parameter values:
        if ds is None:
            ds = 0.001  # default 1 ms
        if min_active is None:
            min_active = 5
        mua = get_mua(data, ds=ds, sigma=sigma, truncate=truncate, _fast=True)
    else:
        raise TypeError(
            "data has to be one of (AnalogSignalArray, SpikeTrainArray, BinnedSpikeTrainArray)"
        )

    # set default parameter values:
    if fs is None:
        fs = mua.fs
    if minLength is None:
        minLength = 0.050  # 50 ms minimum event duration
    if maxLength is None:
        maxLength = 0.750  # 750 ms maximum event duration
    if minThresholdLength is None:
        minThresholdLength = 0.0
    # if PrimaryThreshold is None:
    #         PrimaryThreshold =
    # if SecondaryThreshold is None:
    #     SecondaryThreshold =
    PBE_epochs = get_mua_events(
        mua=mua,
        fs=fs,
        minLength=minLength,
        maxLength=maxLength,
        PrimaryThreshold=PrimaryThreshold,
        minThresholdLength=minThresholdLength,
        SecondaryThreshold=SecondaryThreshold,
    )

    # now require min_active number of sorted cells
    if isinstance(data, (core.EventArray, core.BinnedEventArray)):
        if min_active > 0:
            if unsorted_id is not None:
                # remove unsorted unit, if present:
                unit_ids = copy.deepcopy(data.unit_ids)
                try:
                    unit_ids.remove(unsorted_id)
                except ValueError:
                    pass
                # data_ = data._unit_subset(unit_ids)
                data_ = data.loc[:, unit_ids]
            else:
                data_ = data
            # determine number of active units per epoch:
            n_active = np.array([snippet.n_active for snippet in data_[PBE_epochs]])
            active_epochs_idx = np.argwhere(n_active > min_active).squeeze()
            # only keep those epochs where sufficiently many units are active:
            PBE_epochs = PBE_epochs[active_epochs_idx]
    return PBE_epochs


def get_contiguous_segments(
    data,
    *,
    step=None,
    assume_sorted=None,
    in_core=True,
    index=False,
    inclusive=False,
    fs=None,
    sort=None,
    in_memory=None
):
    """Compute contiguous segments (seperated by step) in a list.

    Note! This function requires that a sorted list is passed.
    It first checks if the list is sorted O(n), and only sorts O(n log(n))
    if necessary. But if you know that the list is already sorted,
    you can pass assume_sorted=True, in which case it will skip
    the O(n) check.

    Returns an array of size (n_segments, 2), with each row
    being of the form ([start, stop]) [inclusive, exclusive].

    NOTE: when possible, use assume_sorted=True, and step=1 as explicit
          arguments to function call.

    WARNING! Step is robustly computed in-core (i.e., when in_core is
        True), but is assumed to be 1 when out-of-core.

    Example
    -------
    >>> data = [1,2,3,4,10,11,12]
    >>> get_contiguous_segments(data)
    ([1,5], [10,13])
    >>> get_contiguous_segments(data, index=True)
    ([0,4], [4,7])

    Parameters
    ----------
    data : array-like
        1D array of sequential data, typically assumed to be integral (sample
        numbers).
    step : float, optional
        Expected step size for neighboring samples. Default uses numpy to find
        the median, but it is much faster and memory efficient to explicitly
        pass in step=1.
    assume_sorted : bool, optional
        If assume_sorted == True, then data is not inspected or re-ordered. This
        can be significantly faster, especially for out-of-core computation, but
        it should only be used when you are confident that the data is indeed
        sorted, otherwise the results from get_contiguous_segments will not be
        reliable.
    in_core : bool, optional
        If True, then we use np.diff which requires all the data to fit
        into memory simultaneously, otherwise we use groupby, which uses
        a generator to process potentially much larger chunks of data,
        but also much slower.
    index : bool, optional
        If True, the indices of segment boundaries will be returned. Otherwise,
        the segment boundaries will be returned in terms of the data itself.
        Default is False.
    inclusive : bool, optional
        If True, the boundaries are returned as [(inclusive idx, inclusive idx)]
        Default is False, and can only be used when index==True.

    Deprecated
    ----------
    in_memory : bool, optional
        This is equivalent to the new 'in-core'.
    sort : bool, optional
        This is equivalent to the new 'assume_sorted'
    fs : sampling rate (Hz) used to extend half-open interval support by 1/fs
    """

    # handle deprecated API calls:
    if in_memory:
        in_core = in_memory
        logging.warning("'in_memory' has been deprecated; use 'in_core' instead")
    if sort:
        assume_sorted = sort
        logging.warning("'sort' has been deprecated; use 'assume_sorted' instead")
    if fs:
        step = 1 / fs
        logging.warning("'fs' has been deprecated; use 'step' instead")

    if inclusive:
        assert index, "option 'inclusive' can only be used with 'index=True'"
    if in_core:
        data = np.asarray(data)

        if not assume_sorted:
            if not is_sorted(data):
                data = np.sort(data)  # algorithm assumes sorted list

        if step is None:
            step = np.median(np.diff(data))

        # assuming that data(t1) is sampled somewhere on [t, t+1/fs) we have a 'continuous' signal as long as
        # data(t2 = t1+1/fs) is sampled somewhere on [t+1/fs, t+2/fs). In the most extreme case, it could happen
        # that t1 = t and t2 = t + 2/fs, i.e. a difference of 2 steps.

        if np.any(np.diff(data) < step):
            logging.warning(
                "some steps in the data are smaller than the requested step size."
            )

        breaks = np.argwhere(np.diff(data) >= 2 * step)
        starts = np.insert(breaks + 1, 0, 0)
        stops = np.append(breaks, len(data) - 1)
        bdries = np.vstack((data[starts], data[stops] + step)).T
        if index:
            if inclusive:
                indices = np.vstack((starts, stops)).T
            else:
                indices = np.vstack((starts, stops + 1)).T
            return indices
    else:
        from itertools import groupby
        from operator import itemgetter

        if not assume_sorted:
            if not is_sorted(data):
                # data = np.sort(data)  # algorithm assumes sorted list
                raise NotImplementedError(
                    "out-of-core sorting has not been implemented yet..."
                )

        if step is None:
            step = 1

        bdries = []

        if not index:
            for k, g in groupby(enumerate(data), lambda ix: (ix[0] - ix[1])):
                f = itemgetter(1)
                gen = (f(x) for x in g)
                start = next(gen)
                stop = start
                for stop in gen:
                    pass
                bdries.append([start, stop + step])
        else:
            counter = 0
            for k, g in groupby(enumerate(data), lambda ix: (ix[0] - ix[1])):
                f = itemgetter(1)
                gen = (f(x) for x in g)
                _ = next(gen)
                start = counter
                stop = start
                for _ in gen:
                    stop += 1
                if inclusive:
                    bdries.append([start, stop])
                else:
                    bdries.append([start, stop + 1])
                counter = stop + 1

    return np.asarray(bdries)


def get_direction(asa, *, sigma=None):
    """Return epochs during which an animal was running left to right, or right
    to left.

    Parameters
    ----------
    asa : AnalogSignalArray 1D
        AnalogSignalArray containing the 1D position data.
    sigma : float, optional
        Smoothing to apply to position (x) before computing gradient estimate.
        Default is 0.

    Returns
    -------
    l2r, r2l : EpochArrays
        EpochArrays corresponding to left-to-right and right-to-left movement.
    """
    if sigma is None:
        sigma = 0
    if not isinstance(asa, core.AnalogSignalArray):
        raise TypeError("AnalogSignalArray expected!")
    assert asa.n_signals == 1, "1D AnalogSignalArray expected!"

    direction = dxdt_AnalogSignalArray(asa.smooth(sigma=sigma), rectify=False).data
    direction[direction >= 0] = 1
    direction[direction < 0] = -1
    direction = direction.squeeze()

    l2r = get_contiguous_segments(np.argwhere(direction > 0).squeeze(), step=1)
    l2r[
        :, 1
    ] -= 1  # change bounds from [inclusive, exclusive] to [inclusive, inclusive]
    l2r = core.EpochArray(asa.abscissa_vals[l2r])

    r2l = get_contiguous_segments(np.argwhere(direction < 0).squeeze(), step=1)
    r2l[
        :, 1
    ] -= 1  # change bounds from [inclusive, exclusive] to [inclusive, inclusive]
    r2l = core.EpochArray(asa.abscissa_vals[r2l])

    return l2r, r2l


class PrettyBytes(int):
    """Prints number of bytes in a more readable format"""

    def __init__(self, val):
        self.val = val

    def __str__(self):
        if self.val < 1024:
            return "{} bytes".format(self.val)
        elif self.val < 1024**2:
            return "{:.3f} kilobytes".format(self.val / 1024)
        elif self.val < 1024**3:
            return "{:.3f} megabytes".format(self.val / 1024**2)
        elif self.val < 1024**4:
            return "{:.3f} gigabytes".format(self.val / 1024**3)

    def __repr__(self):
        return self.__str__()


class PrettyInt(int):
    """Prints integers in a more readable format"""

    def __init__(self, val):
        self.val = val

    def __str__(self):
        return "{:,}".format(self.val)

    def __repr__(self):
        return "{:,}".format(self.val)


class PrettyDuration(float):
    """Time duration with pretty print.

    Behaves like a float, and can always be cast to a float.
    """

    def __init__(self, seconds):
        self.duration = seconds

    def __str__(self):
        return self.time_string(self.duration)

    def __repr__(self):
        return self.time_string(self.duration)

    @staticmethod
    def to_dhms(seconds):
        """convert seconds into hh:mm:ss:ms"""
        pos = seconds >= 0
        if not pos:
            seconds = -seconds
        ms = seconds % 1
        ms = round(ms * 10000) / 10
        seconds = floor(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        Time = namedtuple("Time", "pos dd hh mm ss ms")
        time = Time(pos=pos, dd=d, hh=h, mm=m, ss=s, ms=ms)
        return time

    @staticmethod
    def time_string(seconds):
        """returns a formatted time string."""
        if np.isinf(seconds):
            return "inf"
        pos, dd, hh, mm, ss, s = PrettyDuration.to_dhms(seconds)
        if s > 0:
            if mm == 0:
                # in this case, represent milliseconds in terms of
                # seconds (i.e. a decimal)
                sstr = str(s / 1000).lstrip("0")
                if s >= 999.5:
                    ss += 1
                    s = 0
                    sstr = ""
                    # now propagate the carry:
                    if ss == 60:
                        mm += 1
                        ss = 0
                    if mm == 60:
                        hh += 1
                        mm = 0
                    if hh == 24:
                        dd += 1
                        hh = 0
            else:
                # for all other cases, milliseconds will be represented
                # as an integer
                if s >= 999.5:
                    ss += 1
                    s = 0
                    sstr = ""
                    # now propagate the carry:
                    if ss == 60:
                        mm += 1
                        ss = 0
                    if mm == 60:
                        hh += 1
                        mm = 0
                    if hh == 24:
                        dd += 1
                        hh = 0
                else:
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
            timestr = daystr + "{} milliseconds".format(s)
        if not pos:
            timestr = "-" + timestr
        return timestr

    def __add__(self, other):
        """a + b"""
        return PrettyDuration(self.duration + other)

    def __radd__(self, other):
        """b + a"""
        return self.__add__(other)

    def __sub__(self, other):
        """a - b"""
        return PrettyDuration(self.duration - other)

    def __rsub__(self, other):
        """b - a"""
        return other - self.duration

    def __mul__(self, other):
        """a * b"""
        return PrettyDuration(self.duration * other)

    def __rmul__(self, other):
        """b * a"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """a / b"""
        return PrettyDuration(self.duration / other)


def shrinkMatColsTo(mat, numCols):
    """Docstring goes here
    Shrinks a NxM1 matrix down to an NxM2 matrix, where M2 <= M1"""
    import scipy.ndimage

    numCells = mat.shape[0]
    numColsMat = mat.shape[1]
    a = np.zeros((numCells, numCols))
    for row in np.arange(numCells):
        niurou = scipy.ndimage.interpolation.zoom(
            input=mat[row, :], zoom=(numCols / numColsMat), order=1
        )
        a[row, :] = niurou
    return a


def find_threshold_crossing_events(x, threshold, *, mode="above"):
    """Find threshold crossing events. INCLUSIVE

    Parameters
    ----------
    x : numpy array
        Input data
    threshold : float
        The value whose crossing triggers an event
    mode : string, optional in ['above', 'below']; default 'above'
        event triggering above, or below threshold

    Returns
    -------
    eventlist : list
        List containing the indices corresponding to threshold crossings
    eventmax : list
        List containing the maximum value of each event
    """
    from itertools import groupby
    from operator import itemgetter

    if mode == "below":
        cross_threshold = np.where(x <= threshold, 1, 0)
    elif mode == "above":
        cross_threshold = np.where(x >= threshold, 1, 0)
    else:
        raise NotImplementedError(
            "mode {} not understood for find_threshold_crossing_events".format(
                str(mode)
            )
        )
    eventlist = []
    eventmax = []
    for k, v in groupby(enumerate(cross_threshold), key=itemgetter(1)):
        if k:
            v = list(v)
            eventlist.append([v[0][0], v[-1][0]])
            try:
                eventmax.append(x[v[0][0] : (v[-1][0] + 1)].max())
            except Exception:
                print(v, x[v[0][0] : v[-1][0]])
    eventmax = np.asarray(eventmax)
    eventlist = np.asarray(eventlist)
    return eventlist, eventmax


def get_events_boundaries(
    x,
    *,
    PrimaryThreshold=None,
    SecondaryThreshold=None,
    minThresholdLength=None,
    minLength=None,
    maxLength=None,
    ds=None,
    mode="above"
):
    """get event boundaries such that event.max >= PrimaryThreshold
    and the event extent is defined by SecondaryThreshold.

    Note that when PrimaryThreshold==SecondaryThreshold, then this is a
    simple threshold crossing algorithm.

    NB. minLength and maxLength are applied to the SecondaryThreshold
        events, whereas minThresholdLength is applied to the
        PrimaryThreshold events.

    Parameters
    ----------
    x : numpy array
        Input data
    mode : string, optional in ['above', 'below']; default 'above'
        event triggering above, or below threshold
    PrimaryThreshold : float, optional
        If mode=='above', requires that event.max >= PrimaryThreshold
        If mode=='below', requires that event.min <= PrimaryThreshold
    SecondaryThreshold : float, optional
        The value that defines the event extent
    minThresholdLength : float, optional
        Minimum duration for which the PrimaryThreshold is crossed
    minLength : float, optional
        Minimum duration for which the SecondaryThreshold is crossed
    maxLength : float, optional
        Maximum duration for which the SecondaryThreshold is crossed
    ds : float, optional
        Time step of the input data x

    Returns
    -------
    returns bounds, maxes, events
        where bounds <==> SecondaryThreshold to SecondaryThreshold, inclusive
              maxes  <==> maximum value during each event
              events <==> PrimaryThreshold to PrimaryThreshold, inclusive
    """

    # TODO: x must be a numpy array
    # TODO: ds is often used, but we have no default, and no check for when
    #       it is left as None.
    # TODO: the Docstring should equally be improved.

    x = x.squeeze()
    if x.ndim > 1:
        raise TypeError("multidimensional arrays not supported!")

    if PrimaryThreshold is None:  # by default, threshold is 3 SDs above mean of x
        PrimaryThreshold = np.mean(x) + 3 * np.std(x)

    if SecondaryThreshold is None:  # by default, revert back to mean of x
        SecondaryThreshold = np.mean(x)  # + 0*np.std(x)

    events, primary_maxes = find_threshold_crossing_events(
        x=x, threshold=PrimaryThreshold, mode=mode
    )

    # apply minThresholdLength criterion:
    if minThresholdLength is not None and len(events) > 0:
        durations = (events[:, 1] - events[:, 0] + 1) * ds
        events = events[durations >= minThresholdLength]

    if len(events) == 0:
        bounds, maxes, events = [], [], []
        logging.warning("no events satisfied criteria")
        return bounds, maxes, events

    # Find periods where value is > SecondaryThreshold; note that the previous periods should be within these!
    if mode == "above":
        assert (
            SecondaryThreshold <= PrimaryThreshold
        ), "Secondary Threshold by definition should include more data than Primary Threshold"
    elif mode == "below":
        assert (
            SecondaryThreshold >= PrimaryThreshold
        ), "Secondary Threshold by definition should include more data than Primary Threshold"
    else:
        raise NotImplementedError(
            "mode {} not understood for find_threshold_crossing_events".format(
                str(mode)
            )
        )

    bounds, broader_maxes = find_threshold_crossing_events(
        x=x, threshold=SecondaryThreshold, mode=mode
    )

    # Find corresponding big windows for potential events
    #  Specifically, look for closest left edge that is just smaller
    outer_boundary_indices = np.searchsorted(bounds[:, 0], events[:, 0], side="right")
    #  searchsorted finds the index after, so subtract one to get index before
    outer_boundary_indices = outer_boundary_indices - 1

    # Find extended boundaries for events by pairing to larger windows
    #   (Note that there may be repeats if the larger window contains multiple > 3SD sections)
    bounds = bounds[outer_boundary_indices, :]
    maxes = broader_maxes[outer_boundary_indices]

    if minLength is not None and len(events) > 0:
        durations = (bounds[:, 1] - bounds[:, 0] + 1) * ds
        # TODO: refactor [durations <= maxLength] but be careful about edge cases
        bounds = bounds[durations >= minLength]
        maxes = maxes[durations >= minLength]
        events = events[durations >= minLength]

    if maxLength is not None and len(events) > 0:
        durations = (bounds[:, 1] - bounds[:, 0] + 1) * ds
        # TODO: refactor [durations <= maxLength] but be careful about edge cases
        bounds = bounds[durations <= maxLength]
        maxes = maxes[durations <= maxLength]
        events = events[durations <= maxLength]

    if len(events) == 0:
        bounds, maxes, events = [], [], []
        logging.warning("no events satisfied criteria")
        return bounds, maxes, events

    # Now, since all that we care about are the larger windows, so we should get rid of repeats
    _, unique_idx = np.unique(bounds[:, 0], return_index=True)
    bounds = bounds[unique_idx, :]  # SecondaryThreshold to SecondaryThreshold
    maxes = maxes[unique_idx]  # maximum value during event
    events = events[unique_idx, :]  # PrimaryThreshold to PrimaryThreshold

    return bounds, maxes, events


def signal_envelope1D(data, *, sigma=None, fs=None):
    logging.warnings(
        "'signal_envelope1D' is deprecated; use 'signal_envelope_1d' instead!"
    )
    return signal_envelope_1d(data, sigma=sigma, fs=fs)


def signal_envelope_1d(data, *, sigma=None, fs=None):
    """Finds the signal envelope by taking the absolute value
    of the Hilbert transform

    Parameters
    ----------
    data : numpy array, list, or RegularlySampledAnalogSignalArray
        Input data
        If data is a numpy array, it is expected to have shape
        (n_signals, n_samples)
        If data is a list, it is expected to have length n_signals,
        where each sublist has length n_samples, i.e. data is not
        jagged
    sigma : float, optional
        Standard deviation of the Gaussian kernel used to
        smooth the envelope after applying the Hilbert transform.
        Units of seconds. Default is 4 ms
    fs : float, optional
        Sampling rate of the signal

    Returns
    -------
    out : same type as the input object
        An object containing the signal envelope

    TODO: this is not yet epoch-aware!
    UPDATE: this is actually epoch-aware by now!
    """

    if sigma is None:
        sigma = 0.004  # 4 ms standard deviation
    if fs is None:
        if isinstance(data, (np.ndarray, list)):
            raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, core.RegularlySampledAnalogSignalArray):
            fs = data.fs

    if isinstance(data, (np.ndarray, list)):
        data_array = np.array(data)
        n_dims = np.array(data).ndim
        assert n_dims <= 2, "Only 1D signals supported!"
        if n_dims == 1:
            input_data = data_array.reshape((1, data_array.size))
        else:
            input_data = data_array
        n_signals, n_samples = input_data.shape
        # Compute number of samples to compute fast FFTs
        padlen = next_fast_len(n_samples) - n_samples
        # Pad data
        paddeddata = np.hstack((input_data, np.zeros((n_signals, padlen))))
        # Use hilbert transform to get an envelope
        envelope = np.absolute(hilbert(paddeddata, axis=-1))
        # free up memory
        del paddeddata
        # Truncate results back to original length
        envelope = envelope[..., :n_samples]
        if sigma:
            # Smooth envelope with a gaussian (sigma = 4 ms default)
            EnvelopeSmoothingSD = sigma * fs
            smoothed_envelope = scipy.ndimage.filters.gaussian_filter1d(
                envelope, EnvelopeSmoothingSD, mode="constant", axis=-1
            )
            envelope = smoothed_envelope
        if isinstance(data, list):
            envelope = envelope.tolist()
        return envelope
    elif isinstance(data, core.RegularlySampledAnalogSignalArray):
        # Only ASA data of shape (n_signals, n_timepoints) -> 2D currently supported
        assert data.data.ndim == 2
        cum_lengths = np.insert(np.cumsum(data.lengths), 0, 0)

        newasa = data.copy()
        # for segment in data:
        for idx in range(data.n_epochs):
            # print('hilberting epoch {}/{}'.format(idx+1, data.n_epochs))
            segment_data = data._data[:, cum_lengths[idx] : cum_lengths[idx + 1]]
            n_signals, n_samples = segment_data.shape
            # Compute number of samples to compute fast FFTs:
            padlen = next_fast_len(n_samples) - n_samples
            # Pad data
            paddeddata = np.hstack((segment_data, np.zeros((n_signals, padlen))))
            # Use hilbert transform to get an envelope
            envelope = np.absolute(hilbert(paddeddata, axis=-1))
            # free up memory
            del paddeddata
            # Truncate results back to original length
            envelope = envelope[..., :n_samples]
            if sigma:
                # Smooth envelope with a gaussian (sigma = 4 ms default)
                EnvelopeSmoothingSD = sigma * fs
                smoothed_envelope = scipy.ndimage.filters.gaussian_filter1d(
                    envelope, EnvelopeSmoothingSD, mode="constant", axis=-1
                )
                envelope = smoothed_envelope
            newasa._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = np.atleast_2d(
                envelope
            )
        return newasa


def nextpower(n, base=2.0):
    """Return the next integral power of two greater than the given number.
    Specifically, return m such that
        m >= n
        m == 2**x
    where x is an integer. Use base argument to specify a base other than 2.
    This is useful for ensuring fast FFT sizes.

    From https://gist.github.com/bhawkins/4479607 (Brian Hawkins)
    """
    x = base ** ceil(log(n) / log(base))
    if isinstance(n, np.ndarray):
        return np.asarray(x, dtype=int)
    else:
        return int(x)


def nextfastpower(n):
    """Return the next integral power of small factors greater than the given
    number.  Specifically, return m such that
        m >= n
        m == 2**x * 3**y * 5**z
    where x, y, and z are integers.
    This is useful for ensuring fast FFT sizes.

    From https://gist.github.com/bhawkins/4479607 (Brian Hawkins)

    See also http://scipy.github.io/devdocs/generated/scipy.fftpack.next_fast_len.html
    """
    if n < 7:
        return max(n, 1)

    # x, y, and z are all bounded from above by the formula of nextpower.
    # Compute all possible combinations for powers of 3 and 5.
    # (Not too many for reasonable FFT sizes.)
    def power_series(x, base):
        nmax = int(ceil(log(x) / log(base)))
        return np.logspace(0.0, nmax, num=nmax + 1, base=base)

    n35 = np.outer(power_series(n, 3.0), power_series(n, 5.0))
    n35 = n35[n35 <= n]
    # Lump the powers of 3 and 5 together and solve for the powers of 2.
    n2 = nextpower(n / n35)
    return int(min(n2 * n35))


@keyword_deprecation(replace_x_with_y={"bw": "truncate"})
def gaussian_filter(
    obj,
    *,
    fs=None,
    sigma=None,
    truncate=None,
    inplace=False,
    mode=None,
    cval=None,
    within_intervals=False
):
    """Smooths with a Gaussian kernel.

    Smoothing is applied along the abscissa, and the same smoothing is applied to each
    signal in the RegularlySampledAnalogSignalArray, or to each unit in a BinnedSpikeTrainArray.

    Smoothing is applied ACROSS intervals, but smoothing WITHIN intervals is also supported.

    Parameters
    ----------
    obj : RegularlySampledAnalogSignalArray or BinnedSpikeTrainArray.
    fs : float, optional
        Sampling rate (in obj.base_unit^-1) of obj. If not provided, it will
        be inferred.
    sigma : float, optional
        Standard deviation of Gaussian kernel, in obj.base_units. Default is 0.05
        (50 ms if base_unit=seconds).
    truncate : float, optional
        Bandwidth outside of which the filter value will be zero. Default is 4.0.
    inplace : bool
        If True the data will be replaced with the smoothed data.
        Default is False.
    mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
        The mode parameter determines how the array borders are handled,
        where cval is the value when mode is equal to ‘constant’. Default is
        ‘reflect’.
    cval : scalar, optional
        Value to fill past edges of input if mode is ‘constant’. Default is 0.0.
    within_intervals : boolean, optional
        If True, then smooth within each epoch. Otherwise smooth across epochs.
        Default is False.
        Note that when mode = 'wrap', then smoothing within epochs aren't affected
        by wrapping.

    Returns
    -------
    out : same type as obj
        An object with smoothed data is returned.

    """
    if sigma is None:
        sigma = 0.05
    if truncate is None:
        truncate = 4
    if mode is None:
        mode = "reflect"
    if cval is None:
        cval = 0.0

    if not inplace:
        out = copy.deepcopy(obj)
    else:
        out = obj

    if isinstance(out, core.RegularlySampledAnalogSignalArray) or isinstance(
        out, core._analogsignalarray.PositionArray
    ):
        if fs is None:
            fs = out.fs
        if fs is None:
            raise ValueError(
                "fs must either be specified, or must be contained in the {}!".format(
                    out.type_name
                )
            )
    elif isinstance(out, core.BinnedEventArray):
        bst = out
        if fs is None:
            fs = 1 / bst.ds
        if fs is None:
            raise ValueError(
                "fs must either be specified, or must be contained in the {}!".format(
                    out.type_name
                )
            )
    else:
        raise NotImplementedError(
            "gaussian_filter for {} is not yet supported!".format(str(type(out)))
        )

    sigma = sigma * fs

    if not within_intervals:
        # see https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
        # (1) if smoothing across intervals, we work on a merged support
        # (2) build abscissa_vals, including existing ones, and out-of-support ones
        # (3) to smooth U, build auxiliary arrays V and W, with (V=U).nan=0, and (W=1).nan=0
        # (4) Z = smooth(V)/smooth(W)
        # (5) only keep original support, and original abscissa_vals

        if isinstance(
            out,
            (
                core.RegularlySampledAnalogSignalArray,
                core.BinnedEventArray,
                core._analogsignalarray.PositionArray,
            ),
        ):
            support = out._abscissa.support.merge()
            if not support.domain.is_finite:
                support.domain = (
                    support.start,
                    support.stop,
                )  # TODO: #FIXME might come from abscissa definition, and not from support

            missing_abscissa_vals = []
            for interval in ~support:
                missing_vals = frange(interval.start, interval.stop, 1 / fs)
                missing_abscissa_vals.extend(missing_vals)

            if isinstance(
                out,
                (
                    core.RegularlySampledAnalogSignalArray,
                    core._analogsignalarray.PositionArray,
                ),
            ):
                n_signals = out.n_signals
                n_samples = out.n_samples
            elif isinstance(out, core.BinnedEventArray):
                n_signals = out.n_series
                n_samples = out.n_bins

            V = np.zeros((n_signals, n_samples + len(missing_abscissa_vals)))
            W = np.ones(V.shape)
            all_abscissa_vals = np.sort(
                np.append(out._abscissa_vals, missing_abscissa_vals)
            )
            data_idx = np.searchsorted(all_abscissa_vals, out._abscissa_vals)
            missing_idx = np.searchsorted(all_abscissa_vals, missing_abscissa_vals)
            V[:, data_idx] = out.data
            W[:, missing_idx] = 0

            VV = scipy.ndimage.filters.gaussian_filter(
                V, sigma=(0, sigma), truncate=truncate, mode=mode, cval=cval
            )
            WW = scipy.ndimage.filters.gaussian_filter(
                W, sigma=(0, sigma), truncate=truncate, mode=mode, cval=cval
            )

            Z = VV[:, data_idx] / WW[:, data_idx]

            out._data = Z
        else:
            raise NotImplementedError(
                "gaussian_filter across intervals for {} is not yet supported!".format(
                    str(type(out))
                )
            )
    else:  # within intervals:
        cum_lengths = np.insert(np.cumsum(out.lengths), 0, 0)
        out._data = out._data.astype(float)

        if isinstance(out, core.RegularlySampledAnalogSignalArray):
            # now smooth each interval separately
            for idx in range(out.n_intervals):
                out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = (
                    scipy.ndimage.filters.gaussian_filter(
                        out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]],
                        sigma=(0, sigma),
                        truncate=truncate,
                    )
                )
        elif isinstance(out, core.BinnedSpikeTrainArray):
            # now smooth each interval separately
            for idx in range(out.n_epochs):
                out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = (
                    scipy.ndimage.filters.gaussian_filter(
                        out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]],
                        sigma=(0, sigma),
                        truncate=truncate,
                    )
                )
                # out._data[:,cum_lengths[idx]:cum_lengths[idx+1]] = self._smooth_array(out._data[:,cum_lengths[idx]:cum_lengths[idx+1]], w=w)

    return out


@keyword_deprecation(replace_x_with_y={"bw": "truncate"})
def ddt_asa(
    asa, *, fs=None, smooth=False, rectify=True, sigma=None, truncate=None, norm=False
):
    """Numerical differentiation of a regularly sampled AnalogSignalArray.

    Optionally also smooths result with a Gaussian kernel.

    Smoothing is applied in time, and the same smoothing is applied to each
    signal in the AnalogSignalArray.

    Differentiation, (and if requested, smoothing) is applied within each epoch.

    Parameters
    ----------
    asa : nelpy.RegularlySampledAnalogSignalArray
        Input object.
    fs : float, optional
        Sampling rate (in Hz) of input RSASA. If not provided, it will be obtained
        from asa.fs.
    smooth : bool, optional
        If true, result will be smoothed. Default is False
    rectify : bool, optional
        If True, absolute value of derivative is computed. Default is True.
    sigma : float, optional
        Standard deviation of Gaussian kernel, in seconds. Default is 0.05
        (50 ms).
    truncate : float, optional
        Bandwidth outside of which the filter value will be zero. Default is 4.0
    norm: boolean, optional
        If True, then apply the L2 norm to the result.
    Returns
    -------
    out : nelpy.RegularlySampledAnalogSignalArray
        A RegularlySampledAnalogSignalArray with derivative data (in units
        per second) is returned.

    Notes
    -----
    Central differences are used here.
    """

    if not (
        isinstance(asa, core.RegularlySampledAnalogSignalArray)
        or isinstance(asa, core._analogsignalarray.PositionArray)
    ):
        raise TypeError("Input object must be a RegularlySampledAnalogSignalArray!")
    if fs is None:
        fs = asa.fs
    if sigma is None:
        sigma = 0.05  # 50 ms default

    out = asa.copy()
    cum_lengths = np.insert(np.cumsum(asa.lengths), 0, 0)

    # ensure that datatype is float
    # TODO: this will break complex data
    out._data = out.data.astype(float)

    # now obtain the derivative for each epoch separately
    for idx in range(asa.n_epochs):
        # if 1D:
        if asa.n_signals == 1:
            if (cum_lengths[idx + 1] - cum_lengths[idx]) < 2:
                # only single sample
                out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = 0
            else:
                out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = np.gradient(
                    asa._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]], axis=1
                )
        else:
            if (cum_lengths[idx + 1] - cum_lengths[idx]) < 2:
                # only single sample
                out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = 0
            else:
                out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = np.gradient(
                    asa._data[:, cum_lengths[idx] : cum_lengths[idx + 1]], axis=1
                )

    out._data = out._data * fs

    if norm:
        out._data = np.atleast_2d(np.linalg.norm(out._data, axis=0))

    if rectify:
        out._data = np.abs(out._data)

    if smooth:
        out = gaussian_filter(out, fs=fs, sigma=sigma, truncate=truncate)

    return out


@keyword_deprecation(replace_x_with_y={"bw": "truncate"})
def dxdt_AnalogSignalArray(
    asa, *, fs=None, smooth=False, rectify=True, sigma=None, truncate=None
):
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
    truncate : float, optional
        Bandwidth outside of which the filter value will be zero. Default is 4.0

    Returns
    -------
    out : AnalogSignalArray
        An AnalogSignalArray with derivative data (in units per second) is returned.
    """

    raise DeprecationWarning("use ddt_asa instead!")

    if fs is None:
        fs = asa.fs
    if fs is None:
        raise ValueError(
            "fs must either be specified, or must be contained in the AnalogSignalArray!"
        )
    if sigma is None:
        sigma = 0.05  # 50 ms default

    out = copy.deepcopy(asa)
    cum_lengths = np.insert(np.cumsum(asa.lengths), 0, 0)

    # ensure that datatype is float
    out._data = out.data.astype(float)

    if asa.n_signals == 2:
        out._data = out._data[[0], :]

    # now obtain the derivative for each epoch separately
    for idx in range(asa.n_epochs):
        # if 1D:
        if asa.n_signals == 1:
            if (cum_lengths[idx + 1] - cum_lengths[idx]) < 2:
                # only single sample
                out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = 0
            else:
                out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = np.gradient(
                    asa._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]], axis=1
                )
        elif asa.n_signals == 2:
            if (cum_lengths[idx + 1] - cum_lengths[idx]) < 2:
                # only single sample
                out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = 0
            else:
                out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = (
                    np.linalg.norm(
                        np.gradient(
                            asa._data[:, cum_lengths[idx] : cum_lengths[idx + 1]],
                            axis=1,
                        ),
                        axis=0,
                    )
                )
        else:
            raise TypeError("more than 2D not currently supported!")

    out._data = out._data * fs

    if rectify:
        out._data = np.abs(out._data)

    if smooth:
        out = gaussian_filter(out, fs=fs, sigma=sigma, truncate=truncate)

    return out


def get_threshold_crossing_epochs(asa, t1=None, t2=None, mode="above"):
    """Return epochs where a signal crosses a compound threshold specified by t1
    and t2.

    Parameters
    ----------
    asa : AnalogSignalArray
        AnalogSignalArray containing a single channel
    t1 : float, optional
        Primary threshold. Minimum signal value that has to be reached /
        exceeded during an event. Default is 3 standard deviations above signal
        mean.
    t2 : float, optional
        Secondary threshold. Signal value that defines the event boundaries.
        Default is signal mean.
    mode : string, optional
        Mode of operation. One of ['above', 'below']. If 'above', then return
        epochs where the signal exceeds the compound threshold, and if 'below',
        then return epochs where the signal falls below the compound threshold.
        Default is 'above'.

    Returns
    -------
    epochs : EpochArray
        EpochArray with all the epochs where the signal satisfied the criteria.
    """

    if asa.n_signals > 1:
        raise TypeError("multidimensional AnalogSignalArrays not supported!")
    x = asa.data.squeeze()

    if t1 is None:  # by default, threshold is 3 SDs above mean of x
        t1 = np.mean(x) + 3 * np.std(x)

    if t2 is None:  # by default, revert back to mean of x
        t2 = np.mean(x)

    # compute periods where signal exceeds compound threshold
    epoch_bounds, _, _ = get_events_boundaries(
        x=x, PrimaryThreshold=t1, SecondaryThreshold=t2, mode=mode
    )
    # convert bounds to time in seconds
    epoch_bounds = asa.time[epoch_bounds]
    if len(epoch_bounds) == 0:
        return type(asa._abscissa.support)(empty=True)
    # add 1/fs to stops for open interval
    epoch_bounds[:, 1] += 1 / asa.fs
    # create EpochArray with threshould exceeding bounds
    epochs = type(asa._abscissa.support)(epoch_bounds)
    return epochs


def get_run_epochs(speed, v1=10, v2=8):
    """Return epochs where animal is running at least as fast as
    specified by v1 and v2.

    Parameters
    ----------
    speed : AnalogSignalArray
        AnalogSignalArray containing single channel speed, in units/sec
    v1 : float, optional
        Minimum speed (in same units as speed) that has to be reached /
        exceeded during an event. Default is 10 [units/sec]
    v2 : float, optional
        Speed that defines the event boundaries. Default is 8 [units/sec]

    Returns
    -------
    run_epochs : EpochArray
        EpochArray with all the epochs where speed satisfied the criteria.
    """

    run_epochs = get_threshold_crossing_epochs(asa=speed, t1=v1, t2=v2, mode="above")

    return run_epochs


def get_inactive_epochs(speed, v1=5, v2=7):
    """Return epochs where animal is running no faster than specified by
    v1 and v2.

    Parameters
    ----------
    speed : AnalogSignalArray
        AnalogSignalArray containing single channel speed, in units/sec
    v1 : float, optional
        Minimum speed (in same units as speed) that has to be reached /
        exceeded during an event. Default is 10 [units/sec]
    v2 : float, optional
        Speed that defines the event boundaries. Default is 8 [units/sec]
    Returns
    -------
    inactive_epochs : EpochArray
        EpochArray with all the epochs where speed satisfied the criteria.
    """
    inactive_epochs = get_threshold_crossing_epochs(
        asa=speed, t1=v1, t2=v2, mode="below"
    )
    return inactive_epochs


def spiketrain_union(st1, st2):
    """Join two spiketrains together.

    WARNING! This function should be improved a lot!
    """
    assert st1.n_units == st2.n_units
    support = st1.support.join(st2.support)

    newdata = []
    for unit in range(st1.n_units):
        newdata.append(np.append(st1.time[unit], st2.time[unit]))

    fs = None
    if st1.fs == st2.fs:
        fs = st1.fs

    return core.SpikeTrainArray(newdata, support=support, fs=fs)


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

    TODO: this is a better version that should be incorporated:
    # Based on answer here: http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    def find_nearest(array,values):
        right_idxs = np.searchsorted(array, values, side="left")
        left_idxs = np.where(right_idxs > 0, right_idxs-1, right_idxs)
        right_idxs = np.where(right_idxs == len(array), len(array)-1, right_idxs)
        closest_idx = np.where(np.abs(values - array[right_idxs]) < np.abs(values - array[left_idxs]),
                            right_idxs, left_idxs)
        return closest_idx

    """
    return (np.abs(array - val)).argmin()


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


def collapse_time(obj, gap=0):
    """Collapse all epochs in a SpikeTrainArray and collapse them into a single, contiguous SpikeTrainArray"""

    # TODO: redo SpikeTrainArray so as to keep the epochs separate!, and to support gaps!

    # We'll have to ajust all the spikes per epoch... and we'll have to compute a new support. Also set a flag!

    # If it's a SpikeTrainArray, then we left-shift the spike times. If it's an AnalogSignalArray, then we
    # left-shift the time and tdata.

    # Also set a new attribute, with the boundaries in seconds.

    if isinstance(obj, core.RegularlySampledAnalogSignalArray):
        new_obj = type(obj)(empty=True)
        new_obj._data = obj._data

        durations = obj.support.durations
        starts = np.insert(np.cumsum(durations + gap), 0, 0)[:-1]
        stops = starts + durations
        newsupport = type(obj._abscissa.support)(np.vstack((starts, stops)).T)
        new_obj._support = newsupport

        new_time = obj.time.astype(float)  # fast copy
        time_idx = np.insert(np.cumsum(obj.lengths), 0, 0)

        new_offset = 0
        for epidx in range(obj.n_epochs):
            if epidx > 0:
                new_time[time_idx[epidx] : time_idx[epidx + 1]] = (
                    new_time[time_idx[epidx] : time_idx[epidx + 1]]
                    - obj.time[time_idx[epidx]]
                    + new_offset
                    + gap
                )
                new_offset += durations[epidx] + gap
            else:
                new_time[time_idx[epidx] : time_idx[epidx + 1]] = (
                    new_time[time_idx[epidx] : time_idx[epidx + 1]]
                    - obj.time[time_idx[epidx]]
                    + new_offset
                )
                new_offset += durations[epidx]
        new_obj._time = new_time

        new_obj._fs = obj._fs

    elif isinstance(obj, core.EventArray):
        if gap > 0:
            raise ValueError("gaps not supported for SpikeTrainArrays yet!")
        new_obj = type(obj)(empty=True)
        new_time = [[] for _ in range(obj.n_series)]
        duration = 0
        for st_ in obj:
            le = st_.support.start
            for unit_ in range(obj.n_series):
                new_time[unit_].extend(st_._data[unit_] - le + duration)
            duration += st_.support.duration
        new_time = np.asanyarray(
            [np.asanyarray(unittime) for unittime in new_time], dtype=object
        )
        new_obj._data = new_time
        new_obj.support = type(obj._abscissa.support)([0, duration])
        new_obj._series_ids = obj._series_ids
        new_obj._series_labels = obj._series_labels
        new_obj._series_tags = obj._series_tags
    elif isinstance(obj, core.BinnedEventArray):
        raise NotImplementedError(
            "BinnedEventArrays are not yet supported, but bst.data is essentially already collapsed!"
        )
    else:
        raise TypeError("unsupported type for collapse_time")

    return new_obj


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
    return np.transpose(
        [np.tile(xcenters, len(ycenters)), np.repeat(ycenters, len(xcenters))]
    )
