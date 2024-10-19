# -*- coding: utf-8 -*-
"""
RegularlySampledAnalogSignalArray
-----------------

Core object definition and implementation for RegularlySampledAnalogSignalArray.
"""

__all__ = [
    "RegularlySampledAnalogSignalArray",
    "AnalogSignalArray",
    "PositionArray",  # Trajectory?
    "IMUSensorArray",
    "MinimalExampleArray",
]

import logging
import numpy as np
import copy
import numbers

from sys import float_info
from functools import wraps
from scipy import interpolate
from scipy.stats import zscore
from sys import float_info
from collections import namedtuple

from .. import core
from .. import filtering
from .. import auxiliary
from .. import utils
from .. import version

from ..utils_.decorators import keyword_deprecation, keyword_equivalence


class IntervalSignalSlicer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """intervals, signals"""
        # by default, keep all signals
        signalslice = slice(None, None, None)
        if isinstance(*args, int):
            intervalslice = args[0]
        elif isinstance(*args, core.IntervalArray):
            intervalslice = args[0]
        else:
            try:
                slices = np.s_[args]
                slices = slices[0]
                if len(slices) > 2:
                    raise IndexError(
                        "only [intervals, signal] slicing is supported at this time!"
                    )
                elif len(slices) == 2:
                    intervalslice, signalslice = slices
                else:
                    intervalslice = slices[0]
            except TypeError:
                # only interval to slice:
                intervalslice = slices

        return intervalslice, signalslice


class DataSlicer(object):

    def __init__(self, parent):
        self._parent = parent

    def _data_generator(self, interval_indices, signalslice):
        for start, stop in interval_indices:
            try:
                yield self._parent._data[signalslice, start:stop]
            except StopIteration:
                return

    def __getitem__(self, idx):
        intervalslice, signalslice = self._parent._intervalsignalslicer[idx]

        interval_indices = self._parent._data_interval_indices()
        interval_indices = np.atleast_2d(interval_indices[intervalslice])

        if len(interval_indices) < 2:
            start, stop = interval_indices[0]
            return self._parent._data[signalslice, start:stop]
        else:
            return self._data_generator(interval_indices, signalslice)

    def plot_generator(self):
        interval_indices = self._parent._data_interval_indices()
        for start, stop in interval_indices:
            try:
                yield self._parent._data[:, start:stop]
            except StopIteration:
                return

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index

        if index > self._parent.n_intervals - 1:
            raise StopIteration

        interval_indices = self._parent._data_interval_indices()
        interval_indices = interval_indices[index]
        start, stop = interval_indices

        self._index += 1

        return self._parent._data[:, start:stop]


class AbscissaSlicer(object):

    def __init__(self, parent):
        self._parent = parent

    def _abscissa_vals_generator(self, interval_indices):
        for start, stop in interval_indices:
            try:
                yield self._parent._abscissa_vals[start:stop]
            except StopIteration:
                return

    def __getitem__(self, idx):
        intervalslice, signalslice = self._parent._intervalsignalslicer[idx]

        interval_indices = self._parent._data_interval_indices()
        interval_indices = np.atleast_2d(interval_indices[intervalslice])

        if len(interval_indices) < 2:
            start, stop = interval_indices[0]
            return self._parent._abscissa_vals[start:stop]
        else:
            return self._abscissa_vals_generator(interval_indices)

    def plot_generator(self):
        interval_indices = self._parent._data_interval_indices()
        for start, stop in interval_indices:
            try:
                yield self._parent._abscissa_vals[start:stop]
            except StopIteration:
                return

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index

        if index > self._parent.n_intervals - 1:
            raise StopIteration

        interval_indices = self._parent._data_interval_indices()
        interval_indices = interval_indices[index]
        start, stop = interval_indices

        self._index += 1

        return self._parent._abscissa_vals[start:stop]


def rsasa_init_wrapper(func):
    """Decorator that helps figure out abscissa_vals, fs, and sample numbers"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        if kwargs.get("empty", False):
            func(*args, **kwargs)
            return

        if len(args) > 2:
            raise TypeError(
                "__init__() takes 1 positional arguments but {} positional arguments (and {} keyword-only arguments) were given".format(
                    len(args) - 1, len(kwargs.items())
                )
            )

        data = kwargs.get("data", [])
        if len(data) == 0:
            data = args[1]

        if len(data) == 0:
            logging.warning(
                "No ordinate data! Returning empty RegularlySampledAnalogSignalArray."
            )
            func(*args, **kwargs)
            return

        # handle casting other nelpy objects to RegularlySampledAnalogSignalArrays:
        if isinstance(data, core.BinnedEventArray):
            abscissa_vals = data.bin_centers
            kwargs["abscissa_vals"] = abscissa_vals
            # support = data.support
            # kwargs['support'] = support
            abscissa = data._abscissa
            kwargs["abscissa"] = abscissa
            fs = 1 / data.ds
            kwargs["fs"] = fs
            if list(data.series_labels):
                labels = data.series_labels
            else:
                labels = data.series_ids
            kwargs["labels"] = labels
            data = data.data.astype(float)
        # elif isinstance(data, auxiliary.PositionArray):
        elif isinstance(data, RegularlySampledAnalogSignalArray):
            kwargs["data"] = data
            func(args[0], **kwargs)
            return

        # check if single AnalogSignal or multiple AnalogSignals in array
        # and standardize data to 2D
        if not isinstance(data, np.memmap):  # memmap is a special case
            if not np.any(np.iscomplex(data)):
                data = np.squeeze(data)
        try:
            if data.shape[0] == data.size:
                data = np.expand_dims(data, axis=0)
        except ValueError:
            raise TypeError("Unsupported data type!")

        re_estimate_fs = False
        no_fs = True
        fs = kwargs.get("fs", None)
        if fs is not None:
            no_fs = False
            try:
                if fs <= 0:
                    raise ValueError("fs must be positive")
            except TypeError:
                raise TypeError("fs must be a scalar!")
        else:
            fs = 1
            re_estimate_fs = True

        tdata = kwargs.get("tdata", None)
        if tdata is not None:
            logging.warning(
                "'tdata' has been deprecated! Use 'abscissa_vals' instead. 'tdata' will be interpreted as 'abscissa_vals' in seconds."
            )
            abscissa_vals = tdata
        else:
            abscissa_vals = kwargs.get("abscissa_vals", None)
        if abscissa_vals is None:
            abscissa_vals = np.linspace(0, data.shape[1] / fs, data.shape[1] + 1)
            abscissa_vals = abscissa_vals[:-1]
        else:
            if re_estimate_fs:
                logging.warning(
                    "fs was not specified, so we try to estimate it from the data..."
                )
                fs = 1.0 / np.median(np.diff(abscissa_vals))
                logging.warning("fs was estimated to be {} Hz".format(fs))
            else:
                if no_fs:
                    logging.warning(
                        "fs was not specified, so we will assume default of 1 Hz..."
                    )
                    fs = 1

        kwargs["fs"] = fs
        kwargs["data"] = data
        kwargs["abscissa_vals"] = np.squeeze(abscissa_vals)

        func(args[0], **kwargs)
        return

    return wrapper


########################################################################
# class RegularlySampledAnalogSignalArray
########################################################################
class RegularlySampledAnalogSignalArray:
    """Continuous analog signal(s) with regular sampling rates (irregular
    sampling rates can be corrected with operations on the support) and same
    support. NOTE: data that is not equal dimensionality will NOT work
    and error/warning messages may/may not be sent out. Assumes abscissa_vals
    are identical for all signals passed through and are therefore expected
    to be 1-dimensional.

    Parameters
    ----------
    data : np.ndarray, with shape (n_signals, n_samples).
        Data samples.
    abscissa_vals : np.ndarray, with shape (n_samples, ).
        The abscissa coordinate values. Currently we assume that (1) these values
        are timestamps, and (2) the timestamps are sampled regularly (we rely on
        these assumptions to generate intervals). Irregular sampling rates can be
        corrected with operations on the support.
    fs : float, optional
        The sampling rate. abscissa_vals are still expected to be in units of
        time and fs is expected to be in the corresponding sampling rate (e.g.
        abscissa_vals in seconds, fs in Hz).
        Default is 1 Hz.
    step : float, optional
        The sampling interval of the data, in seconds.
        Default is None.
        specifies step size of samples passed as tdata if fs is given,
        default is None. If not passed it is inferred by the minimum
        difference in between samples of tdata passed in (based on if FS
        is passed). e.g. decimated data would have sample numbers every
        ten samples so step=10
    merge_sample_gap : float, optional
        Optional merging of gaps between support intervals. If intervals are within
        a certain amount of time, gap, they will be merged as one interval. Example
        use case is when there is a dropped sample
    support : nelpy.IntervalArray, optional
        Where the data are defined. Default is [0, last abscissa value] inclusive.
    in_core : bool, optional
        Whether the abscissa values should be treated as residing in core memory.
        During RSASA construction, np.diff() is called, so for large data, passing
        in in_core=True might help. In that case, a slower but much smaller memory
        footprint function is used.
    labels : np.array, dtype=np.str
        Labels for each of the signals. If fewer labels than signals are passed in,
        labels are padded with None's to match the number of signals. If more labels
        than signals are passed in, labels are truncated to match the number of
        signals.
        Default is None.
    empty : bool, optional
        Return an empty RegularlySampledAnalogSignalArray if true else false.
        Default is false.
    abscissa : optional
        The object handling the abscissa values. It is recommended to leave
        this parameter alone and let nelpy take care of this.
        Default is a nelpy.core.Abscissa object.
    ordinate : optional
        The object handling the ordinate values. It is recommended to leave
        this parameter alone and let nelpy take care of this.
        Default is a nelpy.core.Ordinate object.

    Attributes
    ----------
    data : np.ndarray, with shape (n_signals, n_samples)
        The underlying data.
    abscissa_vals : np.ndarray, with shape (n_samples, )
        The values of the abscissa coordinate.
    is1d : bool
        Whether there is only 1 signal in the RSASA
    iswrapped : bool
        Whether the RSASA's data is wrapping.
    base_unit : string
        Base unit of the abscissa.
    signals : list
        A list of RegularlySampledAnalogSignalArrays, each RSASA containing
        a single signal (channel).
        WARNING: this method creates a copy of each signal, so is not
        particularly efficient at this time.
    isreal : bool
        Whether ALL of the values in the RSASA's data are real.
    iscomplex : bool
        Whether ANY values in the data are complex.
    abs : nelpy.RegularlySampledAnalogSignalArray
        A copy of the RSASA, whose data is the absolute value of the original
        original RSASA's (potentially complex) data.
    phase : nelpy.RegularlySampledAnalogSignalArray
        A copy of the RSASA, whose data is just the phase angle (in radians) of
        the original RSASA's data.
    real : nelpy.RegularlySampledAnalogSignalArray
        A copy of the RSASA, whose data is just the real part of the original
        RSASA's data.
    imag : nelpy.RegularlySampledAnalogSignalArray
        A copy of the RSASA, whose data is just the imaginary part of the
        original RSASA's data.
    lengths : list
        The number of samples in each interval.
    labels : list
        The labels corresponding to each signal.
    n_signals : int
        The number of signals in the RSASA.
    support : nelpy.IntervalArray
        The support of the RSASA.
    domain : nelpy.IntervalArray
        The domain of the RSASA.
    range : nelpy.IntervalArray
        The range of the RSASA's data.
    step : float
        The sampling interval of the RSASA. Currently the units are
        in seconds.
    fs : float
        The sampling frequency of the RSASA. Currently the units are
        in Hz.
    isempty : bool
        Whether the underlying data has zero length, i.e. 0 samples
    n_bytes : int
        Approximate number of bytes taken up by the RSASA.
    n_intervals : int
        The number of underlying intervals in the RSASA.
    n_samples : int
        The number of abscissa values in the RSASA.
    """

    __aliases__ = {}

    __attributes__ = [
        "_data",
        "_abscissa_vals",
        "_fs",
        "_support",
        "_interp",
        "_step",
        "_labels",
    ]

    @rsasa_init_wrapper
    def __init__(
        self,
        data=[],
        *,
        abscissa_vals=None,
        fs=None,
        step=None,
        merge_sample_gap=0,
        support=None,
        in_core=True,
        labels=None,
        empty=False,
        abscissa=None,
        ordinate=None
    ):

        self._intervalsignalslicer = IntervalSignalSlicer(self)
        self._intervaldata = DataSlicer(self)
        self._intervaltime = AbscissaSlicer(self)

        self.type_name = self.__class__.__name__
        if abscissa is None:
            abscissa = core.Abscissa()  # TODO: integrate into constructor?
        if ordinate is None:
            ordinate = core.Ordinate()  # TODO: integrate into constructor?

        self._abscissa = abscissa
        self._ordinate = ordinate

        # TODO: #FIXME abscissa and ordinate domain, range, and supports should be integrated and/or coerced with support

        self.__version__ = version.__version__

        # cast derivatives of RegularlySampledAnalogSignalArray back into RegularlySampledAnalogSignalArray:
        # if isinstance(data, auxiliary.PositionArray):
        if isinstance(data, RegularlySampledAnalogSignalArray):
            self.__dict__ = copy.deepcopy(data.__dict__)
            # if self._has_changed:
            # self.__renew__()
            self.__renew__()
            return

        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._abscissa.support = type(self._abscissa.support)(empty=True)
            self._data = np.array([])
            self._abscissa_vals = np.array([])
            self.__bake__()
            return

        self._step = step
        self._fs = fs

        # Note; if we have an empty array of data with no dimension,
        # then calling len(data) will return a TypeError
        try:
            # if no data are given return empty RegularlySampledAnalogSignalArray
            if data.size == 0:
                self.__init__(empty=True)
                return
        except TypeError:
            logging.warning(
                "unsupported type; creating empty RegularlySampledAnalogSignalArray"
            )
            self.__init__(empty=True)
            return

        # Note: if both abscissa_vals and data are given and dimensionality does not
        # match, then TypeError!

        abscissa_vals = np.squeeze(abscissa_vals).astype(float)
        if abscissa_vals.shape[0] != data.shape[1]:
            # self.__init__([],empty=True)
            raise TypeError(
                "abscissa_vals and data size mismatch! Note: data "
                "is expected to have rows containing signals"
            )
        # data is not sorted and user wants it to be
        # TODO: use faster is_sort from jagular
        if not utils.is_sorted(abscissa_vals):
            logging.warning(
                "Data is _not_ sorted! Data will be sorted " "automatically."
            )
            ind = np.argsort(abscissa_vals)
            abscissa_vals = abscissa_vals[ind]
            data = np.take(a=data, indices=ind, axis=-1)

        self._data = data
        self._abscissa_vals = abscissa_vals

        # handle labels
        if labels is not None:
            labels = np.asarray(labels, dtype=str)
            # label size doesn't match
            if labels.shape[0] > data.shape[0]:
                logging.warning(
                    "More labels than data! Labels are truncated to " "size of data"
                )
                labels = labels[0 : data.shape[0]]
            elif labels.shape[0] < data.shape[0]:
                logging.warning(
                    "Fewer labels than abscissa_vals! Labels are filled with "
                    "None to match data shape"
                )
                for i in range(labels.shape[0], data.shape[0]):
                    labels.append(None)
        self._labels = labels

        # Alright, let's handle all the possible parameter cases!
        if support is not None:
            self._restrict_to_interval_array_fast(intervalarray=support)
        else:
            logging.warning(
                "creating support from abscissa_vals and " "sampling rate, fs!"
            )
            self._abscissa.support = type(self._abscissa.support)(
                utils.get_contiguous_segments(
                    self._abscissa_vals, step=self._step, fs=fs, in_core=in_core
                )
            )
            if merge_sample_gap > 0:
                self._abscissa.support = self._abscissa.support.merge(
                    gap=merge_sample_gap
                )

        if np.abs((self.fs - self._estimate_fs()) / self.fs) > 0.01:
            logging.warning("estimated fs and provided fs differ by more than 1%")

    def __bake__(self):
        """Fix object as-is, and bake a new hash.

        For example, if a label has changed, or if an interp has been attached,
        then the object's hash will change, and it needs to be baked
        again for efficiency / consistency.
        """
        self._stored_hash_ = self.__hash__()

    # def _has_changed_data(self):
    #     """Compute hash on abscissa_vals and data and compare to cached hash."""
    #     return self.data.__hash__ elf._data_hash_

    def _has_changed(self):
        """Compute hash on current object, and compare to previously stored hash"""
        return self.__hash__() == self._stored_hash_

    def __renew__(self):
        """Re-attach data slicers."""
        self._intervalsignalslicer = IntervalSignalSlicer(self)
        self._intervaldata = DataSlicer(self)
        self._intervaltime = AbscissaSlicer(self)
        self._interp = None
        self.__bake__()

    def __call__(self, x):
        """RegularlySampledAnalogSignalArray callable method. Returns
        interpolated data at requested points. Note that points falling
        outside the support will not be interpolated.

        Parameters
        ----------
        x : np.ndarray, list, or tuple, with length n_requested_samples
            Points at which to interpolate the RSASA's data

        Returns
        -------
        A np.ndarray with shape (n_signals, n_samples). If all the requested
        points lie in the support, then n_samples = n_requested_samples.
        Otherwise n_samples < n_requested_samples.
        """

        return self.asarray(at=x).yvals

    def center(self, inplace=False):
        """Center data (zero mean)."""
        if inplace:
            out = self
        else:
            out = self.copy()
        out._data = (out._data.T - out.mean()).T
        return out

    def normalize(self, inplace=False):
        """Normalize data (unit standard deviation)."""
        if inplace:
            out = self
        else:
            out = self.copy()
        std = np.atleast_1d(out.std())
        std[std == 0] = 1
        out._data = (out._data.T / std).T
        return out

    def standardize(self, inplace=False):
        """Standardize data (zero mean and unit std deviation)."""
        if inplace:
            out = self
        else:
            out = self.copy()
        out._data = (out._data.T - out.mean()).T
        std = np.atleast_1d(out.std())
        std[std == 0] = 1
        out._data = (out._data.T / std).T
        return out

    @property
    def is_1d(self):
        try:
            return self.n_signals == 1
        except IndexError:
            return False

    @property
    def is_wrapped(self):
        if np.any(self.max() > self._ordinate.range.stop) | np.any(
            self.min() < self._ordinate.range.min
        ):
            self._ordinate._is_wrapped = False
        else:
            self._ordinate._is_wrapped = True

        # if self._ordinate._is_wrapped is None:
        #     if np.any(self.max() > self._ordinate.range.stop) | np.any(self.min() < self._ordinate.range.min):
        #         self._ordinate._is_wrapped = False
        #     else:
        #         self._ordinate._is_wrapped = True
        return self._ordinate._is_wrapped

    def _wrap(self, arr, vmin, vmax):
        """Wrap array within finite range."""
        if np.isinf(vmax - vmin):
            raise ValueError("range has to be finite!")
        return ((arr - vmin) % (vmax - vmin)) + vmin

    def wrap(self, inplace=False):
        """Wrap oridnate within finite range."""
        if inplace:
            out = self
        else:
            out = self.copy()

        out.data = np.atleast_2d(
            out._wrap(out.data, out._ordinate.range.min, out._ordinate.range.max)
        )
        # out._is_wrapped = True
        return out

    def _unwrap(self, arr, vmin, vmax):
        """Unwrap 2D array (with one signal per row) by minimizing total displacement."""
        d = vmax - vmin
        dh = d / 2

        lin = copy.deepcopy(arr) - vmin
        n_signals, n_samples = arr.shape
        for ii in range(1, n_samples):
            h1 = lin[:, ii] - lin[:, ii - 1] >= dh
            lin[h1, ii:] = lin[h1, ii:] - d
            h2 = lin[:, ii] - lin[:, ii - 1] < -dh
            lin[h2, ii:] = lin[h2, ii:] + d
        return np.atleast_2d(lin + vmin)

    def unwrap(self, inplace=False):
        """Unwrap ordinate by minimizing total displacement."""
        if inplace:
            out = self
        else:
            out = self.copy()

        out.data = np.atleast_2d(
            out._unwrap(out._data, out._ordinate.range.min, out._ordinate.range.max)
        )
        # out._is_wrapped = False
        return out

    def _crossvals(self):
        """Return all abscissa values where the orinate crosses.

        Note that this can return multiple values close in succession
        if the signal oscillates around the maximum or minimum range.
        """
        raise NotImplementedError

    @property
    def base_unit(self):
        """Base unit of the abscissa."""
        return self._abscissa.base_unit

    def _data_interval_indices(self):
        """Docstring goes here.
        We use this to get the indices of samples / abscissa_vals within intervals
        """
        tmp = np.insert(np.cumsum(self.lengths), 0, 0)
        indices = np.vstack((tmp[:-1], tmp[1:])).T
        return indices

    def ddt(self, rectify=False):
        """Returns the derivative of each signal in the RegularlySampledAnalogSignalArray.

        asa.data = f(t)
        asa.ddt = d/dt (asa.data)

        Parameters
        ----------
        rectify : boolean, optional
            If True, the absolute value of the derivative will be returned.
            Default is False.

        Returns
        -------
        ddt : RegularlySampledAnalogSignalArray
            Time derivative of each signal in the RegularlySampledAnalogSignalArray.

        Note
        ----
        Second order central differences are used here, and it is assumed that
        the signals are sampled uniformly. If the signals are not uniformly
        sampled, it is recommended to resample the signal before computing the
        derivative.
        """
        ddt = utils.ddt_asa(self, rectify=rectify)
        return ddt

    @property
    def signals(self):
        """Returns a list of RegularlySampledAnalogSignalArrays, each array containing
        a single signal (channel).

        WARNING: this method creates a copy of each signal, so is not
        particularly efficient at this time.

        Example
        =======
        >>> for channel in lfp.signals:
            print(channel)
        """
        signals = []
        for ii in range(self.n_signals):
            signals.append(self[:, ii])
        return signals
        # return np.asanyarray(signals).squeeze()

    @property
    def isreal(self):
        """Returns True if entire signal is real."""
        return np.all(np.isreal(self.data))
        # return np.isrealobj(self._data)

    @property
    def iscomplex(self):
        """Returns True if any part of the signal is complex."""
        return np.any(np.iscomplex(self.data))
        # return np.iscomplexobj(self._data)

    @property
    def abs(self):
        """RegularlySampledAnalogSignalArray with absolute value of (potentially complex) data."""
        out = self.copy()
        out._data = np.abs(self.data)
        return out

    @property
    def angle(self):
        """RegularlySampledAnalogSignalArray with only phase angle (in radians) of data."""
        out = self.copy()
        out._data = np.angle(self.data)
        return out

    @property
    def imag(self):
        """RegularlySampledAnalogSignalArray with only imaginary part of data."""
        out = self.copy()
        out._data = self.data.imag
        return out

    @property
    def real(self):
        """RegularlySampledAnalogSignalArray with only real part of data."""
        out = self.copy()
        out._data = self.data.real
        return out

    def __mul__(self, other):
        """overloaded * operator."""
        if isinstance(other, numbers.Number):
            newasa = self.copy()
            newasa._data = self.data * other
            return newasa
        elif isinstance(other, np.ndarray):
            newasa = self.copy()
            newasa._data = (self.data.T * other).T
            return newasa
        else:
            raise TypeError(
                "unsupported operand type(s) for *: 'RegularlySampledAnalogSignalArray' and '{}'".format(
                    str(type(other))
                )
            )

    def __add__(self, other):
        """overloaded + operator."""
        if isinstance(other, numbers.Number):
            newasa = self.copy()
            newasa._data = self.data + other
            return newasa
        elif isinstance(other, np.ndarray):
            newasa = self.copy()
            newasa._data = (self.data.T + other).T
            return newasa
        else:
            raise TypeError(
                "unsupported operand type(s) for +: 'RegularlySampledAnalogSignalArray' and '{}'".format(
                    str(type(other))
                )
            )

    def __sub__(self, other):
        """overloaded - operator."""
        if isinstance(other, numbers.Number):
            newasa = self.copy()
            newasa._data = self.data - other
            return newasa
        elif isinstance(other, np.ndarray):
            newasa = self.copy()
            newasa._data = (self.data.T - other).T
            return newasa
        else:
            raise TypeError(
                "unsupported operand type(s) for -: 'RegularlySampledAnalogSignalArray' and '{}'".format(
                    str(type(other))
                )
            )

    def zscore(self):
        """Returns an object where each signal has been normalized using z scores."""
        out = self.copy()
        out._data = zscore(out._data, axis=1)
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """overloaded / operator."""
        if isinstance(other, numbers.Number):
            newasa = self.copy()
            newasa._data = self.data / other
            return newasa
        elif isinstance(other, np.ndarray):
            newasa = self.copy()
            newasa._data = (self.data.T / other).T
            return newasa
        else:
            raise TypeError(
                "unsupported operand type(s) for /: 'RegularlySampledAnalogSignalArray' and '{}'".format(
                    str(type(other))
                )
            )

    def __lshift__(self, val):
        """shift abscissa and support to left (<<)"""
        if isinstance(val, numbers.Number):
            new = self.copy()
            new._abscissa_vals -= val
            new._abscissa.support = new._abscissa.support << val
            return new
        else:
            raise TypeError(
                "unsupported operand type(s) for <<: {} and {}".format(
                    str(type(self)), str(type(val))
                )
            )

    def __rshift__(self, val):
        """shift abscissa and support to right (>>)"""
        if isinstance(val, numbers.Number):
            new = self.copy()
            new._abscissa_vals += val
            new._abscissa.support = new._abscissa.support >> val
            return new
        else:
            raise TypeError(
                "unsupported operand type(s) for >>: {} and {}".format(
                    str(type(self)), str(type(val))
                )
            )

    def __len__(self):
        return self.n_intervals

    def _drop_empty_intervals(self):
        """Drops empty intervals from support. In-place."""
        keep_interval_ids = np.argwhere(self.lengths).squeeze().tolist()
        self._abscissa.support = self._abscissa.support[keep_interval_ids]
        return self

    def _estimate_fs(self, abscissa_vals=None):
        """Estimate the sampling rate of the data."""
        if abscissa_vals is None:
            abscissa_vals = self._abscissa_vals
        return 1.0 / np.median(np.diff(abscissa_vals))

    def downsample(self, *, fs_out, aafilter=True, inplace=False, **kwargs):
        """Downsamples the RegularlySampledAnalogSignalArray

        Parameters
        ----------
        fs_out : float, optional
            Desired output sampling rate in Hz
        aafilter : boolean, optional
            Whether to apply an anti-aliasing filter before performing the actual
            downsampling. Default is True
        inplace : boolean, optional
            If True, the output ASA will replace the input ASA. Default is False
        kwargs :
            Other keyword arguments are passed to sosfiltfilt() in the `filtering`
            module

        Returns
        -------
        out : RegularlySampledAnalogSignalArray
            The downsampled RegularlySampledAnalogSignalArray
        """

        if not fs_out < self._fs:
            raise ValueError("fs_out must be less than current sampling rate!")

        if aafilter:
            fh = fs_out / 2.0
            out = filtering.sosfiltfilt(self, fl=None, fh=fh, inplace=inplace, **kwargs)

        downsampled = out.simplify(ds=1 / fs_out)
        out._data = downsampled._data
        out._abscissa_vals = downsampled._abscissa_vals
        out._fs = fs_out

        out.__renew__()
        return out

    def add_signal(self, signal, label=None):
        """Docstring goes here.
        Basically we add a signal, and we add a label. THIS HAPPENS IN PLACE?
        """
        # TODO: add functionality to check that supports are the same, etc.
        if isinstance(signal, RegularlySampledAnalogSignalArray):
            signal = signal.data

        signal = np.squeeze(signal)
        if signal.ndim > 1:
            raise TypeError("Can only add one signal at a time!")
        if self.data.ndim == 1:
            self._data = np.vstack(
                [np.array(self.data, ndmin=2), np.array(signal, ndmin=2)]
            )
        else:
            self._data = np.vstack([self.data, np.array(signal, ndmin=2)])
        if label == None:
            logging.warning("None label appended")
        self._labels = np.append(self._labels, label)
        return self

    def _restrict_to_interval_array_fast(self, *, intervalarray=None, update=True):
        """Restrict self._abscissa_vals and self._data to an IntervalArray. If no
        IntervalArray is specified, self._abscissa.support is used.

        Parameters
        ----------
        intervalarray : IntervalArray, optional
                IntervalArray on which to restrict AnalogSignal. Default is
                self._abscissa.support
        update : bool, optional
                Overwrite self._abscissa.support with intervalarray if True (default).
        """
        if intervalarray is None:
            intervalarray = self._abscissa.support
            update = False  # support did not change; no need to update

        try:
            if intervalarray.isempty:
                logging.warning("Support specified is empty")
                # self.__init__([],empty=True)
                exclude = ["_support", "_data", "_fs", "_step"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                logging.disable(logging.CRITICAL)
                for attr in attrs:
                    exec("self." + attr + " = None")
                logging.disable(0)
                self._data = np.zeros([0, self.data.shape[0]])
                self._data[:] = np.nan
                self._abscissa.support = intervalarray
                return
        except AttributeError:
            raise AttributeError("IntervalArray expected")

        indices = []
        for interval in intervalarray.merge().data:
            a_start = interval[0]
            a_stop = interval[1]
            frm, to = np.searchsorted(self._abscissa_vals, (a_start, a_stop + 1e-10))
            indices.append((frm, to))
        indices = np.array(indices, ndmin=2)
        if np.diff(indices).sum() < len(self._abscissa_vals):
            logging.warning("ignoring signal outside of support")
        # check if only one interval and interval is already bounds of data
        # if so, we don't need to do anything
        if len(indices) == 1:
            if indices[0, 0] == 0 and indices[0, 1] == len(self._abscissa_vals):
                if update:
                    self._abscissa.support = intervalarray
                    return
        try:
            data_list = []
            for start, stop in indices:
                data_list.append(self._data[:, start:stop])
            self._data = np.hstack(data_list)
        except IndexError:
            self._data = np.zeros([0, self.data.shape[0]])
            self._data[:] = np.nan
        time_list = []
        for start, stop in indices:
            time_list.extend(self._abscissa_vals[start:stop])
        self._abscissa_vals = np.array(time_list)
        if update:
            self._abscissa.support = intervalarray

    def _restrict_to_interval_array(self, *, intervalarray=None, update=True):
        """Restrict self._abscissa_vals and self._data to an IntervalArray. If no
        IntervalArray is specified, self._abscissa.support is used.

        This function is quite slow, as it checks each sample for inclusion.
        It does this in a vectorized form, which is fast for small or moderately
        sized objects, but the memory penalty can be large, and it becomes very
        slow for large objects. Consequently, _restrict_to_interval_array_fast
        should be used when possible.

        Parameters
        ----------
        intervalarray : IntervalArray, optional
                IntervalArray on which to restrict AnalogSignal. Default is
                self._abscissa.support
        update : bool, optional
                Overwrite self._abscissa.support with intervalarray if True (default).
        """
        if intervalarray is None:
            intervalarray = self._abscissa.support
            update = False  # support did not change; no need to update

        try:
            if intervalarray.isempty:
                logging.warning("Support specified is empty")
                # self.__init__([],empty=True)
                exclude = ["_support", "_data", "_fs", "_step"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                logging.disable(logging.CRITICAL)
                for attr in attrs:
                    exec("self." + attr + " = None")
                logging.disable(0)
                self._data = np.zeros([0, self.data.shape[0]])
                self._data[:] = np.nan
                self._abscissa.support = intervalarray
                return
        except AttributeError:
            raise AttributeError("IntervalArray expected")

        indices = []
        for interval in intervalarray.merge().data:
            a_start = interval[0]
            a_stop = interval[1]
            indices.append(
                (self._abscissa_vals >= a_start) & (self._abscissa_vals < a_stop)
            )
        indices = np.any(np.column_stack(indices), axis=1)
        if np.count_nonzero(indices) < len(self._abscissa_vals):
            logging.warning("ignoring signal outside of support")
        try:
            self._data = self.data[:, indices]
        except IndexError:
            self._data = np.zeros([0, self.data.shape[0]])
            self._data[:] = np.nan
        self._abscissa_vals = self._abscissa_vals[indices]
        if update:
            self._abscissa.support = intervalarray

    @keyword_deprecation(replace_x_with_y={"bw": "truncate"})
    def smooth(
        self,
        *,
        fs=None,
        sigma=None,
        truncate=None,
        inplace=False,
        mode=None,
        cval=None,
        within_intervals=False
    ):
        """Smooths the regularly sampled RegularlySampledAnalogSignalArray with a Gaussian kernel.

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

        kwargs = {
            "inplace": inplace,
            "fs": fs,
            "sigma": sigma,
            "truncate": truncate,
            "mode": mode,
            "cval": cval,
            "within_intervals": within_intervals,
        }

        if inplace:
            out = self
        else:
            out = self.copy()

        if self._ordinate.is_wrapping:
            ord_is_wrapped = self.is_wrapped

            if ord_is_wrapped:
                out = out.unwrap()

        # case 1: abs.wrapping=False, ord.linking=False, ord.wrapping=False
        if (
            not self._abscissa.is_wrapping
            and not self._ordinate.is_linking
            and not self._ordinate.is_wrapping
        ):
            pass

        # case 2: abs.wrapping=False, ord.linking=False, ord.wrapping=True
        elif (
            not self._abscissa.is_wrapping
            and not self._ordinate.is_linking
            and self._ordinate.is_wrapping
        ):
            pass

        # case 3: abs.wrapping=False, ord.linking=True, ord.wrapping=False
        elif (
            not self._abscissa.is_wrapping
            and self._ordinate.is_linking
            and not self._ordinate.is_wrapping
        ):
            raise NotImplementedError

        # case 4: abs.wrapping=False, ord.linking=True, ord.wrapping=True
        elif (
            not self._abscissa.is_wrapping
            and self._ordinate.is_linking
            and self._ordinate.is_wrapping
        ):
            raise NotImplementedError

        # case 5: abs.wrapping=True, ord.linking=False, ord.wrapping=False
        elif (
            self._abscissa.is_wrapping
            and not self._ordinate.is_linking
            and not self._ordinate.is_wrapping
        ):
            if mode is None:
                kwargs["mode"] = "wrap"

        # case 6: abs.wrapping=True, ord.linking=False, ord.wrapping=True
        elif (
            self._abscissa.is_wrapping
            and not self._ordinate.is_linking
            and self._ordinate.is_wrapping
        ):
            # (1) unwrap ordinate (abscissa wrap=False)
            # (2) smooth unwrapped ordinate (absissa wrap=False)
            # (3) repeat unwrapped signal based on conditions from (2):
            # if smoothed wrapped ordinate samples
            # HH ==> SSS (this must be done on a per-signal basis!!!) H = high; L = low; S = same
            # LL ==> SSS (the vertical offset must be such that neighbors have smallest displacement)
            # LH ==> LSH
            # HL ==> HSL
            # (4) smooth expanded and unwrapped ordinate (abscissa wrap=False)
            # (5) cut out orignal signal

            # (1)
            kwargs["mode"] = "reflect"
            L = out._ordinate.range.max - out._ordinate.range.min
            D = out.domain.length

            tmp = utils.gaussian_filter(out.unwrap(), **kwargs)
            # (2) (3)
            n_reps = int(np.ceil((sigma * truncate) / float(D)))

            smooth_data = []
            for ss, signal in enumerate(tmp.signals):

                # signal = signal.wrap()
                offset = (
                    float((signal._data[:, -1] - signal._data[:, 0]) // (L / 2)) * L
                )
                # print(offset)
                # left_high = signal._data[:,0] >= out._ordinate.range.min + L/2
                # right_high = signal._data[:,-1] >= out._ordinate.range.min + L/2
                # signal = signal.unwrap()

                expanded = signal.copy()
                for nn in range(n_reps):
                    expanded = expanded.join((signal << D * (nn + 1)) - offset).join(
                        (signal >> D * (nn + 1)) + offset
                    )
                    # print(expanded)
                    # if left_high == right_high:
                    #     print('extending flat! signal {}'.format(ss))
                    #     expanded = expanded.join(signal << D*(nn+1)).join(signal >> D*(nn+1))
                    # elif left_high < right_high:
                    #     print('extending LSH! signal {}'.format(ss))
                    #     # LSH
                    #     expanded = expanded.join((signal << D*(nn+1))-L).join((signal >> D*(nn+1))+L)
                    # else:
                    #     # HSL
                    #     print('extending HSL! signal {}'.format(ss))
                    #     expanded = expanded.join((signal << D*(nn+1))+L).join((signal >> D*(nn+1))-L)
                # (4)
                smooth_signal = utils.gaussian_filter(expanded, **kwargs)
                smooth_data.append(
                    smooth_signal._data[
                        :, n_reps * tmp.n_samples : (n_reps + 1) * (tmp.n_samples)
                    ].squeeze()
                )
            # (5)
            out._data = np.array(smooth_data)
            out.__renew__()

            if self._ordinate.is_wrapping:
                if ord_is_wrapped:
                    out = out.wrap()

            return out

        # case 7: abs.wrapping=True, ord.linking=True, ord.wrapping=False
        elif (
            self._abscissa.is_wrapping
            and self._ordinate.is_linking
            and not self._ordinate.is_wrapping
        ):
            raise NotImplementedError

        # case 8: abs.wrapping=True, ord.linking=True, ord.wrapping=True
        elif (
            self._abscissa.is_wrapping
            and self._ordinate.is_linking
            and self._ordinate.is_wrapping
        ):
            raise NotImplementedError

        out = utils.gaussian_filter(out, **kwargs)
        out.__renew__()

        if self._ordinate.is_wrapping:
            if ord_is_wrapped:
                out = out.wrap()

        return out

    @property
    def lengths(self):
        """(list) The number of samples in each interval."""
        indices = []
        for interval in self.support.data:
            a_start = interval[0]
            a_stop = interval[1]
            frm, to = np.searchsorted(self._abscissa_vals, (a_start, a_stop))
            indices.append((frm, to))
        indices = np.array(indices, ndmin=2)
        lengths = np.atleast_1d(np.diff(indices).squeeze())
        return lengths

    @property
    def labels(self):
        """(list) The labels corresponding to each signal."""
        # TODO: make this faster and better!
        return self._labels

    @property
    def n_signals(self):
        """(int) The number of signals."""
        try:
            return utils.PrettyInt(self.data.shape[0])
        except AttributeError:
            return 0

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty " + self.type_name + address_str + ">"
        if self.n_intervals > 1:
            epstr = " ({} segments)".format(self.n_intervals)
        else:
            epstr = ""
        try:
            if self.n_signals > 0:
                nstr = " %s signals%s" % (self.n_signals, epstr)
        except IndexError:
            nstr = " 1 signal%s" % epstr
        dstr = " for a total of {}".format(
            self._abscissa.formatter(self.support.length)
        )
        return "<%s%s:%s>%s" % (self.type_name, address_str, nstr, dstr)

    @keyword_equivalence(this_or_that={"n_intervals": "n_epochs"})
    def partition(self, ds=None, n_intervals=None):
        """Returns an RegularlySampledAnalogSignalArray whose support has been
        partitioned.

        # Irrespective of whether 'ds' or 'n_intervals' are used, the exact
        # underlying support is propagated, and the first and last points
        # of the supports are always included, even if this would cause
        # n_samples or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Maximum duration (in seconds), for each interval.
        n_samples : int, optional
            Number of intervals. If ds is None and n_intervals is None, then
            default is to use n_intervals = 100

        Returns
        -------
        out : RegularlySampledAnalogSignalArray
            RegularlySampledAnalogSignalArray that has been partitioned.
        """

        out = self.copy()
        out._abscissa.support = out.support.partition(ds=ds, n_intervals=n_intervals)
        return out

    # @property
    # def ydata(self):
    #     """(np.array N-Dimensional) data with shape (n_signals, n_samples)."""
    #     # LEGACY
    #     return self.data

    @property
    def data(self):
        """(np.array N-Dimensional) data with shape (n_signals, n_samples)."""
        return self._data

    @data.setter
    def data(self, val):
        """(np.array N-Dimensional) data with shape (n_signals, n_samples)."""
        self._data = val
        # print('data was modified, so clearing interp, etc.')
        self.__renew__()

    @property
    def support(self):
        """(nelpy.IntervalArray) The support of the underlying RegularlySampledAnalogSignalArray."""
        return self._abscissa.support

    @support.setter
    def support(self, val):
        """(nelpy.IntervalArray) The support of the underlying RegularlySampledAnalogSignalArray."""
        # modify support
        if isinstance(val, type(self._abscissa.support)):
            self._abscissa.support = val
        elif isinstance(val, (tuple, list)):
            prev_domain = self._abscissa.domain
            self._abscissa.support = type(self._abscissa.support)([val[0], val[1]])
            self._abscissa.domain = prev_domain
        else:
            raise TypeError(
                "support must be of type {}".format(str(type(self._abscissa.support)))
            )
        # restrict data to new support
        self._restrict_to_interval_array_fast(intervalarray=self._abscissa.support)

    @property
    def domain(self):
        """(nelpy.IntervalArray) The domain of the underlying RegularlySampledAnalogSignalArray."""
        return self._abscissa.domain

    @domain.setter
    def domain(self, val):
        """(nelpy.IntervalArray) The domain of the underlying RegularlySampledAnalogSignalArray."""
        # modify domain
        if isinstance(val, type(self._abscissa.support)):
            self._abscissa.domain = val
        elif isinstance(val, (tuple, list)):
            self._abscissa.domain = type(self._abscissa.support)([val[0], val[1]])
        else:
            raise TypeError(
                "support must be of type {}".format(str(type(self._abscissa.support)))
            )
        # restrict data to new support
        self._restrict_to_interval_array_fast(intervalarray=self._abscissa.support)

    @property
    def range(self):
        """(nelpy.IntervalArray) The range of the underlying RegularlySampledAnalogSignalArray."""
        return self._ordinate.range

    @range.setter
    def range(self, val):
        """(nelpy.IntervalArray) The range of the underlying RegularlySampledAnalogSignalArray."""
        # modify range
        self._ordinate.range = val

    @property
    def step(self):
        """steps per sample
        Example 1: sample_numbers = np.array([1,2,3,4,5,6]) #aka time
        Steps per sample in the above case would be 1

        Example 2: sample_numbers = np.array([1,3,5,7,9]) #aka time
        Steps per sample in Example 2 would be 2
        """
        return self._step

    @property
    def abscissa_vals(self):
        """(np.array 1D) Time in seconds."""
        return self._abscissa_vals

    @abscissa_vals.setter
    def abscissa_vals(self, vals):
        """(np.array 1D) Time in seconds."""
        self._abscissa_vals = vals

    @property
    def fs(self):
        """(float) Sampling frequency."""
        if self._fs is None:
            logging.warning("No sampling frequency has been specified!")
        return self._fs

    @property
    def isempty(self):
        """(bool) checks length of data input"""
        try:
            return self.data.shape[1] == 0
        except IndexError:  # IndexError should happen if _data = []
            return True

    @property
    def n_bytes(self):
        """Approximate number of bytes taken up by object."""
        return utils.PrettyBytes(self.data.nbytes + self._abscissa_vals.nbytes)

    @property
    def n_intervals(self):
        """(int) number of intervals in RegularlySampledAnalogSignalArray"""
        return self._abscissa.support.n_intervals

    @property
    def n_samples(self):
        """(int) number of abscissa samples where signal is defined."""
        if self.isempty:
            return 0
        return utils.PrettyInt(len(self._abscissa_vals))

    def __iter__(self):
        """AnalogSignal iterator initialization"""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """AnalogSignal iterator advancer."""
        index = self._index
        if index > self.n_intervals - 1:
            raise StopIteration
        logging.disable(logging.CRITICAL)
        intervalarray = type(self.support)(empty=True)
        exclude = ["_abscissa_vals"]
        attrs = (x for x in self._abscissa.support.__attributes__ if x not in exclude)

        for attr in attrs:
            exec("intervalarray." + attr + " = self._abscissa.support." + attr)
        try:
            intervalarray._data = self._abscissa.support.data[
                tuple([index]), :
            ]  # use np integer indexing! Cool!
        except IndexError:
            # index is out of bounds, so return an empty IntervalArray
            pass
        logging.disable(0)

        self._index += 1

        asa = type(self)([], empty=True)
        exclude = ["_interp", "_support"]
        attrs = (x for x in self.__attributes__ if x not in exclude)
        logging.disable(logging.CRITICAL)
        for attr in attrs:
            exec("asa." + attr + " = self." + attr)
        logging.disable(0)
        asa._restrict_to_interval_array_fast(intervalarray=intervalarray)
        if asa.support.isempty:
            logging.warning(
                "Support is empty. Empty RegularlySampledAnalogSignalArray returned"
            )
            asa = type(self)([], empty=True)

        asa.__renew__()
        return asa

    def empty(self, inplace=True):
        """Remove data (but not metadata) from RegularlySampledAnalogSignalArray.

        Attributes 'data', 'abscissa_vals', and 'support' are all emptied.

        Note: n_signals is preserved.
        """
        n_signals = self.n_signals
        if not inplace:
            out = self._copy_without_data()
        else:
            out = self
            out._data = np.zeros((n_signals, 0))
        out._abscissa.support = type(self.support)(empty=True)
        out._abscissa_vals = []
        out.__renew__()
        return out

    def __getitem__(self, idx):
        """RegularlySampledAnalogSignalArray index access.

        Parameters
        ----------
        idx : IntervalArray, int, slice
            intersect passed intervalarray with support,
            index particular a singular interval or multiple intervals with slice
        """
        intervalslice, signalslice = self._intervalsignalslicer[idx]

        asa = self._subset(signalslice)

        if asa.isempty:
            asa.__renew__()
            return asa

        if isinstance(intervalslice, slice):
            if (
                intervalslice.start == None
                and intervalslice.stop == None
                and intervalslice.step == None
            ):
                asa.__renew__()
                return asa

        newintervals = self._abscissa.support[intervalslice]
        # TODO: this needs to change so that n_signals etc. are preserved
        ################################################################
        if newintervals.isempty:
            logging.warning("Index resulted in empty interval array")
            return self.empty(inplace=False)
        ################################################################

        asa._restrict_to_interval_array_fast(intervalarray=newintervals)
        asa.__renew__()
        return asa

    def _subset(self, idx):
        asa = self.copy()
        try:
            asa._data = np.atleast_2d(self.data[idx, :])
        except IndexError:
            raise IndexError(
                "index {} is out of bounds for n_signals with size {}".format(
                    idx, self.n_signals
                )
            )
        asa.__renew__()
        return asa

    def _copy_without_data(self):
        """Return a copy of self, without data and abscissa_vals.

        Note: the support is left unchanged.
        """
        out = copy.copy(self)  # shallow copy
        out._abscissa_vals = None
        out._data = np.zeros((self.n_signals, 0))
        out = copy.deepcopy(
            out
        )  # just to be on the safe side, but at least now we are not copying the data!
        out.__renew__()
        return out

    def copy(self):
        """Return a copy of the current object."""
        out = copy.deepcopy(self)
        out.__renew__()
        return out

    def median(self, *, axis=1):
        """Returns the median of each signal in RegularlySampledAnalogSignalArray."""
        try:
            medians = np.nanmedian(self.data, axis=axis).squeeze()
            if medians.size == 1:
                try:
                    return np.asscalar(medians)
                except:
                    return np.ndarray.item(medians)
            return medians
        except IndexError:
            raise IndexError(
                "Empty RegularlySampledAnalogSignalArray cannot calculate median"
            )

    def mean(self, *, axis=1):
        """Returns the mean of each signal in RegularlySampledAnalogSignalArray."""
        try:
            means = np.nanmean(self.data, axis=axis).squeeze()
            if means.size == 1:
                try:
                    return np.asscalar(means)
                except:
                    return np.ndarray.item(means)
            return means
        except IndexError:
            raise IndexError(
                "Empty RegularlySampledAnalogSignalArray cannot calculate mean"
            )

    def std(self, *, axis=1):
        """Returns the standard deviation of each signal in RegularlySampledAnalogSignalArray."""
        try:
            stds = np.nanstd(self.data, axis=axis).squeeze()
            if stds.size == 1:
                try:
                    return np.asscalar(stds)
                except:
                    return np.ndarray.item(stds)
            return stds
        except IndexError:
            raise IndexError(
                "Empty RegularlySampledAnalogSignalArray cannot calculate standard deviation"
            )

    def max(self, *, axis=1):
        """Returns the maximum of each signal in RegularlySampledAnalogSignalArray"""
        try:
            maxes = np.amax(self.data, axis=axis).squeeze()
            if maxes.size == 1:
                try:
                    return np.asscalar(maxes)
                except:
                    return np.ndarray.item(maxes)
            return maxes
        except ValueError:
            raise ValueError(
                "Empty RegularlySampledAnalogSignalArray cannot calculate maximum"
            )

    def min(self, *, axis=1):
        """Returns the minimum of each signal in RegularlySampledAnalogSignalArray"""
        try:
            mins = np.amin(self.data, axis=axis).squeeze()
            if mins.size == 1:
                try:
                    return np.asscalar(mins)
                except:
                    return np.ndarray.item(mins)
            return mins
        except ValueError:
            raise ValueError(
                "Empty RegularlySampledAnalogSignalArray cannot calculate minimum"
            )

    def clip(self, min, max):
        """Clip (limit) the values of each signal to min and max as specified.

        Parameters
        ----------
        min : scalar
            Minimum value
        max : scalar
            Maximum value

        Returns
        ----------
        clipped_analogsignalarray : RegularlySampledAnalogSignalArray
            RegularlySampledAnalogSignalArray with the signal clipped with the elements of data, but where the values <
            min are replaced with min and the values > max are replaced
            with max.
        """
        new_data = np.clip(self.data, min, max)
        newasa = self.copy()
        newasa._data = new_data
        return newasa

    def trim(self, start, stop=None, *, fs=None):
        """Trim the RegularlySampledAnalogSignalArray to a single interval.

        Parameters
        ----------
        start : float or two element array-like
            (float) Left boundary of interval in time (seconds) if
            fs=None, otherwise left boundary is start / fs.
            (2 elements) Left and right boundaries in time (seconds) if
            fs=None, otherwise boundaries are left / fs. Stop must be
            None if 2 element start is used.
        stop : float, optional
            Right boundary of interval in time (seconds) if fs=None,
            otherwise right boundary is stop / fs.
        fs : float, optional
            Sampling rate in Hz.

        Returns
        -------
        trim : RegularlySampledAnalogSignalArray
            The RegularlySampledAnalogSignalArray on the interval [start, stop].

        Examples
        --------
        >>> as.trim([0, 3], fs=1)  # recommended for readability
        >>> as.trim(start=0, stop=3, fs=1)
        >>> as.trim(start=[0, 3])
        >>> as.trim(0, 3)
        >>> as.trim((0, 3))
        >>> as.trim([0, 3])
        >>> as.trim(np.array([0, 3]))
        """
        logging.warning("RegularlySampledAnalogSignalArray: Trim may not work!")
        # TODO: do comprehensive input validation
        if stop is not None:
            try:
                start = np.array(start, ndmin=1)
                if len(start) != 1:
                    raise TypeError("start must be a scalar float")
            except TypeError:
                raise TypeError("start must be a scalar float")
            try:
                stop = np.array(stop, ndmin=1)
                if len(stop) != 1:
                    raise TypeError("stop must be a scalar float")
            except TypeError:
                raise TypeError("stop must be a scalar float")
        else:  # start must have two elements
            try:
                if len(np.array(start, ndmin=1)) > 2:
                    raise TypeError(
                        "unsupported input to RegularlySampledAnalogSignalArray.trim()"
                    )
                stop = np.array(start[1], ndmin=1)
                start = np.array(start[0], ndmin=1)
                if len(start) != 1 or len(stop) != 1:
                    raise TypeError("start and stop must be scalar floats")
            except TypeError:
                raise TypeError("start and stop must be scalar floats")

        logging.disable(logging.CRITICAL)
        interval = self._abscissa.support.intersect(
            type(self.support)([start, stop], fs=fs)
        )
        if not interval.isempty:
            analogsignalarray = self[interval]
        else:
            analogsignalarray = type(self)([], empty=True)
        logging.disable(0)
        analogsignalarray.__renew__()
        return analogsignalarray

    @property
    def _ydata_rowsig(self):
        """returns wide-format data s.t. each row is a signal."""
        # LEGACY
        return self.data

    @property
    def _ydata_colsig(self):
        # LEGACY
        """returns skinny-format data s.t. each column is a signal."""
        return self.data.T

    @property
    def _data_rowsig(self):
        """returns wide-format data s.t. each row is a signal."""
        return self.data

    @property
    def _data_colsig(self):
        """returns skinny-format data s.t. each column is a signal."""
        return self.data.T

    def _get_interp1d(
        self,
        *,
        kind="linear",
        copy=True,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=None
    ):
        """returns a scipy interp1d object, extended to have values at all interval
        boundaries!
        """

        if assume_sorted is None:
            assume_sorted = utils.is_sorted(self._abscissa_vals)

        if self.n_signals > 1:
            axis = 1
        else:
            axis = -1

        abscissa_vals = self._abscissa_vals

        if self._ordinate.is_wrapping:
            yvals = self._unwrap(
                self._data_rowsig, self._ordinate.range.min, self._ordinate.range.max
            )  # always interpolate on the unwrapped data!
        else:
            yvals = self._data_rowsig

        lengths = self.lengths
        empty_interval_ids = np.argwhere(lengths == 0).squeeze().tolist()
        first_abscissavals_per_interval_idx = np.insert(np.cumsum(lengths[:-1]), 0, 0)
        first_abscissavals_per_interval_idx[empty_interval_ids] = 0
        last_abscissavals_per_interval_idx = np.cumsum(lengths) - 1
        last_abscissavals_per_interval_idx[empty_interval_ids] = 0
        first_abscissavals_per_interval = self._abscissa_vals[
            first_abscissavals_per_interval_idx
        ]
        last_abscissavals_per_interval = self._abscissa_vals[
            last_abscissavals_per_interval_idx
        ]

        boundary_abscissa_vals = []
        boundary_vals = []
        for ii, (start, stop) in enumerate(self.support.data):
            if lengths[ii] == 0:
                continue
            if first_abscissavals_per_interval[ii] > start:
                boundary_abscissa_vals.append(start)
                boundary_vals.append(yvals[:, first_abscissavals_per_interval_idx[ii]])
                # print('adding {} at abscissa_vals {}'.format(yvals[:,first_abscissavals_per_interval_idx[ii]], start))
            if last_abscissavals_per_interval[ii] < stop:
                boundary_abscissa_vals.append(stop)
                boundary_vals.append(yvals[:, last_abscissavals_per_interval_idx[ii]])

        if boundary_abscissa_vals:
            insert_locs = np.searchsorted(abscissa_vals, boundary_abscissa_vals)
            abscissa_vals = np.insert(
                abscissa_vals, insert_locs, boundary_abscissa_vals
            )
            yvals = np.insert(yvals, insert_locs, np.array(boundary_vals).T, axis=1)

            abscissa_vals, unique_idx = np.unique(abscissa_vals, return_index=True)
            yvals = yvals[:, unique_idx]

        f = interpolate.interp1d(
            x=abscissa_vals,
            y=yvals,
            kind=kind,
            axis=axis,
            copy=copy,
            bounds_error=bounds_error,
            fill_value=fill_value,
            assume_sorted=assume_sorted,
        )
        return f

    def asarray(
        self,
        *,
        where=None,
        at=None,
        kind="linear",
        copy=True,
        bounds_error=False,
        fill_value=np.nan,
        assume_sorted=None,
        recalculate=False,
        store_interp=True,
        n_samples=None,
        split_by_interval=False
    ):
        """returns a data_like array at requested points.

        Parameters
        ----------
        where : array_like or tuple, optional
            array corresponding to np where condition
            e.g., where=(data[1,:]>5) or tuple where=(speed>5,tspeed)
        at : array_like, optional
            Array of points to evaluate array at. If none given, use
            self._abscissa_vals together with 'where' if applicable.
        n_samples: int, optional
            Number of points to interplate at. These points will be
            distributed uniformly from self.support.start to stop.
        split_by_interval: bool
            If True, separate arrays by intervals and return in a list.
        Returns
        -------
        out : (array, array)
            namedtuple tuple (xvals, yvals) of arrays, where xvals is an
            array of abscissa values for which (interpolated) data are returned.
            yvals has shape (n_signals, n_samples)
        """

        # TODO: implement splitting by interval

        if split_by_interval:
            raise NotImplementedError("split_by_interval not yet implemented...")

        XYArray = namedtuple("XYArray", ["xvals", "yvals"])

        if (
            at is None
            and where is None
            and split_by_interval is False
            and n_samples is None
        ):
            xyarray = XYArray(self._abscissa_vals, self._data_rowsig.squeeze())
            return xyarray

        if where is not None:
            assert (
                at is None and n_samples is None
            ), "'where', 'at', and 'n_samples' cannot be used at the same time"
            if isinstance(where, tuple):
                y = np.array(where[1]).squeeze()
                x = where[0]
                assert len(x) == len(
                    y
                ), "'where' condition and array must have same number of elements"
                at = y[x]
            else:
                x = np.asanyarray(where).squeeze()
                assert len(x) == len(
                    self._abscissa_vals
                ), "'where' condition must have same number of elements as self._abscissa_vals"
                at = self._abscissa_vals[x]
        elif at is not None:
            assert (
                n_samples is None
            ), "'at' and 'n_samples' cannot be used at the same time"
        else:
            at = np.linspace(self.support.start, self.support.stop, n_samples)

        at = np.atleast_1d(at)
        if at.ndim > 1:
            raise ValueError("Requested points must be one-dimensional!")
        if at.shape[0] == 0:
            raise ValueError("No points were requested to interpolate")

        # if we made it this far, either at or where has been specified, and at is now well defined.

        kwargs = {
            "kind": kind,
            "copy": copy,
            "bounds_error": bounds_error,
            "fill_value": fill_value,
            "assume_sorted": assume_sorted,
        }

        # retrieve an existing, or construct a new interpolation object
        if recalculate:
            interpobj = self._get_interp1d(**kwargs)
        else:
            try:
                interpobj = self._interp
                if interpobj is None:
                    interpobj = self._get_interp1d(**kwargs)
            except AttributeError:  # does not exist yet
                interpobj = self._get_interp1d(**kwargs)

        # store interpolation object, if desired
        if store_interp:
            self._interp = interpobj

        # do not interpolate points that lie outside the support
        interval_data = self.support.data[:, :, None]
        # use broadcasting to check in a vectorized manner if
        # each sample falls within the support, haha aren't we clever?
        # (n_intervals, n_requested_samples)
        valid = np.logical_and(
            at >= interval_data[:, 0, :], at <= interval_data[:, 1, :]
        )
        valid_mask = np.any(valid, axis=0)
        n_invalid = at.size - np.sum(valid_mask)
        if n_invalid > 0:
            logging.warning(
                "{} values outside the support were removed".format(n_invalid)
            )
        at = at[valid_mask]

        # do the actual interpolation
        if self._ordinate.is_wrapping:
            try:
                if self.is_wrapped:
                    out = self._wrap(
                        interpobj(at),
                        self._ordinate.range.min,
                        self._ordinate.range.max,
                    )
                else:
                    out = interpobj(at)
            except SystemError:
                interpobj = self._get_interp1d(**kwargs)
                if store_interp:
                    self._interp = interpobj
                if self.is_wrapped:
                    out = self._wrap(
                        interpobj(at),
                        self._ordinate.range.min,
                        self._ordinate.range.max,
                    )
                else:
                    out = interpobj(at)
        else:
            try:
                out = interpobj(at)
            except SystemError:
                interpobj = self._get_interp1d(**kwargs)
                if store_interp:
                    self._interp = interpobj
                out = interpobj(at)

        xyarray = XYArray(xvals=np.asanyarray(at), yvals=np.asanyarray(out))
        return xyarray

    def subsample(self, *, fs):
        """Subsamples a RegularlySampledAnalogSignalArray

        WARNING! Aliasing can occur! It is better to use downsample when
        lowering the sampling rate substantially.

        Parameters
        ----------
        fs : float, optional
            Desired output sampling rate, in Hz

        Returns
        -------
        out : RegularlySampledAnalogSignalArray
            Copy of RegularlySampledAnalogSignalArray where data is only stored at the
            new subset of points.
        """

        return self.simplify(ds=1 / fs)

    def simplify(self, *, ds=None, n_samples=None, **kwargs):
        """Returns an RegularlySampledAnalogSignalArray where the data has been
        simplified / subsampled.

        This function is primarily intended to be used for plotting and
        saving vector graphics without having too large file sizes as
        a result of too many points.

        Irrespective of whether 'ds' or 'n_samples' are used, the exact
        underlying support is propagated, and the first and last points
        of the supports are always included, even if this would cause
        n_samples or ds to be violated.

        WARNING! Simplify can create nan samples, when requesting a timestamp
        within an interval, but outside of the (first, last) abscissa_vals within that
        interval, since we don't extrapolate, but only interpolate. # TODO: fix

        Parameters
        ----------
        ds : float, optional
            Time (in seconds), in which to step points.
        n_samples : int, optional
            Number of points at which to intepolate data. If ds is None
            and n_samples is None, then default is to use n_samples=5,000

        Returns
        -------
        out : RegularlySampledAnalogSignalArray
            Copy of RegularlySampledAnalogSignalArray where data is only stored at the
            new subset of points.
        """

        if self.isempty:
            return self

        # legacy kwarg support:
        n_points = kwargs.pop("n_points", False)
        if n_points:
            n_samples = n_points

        if ds is not None and n_samples is not None:
            raise ValueError("ds and n_samples cannot be used together")

        if n_samples is not None:
            assert float(
                n_samples
            ).is_integer(), "n_samples must be a positive integer!"
            assert n_samples > 1, "n_samples must be a positive integer > 1"
            # determine ds from number of desired points:
            ds = self.support.length / (n_samples - 1)

        if ds is None:
            # neither n_samples nor ds was specified, so assume defaults:
            n_samples = np.min((5000, 250 + self.n_samples // 2, self.n_samples))
            ds = self.support.length / (n_samples - 1)

        # build list of points at which to evaluate the RegularlySampledAnalogSignalArray

        # we exclude all empty intervals:
        at = []
        lengths = self.lengths
        empty_interval_ids = np.argwhere(lengths == 0).squeeze().tolist()
        first_abscissavals_per_interval_idx = np.insert(np.cumsum(lengths[:-1]), 0, 0)
        first_abscissavals_per_interval_idx[empty_interval_ids] = 0
        last_abscissavals_per_interval_idx = np.cumsum(lengths) - 1
        last_abscissavals_per_interval_idx[empty_interval_ids] = 0
        first_abscissavals_per_interval = self._abscissa_vals[
            first_abscissavals_per_interval_idx
        ]
        last_abscissavals_per_interval = self._abscissa_vals[
            last_abscissavals_per_interval_idx
        ]

        for ii, (start, stop) in enumerate(self.support.data):
            if lengths[ii] == 0:
                continue
            newxvals = utils.frange(
                first_abscissavals_per_interval[ii],
                last_abscissavals_per_interval[ii],
                step=ds,
            ).tolist()
            at.extend(newxvals)
            try:
                if newxvals[-1] < last_abscissavals_per_interval[ii]:
                    at.append(last_abscissavals_per_interval[ii])
            except IndexError:
                at.append(first_abscissavals_per_interval[ii])
                at.append(last_abscissavals_per_interval[ii])

        _, yvals = self.asarray(at=at, recalculate=True, store_interp=False)
        yvals = np.array(yvals, ndmin=2)

        asa = self.copy()
        asa._abscissa_vals = np.asanyarray(at)
        asa._data = yvals
        asa._fs = 1 / ds

        return asa

    def join(self, other, *, mode=None, inplace=False):
        """Join another RegularlySampledAnalogSignalArray to this one.

        WARNING! Numerical precision might cause some epochs to be considered
        non-disjoint even when they really are, so a better check than ep1[ep2].isempty
        is to check for samples contained in the intersection of ep1 and ep2.

        Parameters
        ----------
        other : RegularlySampledAnalogSignalArray
            RegularlySampledAnalogSignalArray (or derived type) to join to the current
            RegularlySampledAnalogSignalArray. Other must have the same number of signals as
            the current RegularlySampledAnalogSignalArray.
        mode : string, optional
            One of ['max', 'min', 'left', 'right', 'mean']. Specifies how the
            signals are merged inside overlapping intervals. Default is 'left'.
        inplace : boolean, optional
            If True, then current RegularlySampledAnalogSignalArray is modified. If False, then
            a copy with the joined result is returned. Default is False.

        Returns
        -------
        out : RegularlySampledAnalogSignalArray
            Copy of RegularlySampledAnalogSignalArray where the new RegularlySampledAnalogSignalArray has been
            joined to the current RegularlySampledAnalogSignalArray.
        """

        if mode is None:
            mode = "left"

        asa = self.copy()  # copy without data since we change data at the end?

        times = np.zeros((1, 0))
        data = np.zeros((asa.n_signals, 0))

        # if ASAs are disjoint:
        if not self.support[other.support].length > 50 * float_info.epsilon:
            # do a simple-as-butter join (concat) and sort
            times = np.append(times, self._abscissa_vals)
            data = np.hstack((data, self.data))
            times = np.append(times, other._abscissa_vals)
            data = np.hstack((data, other.data))
        else:  # not disjoint
            both_eps = self.support[other.support]
            self_eps = self.support - both_eps - other.support
            other_eps = other.support - both_eps - self.support

            if mode == "left":
                self_eps += both_eps
                # print(self_eps)

                tmp = self[self_eps]
                times = np.append(times, tmp._abscissa_vals)
                data = np.hstack((data, tmp.data))

                if not other_eps.isempty:
                    tmp = other[other_eps]
                    times = np.append(times, tmp._abscissa_vals)
                    data = np.hstack((data, tmp.data))
            elif mode == "right":
                other_eps += both_eps

                tmp = other[other_eps]
                times = np.append(times, tmp._abscissa_vals)
                data = np.hstack((data, tmp.data))

                if not self_eps.isempty:
                    tmp = self[self_eps]
                    times = np.append(times, tmp._abscissa_vals)
                    data = np.hstack((data, tmp.data))
            else:
                raise NotImplementedError(
                    "asa.join() has not yet been implemented for mode '{}'!".format(
                        mode
                    )
                )

        sample_order = np.argsort(times)
        times = times[sample_order]
        data = data[:, sample_order]

        asa._data = data
        asa._abscissa_vals = times
        dom1 = self.domain
        dom2 = other.domain
        asa._abscissa.support = (self.support + other.support).merge()
        asa._abscissa.support.domain = (dom1 + dom2).merge()
        return asa

    def _pdf(self, bins=None, n_samples=None):
        """Return the probability distribution function for each signal."""
        from scipy import integrate

        if bins is None:
            bins = 100

        if n_samples is None:
            n_samples = 100

        if self.n_signals > 1:
            raise NotImplementedError("multiple signals not supported yet!")

        # fx, bins = np.histogram(self.data.squeeze(), bins=bins, normed=True)
        fx, bins = np.histogram(self.data.squeeze(), bins=bins)
        bin_centers = (bins + (bins[1] - bins[0]) / 2)[:-1]

        Ifx = integrate.simps(fx, bin_centers)

        pdf = type(self)(
            abscissa_vals=bin_centers,
            data=fx / Ifx,
            fs=1 / (bin_centers[1] - bin_centers[0]),
            support=type(self.support)(self.data.min(), self.data.max()),
        ).simplify(n_samples=n_samples)

        return pdf

        # data = []
        # for signal in self.data:
        #     fx, bins = np.histogram(signal, bins=bins)
        #     bin_centers = (bins + (bins[1]-bins[0])/2)[:-1]

    def _cdf(self, n_samples=None):
        """Return the probability distribution function for each signal."""

        if n_samples is None:
            n_samples = 100

        if self.n_signals > 1:
            raise NotImplementedError("multiple signals not supported yet!")

        X = np.sort(self.data.squeeze())
        F = np.array(range(self.n_samples)) / float(self.n_samples)

        logging.disable(logging.CRITICAL)
        cdf = type(self)(
            abscissa_vals=X,
            data=F,
            support=type(self.support)(self.data.min(), self.data.max()),
        ).simplify(n_samples=n_samples)
        logging.disable(0)

        return cdf

    def _eegplot(self, ax=None, normalize=False, pad=None, fill=True, color=None):
        """custom_func docstring goes here."""

        import matplotlib.pyplot as plt
        from ..plotting import utils as plotutils

        if ax is None:
            ax = plt.gca()

        xmin = self.support.min
        xmax = self.support.max
        xvals = self._abscissa_vals

        if pad is None:
            pad = np.mean(self.data) / 2

        data = self.data.copy()

        if normalize:
            peak_vals = self.max()
            data = (data.T / peak_vals).T

        n_traces = self.n_signals

        for tt, trace in enumerate(data):
            if color is None:
                line = ax.plot(
                    xvals, tt * pad + trace, zorder=int(10 + 2 * n_traces - 2 * tt)
                )
            else:
                line = ax.plot(
                    xvals,
                    tt * pad + trace,
                    zorder=int(10 + 2 * n_traces - 2 * tt),
                    color=color,
                )
            if fill:
                # Get the color from the current curve
                fillcolor = line[0].get_color()
                ax.fill_between(
                    xvals,
                    tt * pad,
                    tt * pad + trace,
                    alpha=0.3,
                    color=fillcolor,
                    zorder=int(10 + 2 * n_traces - 2 * tt - 1),
                )

        ax.set_xlim(xmin, xmax)
        if pad != 0:
            # yticks = np.arange(n_traces)*pad + 0.5*pad
            yticks = []
            ax.set_yticks(yticks)
            ax.set_xlabel(self._abscissa.label)
            ax.set_ylabel(self._ordinate.label)
            plotutils.no_yticks(ax)
            plotutils.clear_left(ax)

        plotutils.clear_top(ax)
        plotutils.clear_right(ax)

        return ax

    def __setattr__(self, name, value):
        # https://stackoverflow.com/questions/4017572/how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
        name = self.__aliases__.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # https://stackoverflow.com/questions/4017572/how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.__aliases__.get(name, name)
        # return getattr(self, name) #Causes infinite recursion on non-existent attribute
        return object.__getattribute__(self, name)


def legacyASAkwargs(**kwargs):
    """Provide support for legacy AnalogSignalArray kwargs.

    kwarg: time <==> timestamps <==> abscissa_vals
    kwarg: data <==> ydata

    Examples
    --------
    asa = nel.AnalogSignalArray(time=..., data=...)
    asa = nel.AnalogSignalArray(timestamps=..., data=...)
    asa = nel.AnalogSignalArray(time=..., ydata=...)
    asa = nel.AnalogSignalArray(ydata=...)
    asa = nel.AnalogSignalArray(abscissa_vals=..., data=...)
    """

    def only_one_of(*args):
        num_non_null_args = 0
        out = None
        for arg in args:
            if arg is not None:
                num_non_null_args += 1
                out = arg
        if num_non_null_args > 1:
            raise ValueError("multiple conflicting arguments received")
        return out

    # legacy ASA constructor support for backward compatibility
    abscissa_vals = kwargs.pop("abscissa_vals", None)
    timestamps = kwargs.pop("timestamps", None)
    time = kwargs.pop("time", None)
    # only one of the above, else raise exception
    abscissa_vals = only_one_of(abscissa_vals, timestamps, time)
    if abscissa_vals is not None:
        kwargs["abscissa_vals"] = abscissa_vals

    data = kwargs.pop("data", None)
    ydata = kwargs.pop("ydata", None)
    # only one of the above, else raise exception
    data = only_one_of(data, ydata)
    if data is not None:
        kwargs["data"] = data

    return kwargs


########################################################################
# class AnalogSignalArray
########################################################################
class AnalogSignalArray(RegularlySampledAnalogSignalArray):
    """Custom ASA docstring with kwarg descriptions.

    TODO: add the ASA docstring here, using the aliases in the constructor.
    """

    # specify class-specific aliases:
    __aliases__ = {
        "time": "abscissa_vals",
        "_time": "_abscissa_vals",
        "n_epochs": "n_intervals",
        "ydata": "data",  # legacy support
        "_ydata": "_data",  # legacy support
    }

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}

        # legacy ASA constructor support for backward compatibility
        kwargs = legacyASAkwargs(**kwargs)

        support = kwargs.get("support", core.EpochArray(empty=True))
        abscissa = kwargs.get(
            "abscissa", core.AnalogSignalArrayAbscissa(support=support)
        )
        ordinate = kwargs.get("ordinate", core.AnalogSignalArrayOrdinate())

        kwargs["abscissa"] = abscissa
        kwargs["ordinate"] = ordinate

        super().__init__(*args, **kwargs)


########################################################################
# class PositionArray
########################################################################
class PositionArray(AnalogSignalArray):
    """Custom PositionArray docstring with kwarg descriptions.

    TODO: add the docstring here, using the aliases in the constructor.
    """

    # specify class-specific aliases:
    __aliases__ = {"posdata": "data"}

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}
        xlim = kwargs.pop("xlim", None)
        ylim = kwargs.pop("ylim", None)
        super().__init__(*args, **kwargs)
        self._xlim = xlim
        self._ylim = ylim

    @property
    def is_2d(self):
        try:
            return self.n_signals == 2
        except IndexError:
            return False

    @property
    def is_1d(self):
        try:
            return self.n_signals == 1
        except IndexError:
            return False

    @property
    def x(self):
        """return x-values, as numpy array."""
        return self.data[0, :]

    @property
    def y(self):
        """return y-values, as numpy array."""
        if self.is_2d:
            return self.data[1, :]
        raise ValueError(
            "PositionArray is not 2 dimensional, so y-values are undefined!"
        )

    @property
    def xlim(self):
        if self.is_2d:
            return self._xlim
        raise ValueError(
            "PositionArray is not 2 dimensional, so xlim is not undefined!"
        )

    @xlim.setter
    def xlim(self, val):
        if self.is_2d:
            self._xlim = xlim  # noqa: F821
        raise ValueError(
            "PositionArray is not 2 dimensional, so xlim cannot be defined!"
        )

    @property
    def ylim(self):
        if self.is_2d:
            return self._ylim
        raise ValueError(
            "PositionArray is not 2 dimensional, so ylim is not undefined!"
        )

    @ylim.setter
    def ylim(self, val):
        if self.is_2d:
            self._ylim = ylim  # noqa: F821
        raise ValueError(
            "PositionArray is not 2 dimensional, so ylim cannot be defined!"
        )


########################################################################
# class IMUSensorArray
########################################################################
class IMUSensorArray(RegularlySampledAnalogSignalArray):
    """Custom IMUSensorArray docstring with kwarg descriptions.

    TODO: add the docstring here, using the aliases in the constructor.
    """

    # specify class-specific aliases:
    __aliases__ = {}

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}
        super().__init__(*args, **kwargs)


########################################################################
# class MinimalExampleArray
########################################################################
class MinimalExampleArray(RegularlySampledAnalogSignalArray):
    """Custom MinimalExampleArray docstring with kwarg descriptions.

    TODO: add the docstring here, using the aliases in the constructor.
    """

    # specify class-specific aliases:
    __aliases__ = {}

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}
        super().__init__(*args, **kwargs)

    def custom_func(self):
        """custom_func docstring goes here."""
        print("Woot! We have some special skillz!")
