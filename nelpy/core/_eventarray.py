__all__ = ['EventArray']

"""EventArray

EventArray (supports binning)
  |_ ValueEventArray (does not support binning, nor queries)
    |_ StatefulEventArray (supports queries, casting to-and-from AnalogSignalArrays)

eva, veva, seva
"""

import warnings
import numpy as np
import copy
import numbers

from functools import wraps
from scipy import interpolate
from sys import float_info
from collections import namedtuple

from .. import core
from .. import utils
from .. import version

# TODO: EpochArray from EventArray
# TODO: casting any nelpy obj to EpochArray returns its support with
#       proper domain

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

class EpochSignalSlicer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """epochs, signals"""
        # by default, keep all signals
        signalslice = slice(None, None, None)
        if isinstance(*args, int):
            epochslice = args[0]
        elif isinstance(*args, core.EpochArray):
            epochslice = args[0]
        else:
            try:
                slices = np.s_[args]; slices = slices[0]
                if len(slices) > 2:
                    raise IndexError("only [epochs, signal] slicing is supported at this time!")
                elif len(slices) == 2:
                    epochslice, signalslice = slices
                else:
                    epochslice = slices[0]
            except TypeError:
                # only epoch to slice:
                epochslice = slices

        return epochslice, signalslice

class DataSlicer(object):

    def __init__(self, parent):
        self._parent = parent

    def _data_generator(self, epoch_indices, signalslice):
        for start, stop in epoch_indices:
            yield self._parent._ydata[signalslice, start: stop]

    def __getitem__(self, idx):
        epochslice, signalslice = self._parent._epochsignalslicer[idx]

        epoch_indices = self._parent._data_epoch_indices()
        epoch_indices = np.atleast_2d(epoch_indices[epochslice])

        if len(epoch_indices) < 2:
            start, stop = epoch_indices[0]
            return self._parent._ydata[signalslice, start: stop]
        else:
            return self._data_generator(epoch_indices, signalslice)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index

        if index > self._parent.n_epochs - 1:
            raise StopIteration

        epoch_indices = self._parent._data_epoch_indices()
        epoch_indices = epoch_indices[index]
        start, stop = epoch_indices

        self._index +=1

        return self._parent._ydata[:, start: stop]

class TimestampSlicer(object):

    def __init__(self, parent):
        self._parent = parent

    def _timestamp_generator(self, epoch_indices):
        for start, stop in epoch_indices:
            yield self._parent._time[start: stop]

    def __getitem__(self, idx):
        epochslice, signalslice = self._parent._epochsignalslicer[idx]

        epoch_indices = self._parent._data_epoch_indices()
        epoch_indices = np.atleast_2d(epoch_indices[epochslice])

        if len(epoch_indices) < 2:
            start, stop = epoch_indices[0]
            return self._parent._time[start: stop]
        else:
            return self._timestamp_generator(epoch_indices)

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        index = self._index

        if index > self._parent.n_epochs - 1:
            raise StopIteration

        epoch_indices = self._parent._data_epoch_indices()
        epoch_indices = epoch_indices[index]
        start, stop = epoch_indices

        self._index +=1

        return self._parent._time[start: stop]


def eva_init_wrapper(func):
    """Decorator that helps figure out timestamps, and sample numbers"""

    @wraps(func)
    def wrapper(*args, **kwargs):

        if kwargs.get('empty', False):
            func(*args, **kwargs)
            return

        if len(args) > 2:
            raise TypeError("__init__() takes 1 positional arguments but {} positional arguments (and {} keyword-only arguments) were given".format(len(args)-1, len(kwargs.items())))

        ydata = kwargs.get('ydata', [])
        if ydata == []:
            ydata = args[1]

        if ydata == []:
            warnings.warn('No data! Returning empty EventArray.')
            func(*args, **kwargs)
            return

        #check if single EventSignal or multiple EventSignals in array
        #and standardize ydata to 2D
        ydata = np.squeeze(ydata)
        try:
            if(ydata.shape[0] == ydata.size):
                ydata = np.array(ydata,ndmin=2)
        except ValueError:
            raise TypeError("Unsupported ydata type!")

        time = kwargs.get('timestamps', None)
        if time is None:
            time = np.linspace(0, ydata.shape[1]/fs, ydata.shape[1]+1)
            time = time[:-1]

        kwargs['ydata'] = ydata
        kwargs['timestamps'] = np.squeeze(time)

        func(args[0], **kwargs)
        return

    return wrapper

########################################################################
# class EventArray
########################################################################
class EventArray:

    raise NotImplementedError
    """

    Temp text: like spike train, in that EventSignals can have arbitrary
    event times. Collapsing them then results in a single EventSignal with
    categorical labels being the union of all signal labels.

    Categorical event signal(s) with irregular sampling rates and same
    support.

    Parameters
    ----------
    XXX : np.array
        With shape (M,N).

    Attributes
    ----------
    XXX : np.array
        With shape (M,N).
    """
    raise NotImplementedError
    __attributes__ = ['_ydata','_time', '_support', \
                      '_interp', '_step', '_labels']

    @eva_init_wrapper
    def __init__(self, ydata=[], *, timestamps=None, fs=None,
                 step=None, merge_sample_gap=0, support=None,
                 in_memory=True, labels=None, empty=False):

        raise NotImplementedError

        self._epochsignalslicer = EpochSignalSlicer(self)
        self._epochdata = DataSlicer(self)
        self._epochtime = TimestampSlicer(self)

        self.__version__ = version.__version__

        if(empty):
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = core.EpochArray(empty=True)
            return

        self._step = step
        self._fs = fs

        # Note; if we have an empty array of ydata with no dimension,
        # then calling len(ydata) will return a TypeError
        try:
            # if no ydata are given return empty EventArray
            if ydata.size == 0:
                self.__init__(empty=True)
                return
        except TypeError:
            warnings.warn("unsupported type; creating empty EventArray")
            self.__init__(empty=True)
            return

        # Note: if both time and ydata are given and dimensionality does not
        # match, then TypeError!

        time = np.squeeze(timestamps).astype(float)
        if(time.shape[0] != ydata.shape[1]):
            # self.__init__([],empty=True)
            raise TypeError("time and ydata size mismatch! Note: ydata "
                            "is expected to have rows containing signals")
        #data is not sorted and user wants it to be
        # TODO: use faster is_sort from jagular
        if not utils.is_sorted(time):
            warnings.warn("Data is _not_ sorted! Data will be sorted "\
                            "automatically.")
            ind = np.argsort(time)
            time = time[ind]
            ydata = np.take(a=ydata, indices=ind, axis=-1)

        self._ydata = ydata
        self._time = time

        #handle labels
        if labels is not None:
            labels = np.asarray(labels,dtype=np.str)
            #label size doesn't match
            if labels.shape[0] > ydata.shape[0]:
                warnings.warn("More labels than ydata! labels are sliced to "
                              "size of ydata")
                labels = labels[0:ydata.shape[0]]
            elif labels.shape[0] < ydata.shape[0]:
                warnings.warn("Less labels than time! labels are filled with "
                              "None to match ydata shape")
                for i in range(labels.shape[0],ydata.shape[0]):
                    labels.append(None)
        self._labels = labels

        # Alright, let's handle all the possible parameter cases!
        if support is not None:
            self._restrict_to_epoch_array_fast(epocharray=support)
        else:
            warnings.warn("creating support from time and "
                            "sampling rate, fs!")
            self._support = core.EpochArray(
                utils.get_contiguous_segments(
                    self.time,
                    step=self._step,
                    fs=fs,
                    in_memory=in_memory))
            if merge_sample_gap > 0:
                self._support = self._support.merge(gap=merge_sample_gap)

        if np.abs((self.fs - self._estimate_fs())/self.fs) > 0.01:
            warnings.warn("estimated fs and provided fs differ by more than 1%")

    def _data_epoch_indices(self):
        raise NotImplementedError
        """Docstring goes here.
        We use this to get the indices of samples / timestamps within epochs
        """
        tmp = np.insert(np.cumsum(self.lengths),0,0)
        indices = np.vstack((tmp[:-1], tmp[1:])).T
        return indices

    @property
    def signals(self):
        raise NotImplementedError
        """Returns a list of EventArrays, each array containing
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
            signals.append(self[:,ii])
        return signals

    def __mul__(self, other):
        """overloaded * operator."""
        raise NotImplementedError

    def __add__(self, other):
        """overloaded + operator."""
        raise NotImplementedError

    def __sub__(self, other):
        """overloaded - operator."""
        raise NotImplementedError

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """overloaded / operator."""
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _drop_empty_epochs(self):
        """Drops empty epochs from support. In-place."""
        keep_epoch_ids = np.argwhere(self.lengths).squeeze().tolist()
        self._support = self.support[keep_epoch_ids]
        return self

    def add_signal(self, signal, label=None):
        """Docstring goes here.
        Basically we add a signal, and we add a label
        """
        raise NotImplementedError

    def _restrict_to_epoch_array(self, *, epocharray=None, update=True):
        raise NotImplementedError
        """Restrict self._time and self._ydata to an EpochArray. If no
        EpochArray is specified, self._support is used.

        This function is quite slow, as it checks each sample for inclusion.
        It does this in a vectorized form, which is fast for small or moderately
        sized objects, but the memory penalty can be large, and it becomes very
        slow for large objects. Consequently, _restrict_to_epoch_array_fast
        should be used when possible.

        Parameters
        ----------
        epocharray : EpochArray, optional
        	EpochArray on which to restrict AnalogSignal. Default is
        	self._support
        update : bool, optional
        	Overwrite self._support with epocharray if True (default).
        """
        if epocharray is None:
            epocharray = self._support
            update = False # support did not change; no need to update

        try:
            if epocharray.isempty:
                warnings.warn("Support specified is empty")
                # self.__init__([],empty=True)
                exclude = ['_support','_ydata','_fs','_step']
                attrs = (x for x in self.__attributes__ if x not in exclude)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for attr in attrs:
                        exec("self." + attr + " = None")
                self._ydata = np.zeros([0,self._ydata.shape[0]])
                self._ydata[:] = np.nan
                self._support = epocharray
                return
        except AttributeError:
            raise AttributeError("EpochArray expected")

        indices = []
        for eptime in epocharray.time:
            t_start = eptime[0]
            t_stop = eptime[1]
            indices.append((self._time >= t_start) & (self._time < t_stop))
        indices = np.any(np.column_stack(indices), axis=1)
        if np.count_nonzero(indices) < len(self._time):
            warnings.warn(
                'ignoring signal outside of support')
        try:
            self._ydata = self._ydata[:,indices]
        except IndexError:
            self._ydata = np.zeros([0,self._ydata.shape[0]])
            self._ydata[:] = np.nan
        self._time = self._time[indices]
        if update:
            self._support = epocharray

    @property
    def lengths(self):
        raise NotImplementedError
        """(list) The number of samples in each epoch."""
        indices = []
        for eptime in self.support.time:
            t_start = eptime[0]
            t_stop = eptime[1]
            frm, to = np.searchsorted(self._time, (t_start, t_stop))
            indices.append((frm, to))
        indices = np.array(indices, ndmin=2)
        lengths = np.atleast_1d(np.diff(indices).squeeze())
        return lengths

    @property
    def labels(self):
        raise NotImplementedError
        """(list) The number of samples (events) in each epoch."""
        # TODO: make this faster and better!
        return self._labels

    @property
    def n_signals(self):
        """(int) The number of signals."""
        try:
            return utils.PrettyInt(self._ydata.shape[0])
        except AttributeError:
            return 0

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty EventArray" + address_str + ">"
        if self.n_epochs > 1:
            epstr = " ({} segments)".format(self.n_epochs)
        else:
            epstr = ""
        try:
            if(self.n_signals > 0):
                nstr = " %s signals%s" % (self.n_signals, epstr)
        except IndexError:
            nstr = " 1 signal%s" % epstr
        dstr = " for a total of {}".format(utils.PrettyDuration(self.support.duration))
        return "<EventArray%s:%s>%s" % (address_str, nstr, dstr)

    def partition(self, ds=None, n_epochs=None):
        """Returns an EventArray whose support has been
        partitioned.

        # Irrespective of whether 'ds' or 'n_epochs' are used, the exact
        # underlying support is propagated, and the first and last points
        # of the supports are always included, even if this would cause
        # n_points or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Maximum duration (in seconds), for each epoch.
        n_points : int, optional
            Number of epochs. If ds is None and n_epochs is None, then
            default is to use n_epochs = 100

        Returns
        -------
        out : EventArray
            EventArray that has been partitioned.
        """

        out = copy.copy(self)
        out._support = out.support.partition(ds=ds, n_epochs=n_epochs)
        return out

    @property
    def ydata(self):
        raise NotImplementedError
        """(np.array N-Dimensional) ydata that was initially passed in but transposed
        """
        return self._ydata

    @property
    def support(self):
        """(nelpy.EpochArray) The support of the underlying EventArray
        (in seconds).
         """
        return self._support

    @property
    def time(self):
        raise NotImplementedError
        """(np.array 1D) Time in seconds."""
        return self._time

    @property
    def isempty(self):
        raise NotImplementedError
        """(bool) checks length of ydata input"""
        try:
            return len(self._ydata) == 0
        except TypeError: #TypeError should happen if _ydata = []
            return True

    @property
    def n_epochs(self):
        """(int) number of epochs in EventArray"""
        return self._support.n_epochs

    @property
    def n_samples(self):
        raise NotImplementedError
        """(int) number of time samples where signal is defined."""
        if self.isempty:
            return 0
        return utils.PrettyInt(len(self.time))

    def __iter__(self):
        """EventArray iterator initialization"""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        raise NotImplementedError
        """EventArray iterator advancer."""
        index = self._index
        if index > self.n_epochs - 1:
            raise StopIteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epoch = core.EpochArray(empty=True)
            exclude = ["_time"]
            attrs = (x for x in self._support.__attributes__ if x not in exclude)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("epoch." + attr + " = self._support." + attr)
                try:
                    epoch._time = self._support.time[[index], :]  # use np integer indexing! Cool!
                except IndexError:
                    # index is out of bounds, so return an empty EpochArray
                    pass

        self._index += 1

        eva = EventArray([],empty=True)
        exclude = ['_interp','_support']
        attrs = (x for x in self.__attributes__ if x not in exclude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("eva." + attr + " = self." + attr)
        eva._restrict_to_epoch_array_fast(epocharray=epoch)
        if(eva.support.isempty):
            warnings.warn("Support is empty. Empty EventArray returned")
            eva = EventArray([],empty=True)
        return eva

    def __getitem__(self, idx):
        raise NotImplementedError
        """EventArray index access.
        Parameters
        Parameters
        ----------
        idx : EpochArray, int, slice
            intersect passed epocharray with support,
            index particular a singular epoch or multiple epochs with slice
        """
        epochslice, signalslice = self._epochsignalslicer[idx]

        eva = self._subset(signalslice)

        if eva.isempty:
            return eva

        if isinstance(epochslice, slice):
            if epochslice.start == None and epochslice.stop == None and epochslice.step == None:
                return eva

        newepochs = self._support[epochslice]
        # TODO: this needs to change so that n_signals etc. are preserved
        ################################################################
        if newepochs.isempty:
            warnings.warn("Index resulted in empty epoch array")
            return EventArray([], empty=True)
        ################################################################

        eva._restrict_to_epoch_array_fast(epocharray=newepochs)

        return eva

    def _subset(self, idx):
        raise NotImplementedError
        eva = self.copy()
        try:
            eva._ydata = np.atleast_2d(self._ydata[idx,:])
        except IndexError:
            raise IndexError("index {} is out of bounds for n_signals with size {}".format(idx, self.n_signals))
        return eva

    def copy(self):
        eva = EventArray([], empty=True)
        exclude = []
        attrs = (x for x in self.__attributes__ if x not in exclude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("eva." + attr + " = self." + attr)
        return eva

#----------------------------------------------------------------------#
#======================================================================#
