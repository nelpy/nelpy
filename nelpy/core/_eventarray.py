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

from abc import ABC, abstractmethod

# from functools import wraps
# from scipy import interpolate
# from sys import float_info
# from collections import namedtuple

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

class EpochSourceSlicer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """epochs, sources"""
        # by default, keep all sources
        sourceslice = slice(None, None, None)
        if isinstance(*args, int):
            epochslice = args[0]
        elif isinstance(*args, core.EpochArray):
            epochslice = args[0]
        else:
            try:
                slices = np.s_[args]; slices = slices[0]
                if len(slices) > 2:
                    raise IndexError("only [epochs, sources] slicing is supported at this time!")
                elif len(slices) == 2:
                    epochslice, sourceslice = slices
                else:
                    epochslice = slices[0]
            except TypeError:
                # only epoch to slice:
                epochslice = slices

        return epochslice, sourceslice

class ItemGetter_loc(object):
    """.loc is primarily label based (that is, source_id based)

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
        """epochs, sources"""
        epochslice, sourceslice = self.obj._slicer[idx]

        # first convert source slice into list
        if isinstance(sourceslice, slice):
            start = sourceslice.start
            stop = sourceslice.stop
            istep = sourceslice.step
            try:
                if start is None:
                    istart = 0
                else:
                    istart = self.obj._source_ids.index(start)
            except ValueError:
                raise KeyError('source_id {} could not be found in EventArray!'.format(start))
            try:
                if stop is None:
                    istop = self.obj.n_sources
                else:
                    istop = self.obj._source_ids.index(stop) + 1
            except ValueError:
                raise KeyError('source_id {} could not be found in EventArray!'.format(stop))
            if istep is None:
                istep = 1
            if istep < 0:
                istop -=1
                istart -=1
                istart, istop = istop, istart
            source_idx_list = list(range(istart, istop, istep))
        else:
            source_idx_list = []
            sourceslice = np.atleast_1d(sourceslice)
            for source in sourceslice:
                try:
                    uidx = self.obj.source_ids.index(source)
                except ValueError:
                    raise KeyError("source_id {} could not be found in EventArray!".format(source))
                else:
                    source_idx_list.append(uidx)

        if not isinstance(source_idx_list, list):
            source_idx_list = list(source_idx_list)
        out = copy.copy(self.obj)
        out._time = out._time[source_idx_list]
        singlesource = len(out._time)==1
        if singlesource:
            out._time = np.array(out._time[0], ndmin=2)
        out._source_ids = list(np.atleast_1d(np.atleast_1d(out._source_ids)[source_idx_list]))
        out._source_labels = list(np.atleast_1d(np.atleast_1d(out._source_labels)[source_idx_list]))
        # TODO: update tags
        if isinstance(epochslice, slice):
            if epochslice.start == None and epochslice.stop == None and epochslice.step == None:
                out.loc = ItemGetter_loc(out)
                out.iloc = ItemGetter_iloc(out)
                return out
        out = out._epochslicer(epochslice)
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

class ItemGetter_iloc(object):
    """.iloc is primarily integer position based (from 0 to length-1
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
        """epochs, sources"""
        epochslice, sourceslice = self.obj._slicer[idx]
        out = copy.copy(self.obj)
        if isinstance(sourceslice, int):
            sourceslice = [sourceslice]
        out._time = out._time[sourceslice]
        singlesource = len(out._time)==1
        if singlesource:
            out._time = np.array(out._time[0], ndmin=2)
        out._source_ids = list(np.atleast_1d(np.atleast_1d(out._source_ids)[sourceslice]))
        out._source_labels = list(np.atleast_1d(np.atleast_1d(out._source_labels)[sourceslice]))
        # TODO: update tags
        if isinstance(epochslice, slice):
            if epochslice.start == None and epochslice.stop == None and epochslice.step == None:
                out.loc = ItemGetter_loc(out)
                out.iloc = ItemGetter_iloc(out)
                return out
        out = out._epochslicer(epochslice)
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

########################################################################
# class EventBase
########################################################################
class EventBase(ABC):
    """Base class for EventArray, ValueEventArray and StatefulEventArray.

    NOTE: This class can't really be instantiated, almost like a pseudo
    abstract class. In particular, during initialization it might fail
    because it checks the n_sources of its derived classes to validate
    input to source_ids and source_labels. If NoneTypes are used, then you
    may actually succeed in creating an instance of this class, but it
    will be pretty useless.

    Parameters
    ----------
    fs: float, optional
        Sampling rate / frequency (Hz).
    source_ids : list of int, optional
        Source IDs preferabbly in integers
    source_labels : list of str, optional
        Labels corresponding to sources. Default casts source_ids to str.
    label : str or None, optional
        Information pertaining to the source of the event train.


    Attributes
    ----------
    n_sources : int
        Number of sources in event train.
    source_ids : list of int
        Source integer IDs.
    source_labels : list of str
        Labels corresponding to sources. Default casts source_ids to str.
    source_tags : dict of tags and corresponding source_ids
        Tags corresponding to sources.
    issempty
    **********
    support
    **********
    fs: float
        Sampling frequency (Hz).
    label : str or None
        Information pertaining to the source of the event train.
    """

    __attributes__ = ["_fs", "_source_ids", "_source_labels", "_source_tags", "_label"]

    def __init__(self, *, fs=None, source_ids=None, source_labels=None,
                 source_tags=None, label=None, empty=False):

        self.__version__ = version.__version__

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = core.EpochArray(empty=True)
            self._slicer = EpochSourceSlicer(self)
            self.loc = ItemGetter_loc(self)
            self.iloc = ItemGetter_iloc(self)
            return

        # set initial fs to None
        self._fs = None
        # then attempt to update the fs; this does input validation:
        self.fs = fs

        # WARNING! we need to ensure that self.n_sources can work BEFORE
        # we can set self.source_ids or self.source_labels, since those
        # setters check that the lengths of the inputs are consistent
        # with self.n_sources.

        # inherit source IDs if available, otherwise initialize to default
        if source_ids is None:
            source_ids = list(range(self.n_sources))

        source_ids = np.array(source_ids, ndmin=1)  # standardize source_ids

        # if source_labels is empty, default to source_ids
        if source_labels is None:
            source_labels = source_ids

        source_labels = np.array(source_labels, ndmin=1)  # standardize

        self.source_ids = source_ids
        self.source_labels = source_labels
        self._source_tags = source_tags  # no input validation yet
        self.label = label

        self._slicer = EpochSourceSlicer(self)
        self.loc = ItemGetter_loc(self)
        self.iloc = ItemGetter_iloc(self)

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        return "<base EventObject" + address_str + ">"

    @abstractmethod
    def partition(self, ds=None, n_epochs=None):
        """Returns an EventArray whose support has been partitioned.

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
        return

    @abstractmethod
    def isempty(self):
        """(bool) Empty EventArray."""
        return

    @abstractmethod
    def n_sources(self):
        """(int) The number of sources."""
        return

    @property
    def n_epochs(self):
        if self.isempty:
            return 0
        """(int) The number of underlying epochs."""
        return self.support.n_epochs

    @property
    def source_ids(self):
        """Source IDs contained in the EventArray."""
        return self._source_ids

    @source_ids.setter
    def source_ids(self, val):
        if len(val) != self.n_sources:
            raise TypeError("source_ids must be of length n_sources")
        elif len(set(val)) < len(val):
            raise TypeError("duplicate source_ids are not allowed")
        else:
            try:
                # cast to int:
                source_ids = [int(id) for id in val]
            except TypeError:
                raise TypeError("source_ids must be int-like")
        self._source_ids = source_ids

    @property
    def source_labels(self):
        """Labels corresponding to sources contained in the EventArray."""
        if self._source_labels is None:
            warnings.warn("source labels have not yet been specified")
        return self._source_labels

    @source_labels.setter
    def source_labels(self, val):
        if len(val) != self.n_sources:
            raise TypeError("labels must be of length n_sources")
        else:
            try:
                # cast to str:
                labels = [str(label) for label in val]
            except TypeError:
                raise TypeError("labels must be string-like")
        self._source_labels = labels

    @property
    def source_tags(self):
        """Tags corresponding to sources contained in the EventArray"""
        if self._source_tags is None:
            warnings.warn("source tags have not yet been specified")
        return self._source_tags

    @property
    def support(self):
        """(nelpy.EpochArray) The support of the underlying event train
        (in seconds).
         """
        return self._support

    @property
    def fs(self):
        """(float) Sampling rate / frequency (Hz)."""
        return self._fs

    @fs.setter
    def fs(self, val):
        """(float) Sampling rate / frequency (Hz)."""
        if self._fs == val:
            return
        try:
            if val <= 0:
                raise ValueError("sampling rate must be positive")
        except:
            raise TypeError("sampling rate must be a scalar")
        self._fs = val

    @property
    def label(self):
        """Label pertaining to the source of the event train."""
        if self._label is None:
            warnings.warn("label has not yet been specified")
        return self._label

    @label.setter
    def label(self, val):
        if val is not None:
            try:  # cast to str:
                label = str(val)
            except TypeError:
                raise TypeError("cannot convert label to string")
        else:
            label = val
        self._label = label

    def _source_subset(self, source_list):
        """Return an EventArray restricted to a subset of sources.

        Parameters
        ----------
        source_list : array-like
            Array or list of source_ids.
        """
        source_subset_ids = []
        for source in source_list:
            try:
                id = self.source_ids.index(source)
            except ValueError:
                warnings.warn("source_id " + str(source) + " not found in EventArray; ignoring")
                pass
            else:
                source_subset_ids.append(id)

        new_source_ids = (np.asarray(self.source_ids)[source_subset_ids]).tolist()
        new_source_labels = (np.asarray(self.source_labels)[source_subset_ids]).tolist()

        if isinstance(self, EventArray):
            if len(source_subset_ids) == 0:
                warnings.warn("no sources remaining in requested source subset")
                return EventArray(empty=True)

            eventtrainarray = EventArray(empty=True)
            exclude = ["_time", "source_ids", "source_labels"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("eventtrainarray." + attr + " = self." + attr)

            eventtrainarray._time = self.time[source_subset_ids]
            eventtrainarray._source_ids = new_source_ids
            eventtrainarray._source_labels = new_source_labels
            eventtrainarray.loc = ItemGetter_loc(eventtrainarray)
            eventtrainarray.iloc = ItemGetter_iloc(eventtrainarray)

            return eventtrainarray
        elif isinstance(self, BinnedEventArray):
            if len(source_subset_ids) == 0:
                warnings.warn("no sources remaining in requested source subset")
                return BinnedEventArray(empty=True)

            binnedeventtrainarray = BinnedEventArray(empty=True)
            exclude = ["_data", "source_ids", "source_labels"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("binnedeventtrainarray." + attr + " = self." + attr)

            binnedeventtrainarray._data = self.data[source_subset_ids,:]
            binnedeventtrainarray._source_ids = new_source_ids
            binnedeventtrainarray._source_labels = new_source_labels
            binnedeventtrainarray.loc = ItemGetter_loc(binnedeventtrainarray)
            binnedeventtrainarray.iloc = ItemGetter_iloc(binnedeventtrainarray)

            return binnedeventtrainarray
        else:
            raise NotImplementedError(
            "EventArray._source_slice() not supported for this type yet!")


########################################################################
# class EventArray
########################################################################
class EventArray(EventBase):
    """A multisource event train array with shared support.

    Parameters
    ----------
    timestamps : array of np.array(dtype=np.float64) event times in seconds.
        Array of length n_sources, each entry with shape (n_time,)
    fs : float, optional
        Sampling rate in Hz. Default is 30,000
    support : EpochArray, optional
        EpochArray on which eventtrains are defined.
        Default is [0, last event] inclusive.
    label : str or None, optional
        Information pertaining to the source of the eventtrain array.
    source_ids : list (of length n_sources) of indices corresponding to
        curated data. If no source_ids are specified, then [0,...,n_sources-1]
        will be used.
    meta : dict
        Metadata associated with EventArray.

    Attributes
    ----------
    time : array of np.array(dtype=np.float64) event times in seconds.
        Array of length n_sources, each entry with shape (n_time,)
    support : EpochArray on which EventArray is defined.
    n_events: np.array(dtype=np.int) of shape (n_sources,)
        Number of events in each source.
    fs: float
        Sampling frequency (Hz).
    label : str or None
        Information pertaining to the source of the eventtrain.
    meta : dict
        Metadata associated with eventtrain.
    """

    __attributes__ = ["_time", "_support"]
    __attributes__.extend(EventBase.__attributes__)
    def __init__(self, timestamps=None, *, fs=None, support=None,
                 source_ids=None, source_labels=None, source_tags=None,
                 label=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = core.EpochArray(empty=True)
            return

        # set default sampling rate
        if fs is None:
            fs = 30000
            warnings.warn("No sampling rate was specified! Assuming default of {} Hz.".format(fs))

        def is_singletons(data):
            """Returns True if data is a list of singletons (more than one)."""
            data = np.array(data)
            try:
                if data.shape[-1] < 2 and np.max(data.shape) > 1:
                    return True
                if max(np.array(data).shape[:-1]) > 1 and data.shape[-1] == 1:
                    return True
            except (IndexError, TypeError, ValueError):
                return False
            return False

        def is_single_source(data):
            """Returns True if data represents event times from a single source.

            Examples
            ========
            [1, 2, 3]           : True
            [[1, 2, 3]]         : True
            [[1, 2, 3], []]     : False
            [[], [], []]        : False
            [[[[1, 2, 3]]]]     : True
            [[[[[1],[2],[3]]]]] : False
            """
            try:
                if isinstance(data[0][0], list) or isinstance(data[0][0], np.ndarray):
                    warnings.warn("event times input has too many layers!")
                    if max(np.array(data).shape[:-1]) > 1:
        #                 singletons = True
                        return False
                    data = np.squeeze(data)
            except (IndexError, TypeError):
                pass
            try:
                if isinstance(data[1], list) or isinstance(data[1], np.ndarray):
                    return False
            except (IndexError, TypeError):
                pass
            return True

        def standardize_to_2d(data):
            if is_single_source(data):
                return np.array(np.squeeze(data), ndmin=2)
            if is_singletons(data):
                data = np.squeeze(data)
                n = np.max(data.shape)
                if len(data.shape) == 1:
                    m = 1
                else:
                    m = np.min(data.shape)
                data = np.reshape(data, (n,m))
            else:
                data = np.squeeze(data)
                if data.dtype == np.dtype('O'):
                    jagged = True
                else:
                    jagged = False
                if jagged:  # jagged array
                    # standardize input so that a list of lists is converted
                    # to an array of arrays:
                    data = np.array(
                        [np.array(st, ndmin=1, copy=False) for st in data])
                else:
                    data = np.array(data, ndmin=2)
            return data

        time = standardize_to_2d(timestamps)

        #sort event trains, but only if necessary:
        for ii, train in enumerate(time):
            if not utils.is_sorted(train):
                time[ii] = np.sort(train)

        kwargs = {"fs": fs,
                  "source_ids": source_ids,
                  "source_labels": source_labels,
                  "source_tags": source_tags,
                  "label": label}

        self._time = time  # this is necessary so that
        # super() can determine self.n_sources when initializing.

        # initialize super so that self.fs is set:
        super().__init__(**kwargs)

        # if only empty time were received AND no support, attach an
        # empty support:
        if np.sum([st.size for st in time]) == 0 and support is None:
            warnings.warn("no events; cannot automatically determine support")
            support = core.EpochArray(empty=True)

        # determine eventtrain array support:
        if support is None:
            first_event = np.nanmin(np.array([source[0] for source in time if len(source) !=0]))
            # BUG: if eventtrain is empty np.array([]) then source[-1]
            # raises an error in the following:
            # FIX: list[-1] raises an IndexError for an empty list,
            # whereas list[-1:] returns an empty list.
            last_event = np.nanmax(np.array([source[-1:] for source in time if len(source) !=0]))
            self._support = core.EpochArray(np.array([first_event, last_event + 1/fs]))
            # in the above, there's no reason to restrict to support
        else:
            # restrict events to only those within the eventtrain
            # array's support:
            self._support = support

        # TODO: if sorted, we may as well use the fast restrict here as well?
        time = self._restrict_to_epoch_array(
            epocharray=self._support,
            time=time)

        self._time = time

    def partition(self, ds=None, n_epochs=None):
        """Returns an EventArray whose support has been partitioned.

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
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        #TODO: renew epoch slicers !
        return out

    def copy(self):
        """Returns a copy of the EventArray."""
        newcopy = copy.deepcopy(self)
        newcopy.loc = ItemGetter_loc(newcopy)
        newcopy.iloc = ItemGetter_iloc(newcopy)
        #TODO: renew epoch slicers !
        return newcopy

    def __add__(self, other):
        """Overloaded + operator"""

        #TODO: additional checks need to be done, e.g., same source ids...
        #TODO: it's better to copy into self, so that metadata are preserved
        assert self.n_sources == other.n_sources
        support = self.support + other.support

        newdata = []
        for source in range(self.n_sources):
            newdata.append(np.append(self.time[source], other.time[source]))

        fs = self.fs
        if self.fs != other.fs:
            fs = None
        return EventArray(newdata, support=support, fs=fs)

    def __iter__(self):
        """EventArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """EventArray iterator advancer."""
        index = self._index
        if index > self.support.n_epochs - 1:
            raise StopIteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            support = self.support[index]
            time = self._restrict_to_epoch_array_fast(
                epocharray=support,
                time=self.time,
                copyover=True
                )
            eventtrain = EventArray(empty=True)
            exclude = ["_time", "_support"]
            attrs = (x for x in self.__attributes__ if x not in exclude)
            for attr in attrs:
                exec("eventtrain." + attr + " = self." + attr)
            eventtrain._time = time
            eventtrain._support = support
            eventtrain.loc = ItemGetter_loc(eventtrain)
            eventtrain.iloc = ItemGetter_iloc(eventtrain)
        self._index += 1
        return eventtrain

    def _epochslicer(self, idx):
        """Helper function to restrict object to EpochArray."""
        # if self.isempty:
        #     return self

        if isinstance(idx, core.EpochArray):
            if idx.isempty:
                return EventArray(empty=True)
            support = self.support.intersect(
                    epoch=idx,
                    boundaries=True
                    ) # what if fs of slicing epoch is different?
            if support.isempty:
                return EventArray(empty=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                time = self._restrict_to_epoch_array_fast(
                    epocharray=support,
                    time=self.time,
                    copyover=True
                    )
                eventtrain = EventArray(empty=True)
                exclude = ["_time", "_support"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("eventtrain." + attr + " = self." + attr)
                eventtrain._time = time
                eventtrain._support = support
                eventtrain.loc = ItemGetter_loc(eventtrain)
                eventtrain.iloc = ItemGetter_iloc(eventtrain)
            return eventtrain
        elif isinstance(idx, int):
            eventtrain = EventArray(empty=True)
            exclude = ["_time", "_support"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("eventtrain." + attr + " = self." + attr)
                support = self.support[idx]
                eventtrain._support = support
            if (idx >= self.support.n_epochs) or idx < (-self.support.n_epochs):
                eventtrain.loc = ItemGetter_loc(eventtrain)
                eventtrain.iloc = ItemGetter_iloc(eventtrain)
                return eventtrain
            else:
                time = self._restrict_to_epoch_array_fast(
                        epocharray=support,
                        time=self.time,
                        copyover=True
                        )
                eventtrain._time = time
                eventtrain._support = support
                eventtrain.loc = ItemGetter_loc(eventtrain)
                eventtrain.iloc = ItemGetter_iloc(eventtrain)
                return eventtrain
        else:  # most likely slice indexing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    support = self.support[idx]
                    time = self._restrict_to_epoch_array_fast(
                        epocharray=support,
                        time=self.time,
                        copyover=True
                        )
                    eventtrain = EventArray(empty=True)
                    exclude = ["_time", "_support"]
                    attrs = (x for x in self.__attributes__ if x not in exclude)
                    for attr in attrs:
                        exec("eventtrain." + attr + " = self." + attr)
                    eventtrain._time = time
                    eventtrain._support = support
                    eventtrain.loc = ItemGetter_loc(eventtrain)
                    eventtrain.iloc = ItemGetter_iloc(eventtrain)
                return eventtrain
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))


    def __getitem__(self, idx):
        """EventArray index access.

        By default, this method is bound to EventArray.loc
        """
        return self.loc[idx]

    @property
    def isempty(self):
        """(bool) Empty EventArray."""
        try:
            return np.sum([len(st) for st in self.time]) == 0
        except TypeError:
            return True  # this happens when self.time == None

    @property
    def n_sources(self):
        """(int) The number of sources."""
        try:
            return utils.PrettyInt(len(self.time))
        except TypeError:
            return 0

    @property
    def n_active(self):
        """(int) The number of active sources.

        A source is considered active if it fired at least one event.
        """
        if self.isempty:
            return 0
        return utils.PrettyInt(np.count_nonzero(self.n_events))

    def _copy_without_data(self):
        """Return a copy of self, without event times."""
        out = copy.copy(self) # shallow copy
        out._time = None
        out = copy.deepcopy(self) # just to be on the safe side, but at least now we are not copying the data!

        return out

    def flatten(self, *, source_id=None, source_label=None):
        """Collapse events across all sources.

        WARNING! source_tags are thrown away when flattening.

        Parameters
        ----------
        source_id: (int)
            (source) ID to assign to flattened event train, default is 0.
        source_label (str)
            (source) Label for event train, default is 'flattened'.
        """
        if self.n_sources < 2:  # already flattened
            return self

        # default args:
        if source_id is None:
            source_id = 0
        if source_label is None:
            source_label = "flattened"

        flattened = self._copy_without_data()

        flattened._source_ids = [source_id]
        flattened._source_labels = [source_label]
        flattened._source_tags = None

        alltimes = self.time[0]
        for source in range(1,self.n_sources):
            alltimes = utils.linear_merge(alltimes, self.time[source])

        flattened._time = np.array(list(alltimes), ndmin=2)
        flattened.loc = ItemGetter_loc(flattened)
        flattened.iloc = ItemGetter_iloc(flattened)
        return flattened

    @staticmethod
    def _restrict_to_epoch_array_fast(epocharray, time, copyover=True):
        """Return time restricted to an EpochArray.

        This function assumes sorted event times, so that binary search can
        be used to quickly identify slices that should be kept in the
        restriction. It does not check every event time.

        Parameters
        ----------
        epocharray : EpochArray
        time : array-like
        """
        if epocharray.isempty:
            n_sources = len(time)
            time = np.zeros((n_sources,0))
            return time

        singlesource = len(time)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            time = copy.copy(time)

        # NOTE: this used to assume multiple sources for the enumeration to work
        for source, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                frm, to = np.searchsorted(st_time, (t_start, t_stop))
                indices.append((frm, to))
            indices = np.array(indices, ndmin=2)
            if np.diff(indices).sum() < len(st_time):
                warnings.warn(
                    'ignoring events outside of eventtrain support')
            if singlesource:
                time_list = []
                for start, stop in indices:
                    time_list.extend(st_time[start:stop])
                time = np.array(time_list, ndmin=2)
            else:
                # here we have to do some annoying conversion between
                # arrays and lists to fully support jagged array
                # mutation
                time_list = []
                for start, stop in indices:
                    time_list.extend(st_time[start:stop])
                time_ = time.tolist()
                time_[source] = np.array(time_list)
                time = np.array(time_)
        return time

    @staticmethod
    def _restrict_to_epoch_array(epocharray, time, copyover=True):
        """Return time restricted to an EpochArray.

        This function is quite slow, as it checks each event time for inclusion.
        It does this in a vectorized form, which is fast for small or moderately
        sized objects, but the memory penalty can be large, and it becomes very
        slow for large objects. Consequently, _restrict_to_epoch_array_fast
        should be used when possible.

        Parameters
        ----------
        epocharray : EpochArray
        time : array-like
        """
        if epocharray.isempty:
            n_sources = len(time)
            time = np.zeros((n_sources,0))
            return time

        singlesource = len(time)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            time = copy.copy(time)

        # NOTE: this used to assume multiple sources for the enumeration to work
        for source, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                indices.append((st_time >= t_start) & (st_time < t_stop))
            indices = np.any(np.column_stack(indices), axis=1)
            if np.count_nonzero(indices) < len(st_time):
                warnings.warn(
                    'ignoring events outside of eventtrain support')
            if singlesource:
                time = np.array([time[0][indices]], ndmin=2)
            else:
                # here we have to do some annoying conversion between
                # arrays and lists to fully support jagged array
                # mutation
                time_ = time.tolist()
                time_[source] = np.array(time_[source])
                time_[source] = time_[source][indices]
                time = np.array(time_)
        return time

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.isempty:
                return "<empty EventArray" + address_str + ">"
            if self.support.n_epochs > 1:
                epstr = " ({} segments)".format(self.support.n_epochs)
            else:
                epstr = ""
            if self.fs is not None:
                fsstr = " at %s Hz" % self.fs
            else:
                fsstr = ""
            if self.label is not None:
                labelstr = " from %s" % self.label
            else:
                labelstr = ""
            numstr = " %s sources" % self.n_sources
        return "<EventArray%s:%s%s>%s%s" % (address_str, numstr, epstr, fsstr, labelstr)

    def bin(self, *, ds=None):
        """Return a binned eventtrain array."""
        return BinnedEventArray(self, ds=ds)

    @property
    def time(self):
        """Event times in seconds."""
        return self._time

    @property
    def n_events(self):
        """(np.array) The number of events in each source."""
        if self.isempty:
            return 0
        return np.array([len(source) for source in self.time])

    @property
    def issorted(self):
        """(bool) Sorted EventArray."""
        if self.isempty:
            return True
        return np.array(
            [utils.is_sorted(eventtrain) for eventtrain in self.time]
            ).all()

    def _reorder_sources_by_idx(self, neworder, inplace=False):
        """Reorder sources according to a specified order.

        neworder must be list-like, of size (n_sources,)

        Return
        ------
        out : reordered EventArray
        """

        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._time, frm, to)
            out._source_ids[frm], out._source_ids[to] = out._source_ids[to], out._source_ids[frm]
            out._source_labels[frm], out._source_labels[to] = out._source_labels[to], out._source_labels[frm]
            # TODO: re-build source tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

    def reorder_sources(self, neworder, *, inplace=False):
        """Reorder sources according to a specified order.

        neworder must be list-like, of size (n_sources,) and in terms of
        source_ids

        Return
        ------
        out : reordered EventArray
        """
        raise DeprecationWarning("reorder_sources has been deprecated. Use reorder_sources_by_id(x/s) instead!")

    def reorder_sources_by_ids(self, neworder, *, inplace=False):
        """Reorder sources according to a specified order.

        neworder must be list-like, of size (n_sources,) and in terms of
        source_ids

        Return
        ------
        out : reordered EventArray
        """
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        neworder = [self.source_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._time, frm, to)
            out._source_ids[frm], out._source_ids[to] = out._source_ids[to], out._source_ids[frm]
            out._source_labels[frm], out._source_labels[to] = out._source_labels[to], out._source_labels[frm]
            # TODO: re-build source tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

    def get_event_firing_order(self):
        """Returns a list of source_ids such that the sources are ordered
        by when they first fire in the EventArray.

        Return
        ------
        firing_order : list of source_ids
        """

        first_events = [(ii, source[0]) for (ii, source) in enumerate(self.time) if len(source) !=0]
        first_events_source_ids = np.array(self.source_ids)[[fs[0] for fs in first_events]]
        first_events_times = np.array([fs[1] for fs in first_events])
        sortorder = np.argsort(first_events_times)
        first_events_source_ids = first_events_source_ids[sortorder]
        remaining_ids = list(set(self.source_ids) - set(first_events_source_ids))
        firing_order = list(first_events_source_ids)
        firing_order.extend(remaining_ids)

        return firing_order

#----------------------------------------------------------------------#
#======================================================================#

########################################################################
# class ValueEventArray
########################################################################
class ValueEventArray(EventBase):
    """A multisource event train array with a value associated with each event.

    Parameters
    ----------
    timestamps : array of np.array(dtype=np.float64) event times in seconds.
        Array of length n_sources, each entry with shape (n_times,)
    eventvalues : array of event values.
        Array of length n_sources, each entry with shape (n_times,)
    fs : float, optional
        Sampling rate in Hz. Default is 30,000
    support : EpochArray, optional
        EpochArray on which eventtrains are defined.
        Default is [0, last event] inclusive.
    label : str or None, optional
        Information pertaining to the source of the eventtrain array.
    source_ids : list (of length n_sources) of indices corresponding to
        curated data. If no source_ids are specified, then [0,...,n_sources-1]
        will be used.
    meta : dict
        Metadata associated with EventArray.

    Attributes
    ----------
    time : array of np.array(dtype=np.float64) event times in seconds.
        Array of length n_sources, each entry with shape (n_time,)
    values : array of event values.
        Array of length n_sources, each entry with shape (n_time,)
    support : EpochArray on which EventArray is defined.
    n_events: np.array(dtype=np.int) of shape (n_sources,)
        Number of events in each source.
    fs: float
        Sampling frequency (Hz).
    label : str or None
        Information pertaining to the source of the eventtrain.
    meta : dict
        Metadata associated with eventtrain.
    """

    __attributes__ = ["_time", "_values", "_support"]
    __attributes__.extend(EventBase.__attributes__)
    def __init__(self, timestamps=None, *, eventvalues=None, fs=None, support=None,
                 source_ids=None, source_labels=None, source_tags=None,
                 label=None, empty=False):

        default_val = 0; # default event value (not yet exposed by API)

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = core.EpochArray(empty=True)
            return

        # set default sampling rate
        if fs is None:
            fs = 30000
            warnings.warn("No sampling rate was specified! Assuming default of {} Hz.".format(fs))

        def is_singletons(data):
            """Returns True if data is a list of singletons (more than one)."""
            data = np.array(data)
            try:
                if data.shape[-1] < 2 and np.max(data.shape) > 1:
                    return True
                if max(np.array(data).shape[:-1]) > 1 and data.shape[-1] == 1:
                    return True
            except (IndexError, TypeError, ValueError):
                return False
            return False

        def is_single_source(data):
            """Returns True if data represents event times from a single source.

            Examples
            ========
            [1, 2, 3]           : True
            [[1, 2, 3]]         : True
            [[1, 2, 3], []]     : False
            [[], [], []]        : False
            [[[[1, 2, 3]]]]     : True
            [[[[[1],[2],[3]]]]] : False
            """
            try:
                if isinstance(data[0][0], list) or isinstance(data[0][0], np.ndarray):
                    warnings.warn("event times input has too many layers!")
                    if max(np.array(data).shape[:-1]) > 1:
        #                 singletons = True
                        return False
                    data = np.squeeze(data)
            except (IndexError, TypeError):
                pass
            try:
                if isinstance(data[1], list) or isinstance(data[1], np.ndarray):
                    return False
            except (IndexError, TypeError):
                pass
            return True

        def standardize_to_2d(data):
            if is_single_source(data):
                return np.array(np.squeeze(data), ndmin=2)
            if is_singletons(data):
                data = np.squeeze(data)
                n = np.max(data.shape)
                if len(data.shape) == 1:
                    m = 1
                else:
                    m = np.min(data.shape)
                data = np.reshape(data, (n,m))
            else:
                data = np.squeeze(data)
                if data.dtype == np.dtype('O'):
                    jagged = True
                else:
                    jagged = False
                if jagged:  # jagged array
                    # standardize input so that a list of lists is converted
                    # to an array of arrays:
                    data = np.array(
                        [np.array(st, ndmin=1, copy=False) for st in data])
                else:
                    data = np.array(data, ndmin=2)
            return data

        time = standardize_to_2d(timestamps)

        if eventvalues is not None:
            values = standardize_to_2d(eventvalues)
            if values.shape != time.shape:
                raise ValueError('timestamps and eventvalues must have the same size!')
        else:
            values = np.ones(time.shape)*default_val

        #sort event trains, but only if necessary:
        for ii, train in enumerate(time):
            if not utils.is_sorted(train):
                time[ii] = np.sort(train)

        kwargs = {"fs": fs,
                  "source_ids": source_ids,
                  "source_labels": source_labels,
                  "source_tags": source_tags,
                  "label": label}

        self._time = time  # this is necessary so that
        # super() can determine self.n_sources when initializing.
        self._values = values

        # initialize super so that self.fs is set:
        super().__init__(**kwargs)

        # if only empty time were received AND no support, attach an
        # empty support:
        if np.sum([st.size for st in time]) == 0 and support is None:
            warnings.warn("no events; cannot automatically determine support")
            support = core.EpochArray(empty=True)

        # determine eventtrain array support:
        if support is None:
            first_event = np.nanmin(np.array([source[0] for source in time if len(source) !=0]))
            # BUG: if eventtrain is empty np.array([]) then source[-1]
            # raises an error in the following:
            # FIX: list[-1] raises an IndexError for an empty list,
            # whereas list[-1:] returns an empty list.
            last_event = np.nanmax(np.array([source[-1:] for source in time if len(source) !=0]))
            self._support = core.EpochArray(np.array([first_event, last_event + 1/fs]))
            # in the above, there's no reason to restrict to support
        else:
            # restrict events to only those within the eventtrain
            # array's support:
            self._support = support

        # TODO: if sorted, we may as well use the fast restrict here as well?
        time, values = self._restrict_to_epoch_array(
            epocharray=self._support,
            time=time, values=values)

        self._time = time
        self._values = values

    def partition(self, ds=None, n_epochs=None):
        """Returns an EventArray whose support has been partitioned.

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
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        #TODO: renew epoch slicers !
        return out

    def copy(self):
        """Returns a copy of the EventArray."""
        newcopy = copy.deepcopy(self)
        newcopy.loc = ItemGetter_loc(newcopy)
        newcopy.iloc = ItemGetter_iloc(newcopy)
        #TODO: renew epoch slicers !
        return newcopy

    def __add__(self, other):
        """Overloaded + operator"""

        raise NotImplementedError
        #TODO: additional checks need to be done, e.g., same source ids...
        #TODO: it's better to copy into self, so that metadata are preserved
        assert self.n_sources == other.n_sources
        support = self.support + other.support

        newdata = []
        for source in range(self.n_sources):
            newdata.append(np.append(self.time[source], other.time[source]))

        fs = self.fs
        if self.fs != other.fs:
            fs = None
        return EventArray(newdata, support=support, fs=fs)

    def __iter__(self):
        """EventArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """EventArray iterator advancer."""
        raise NotImplementedError
        index = self._index
        if index > self.support.n_epochs - 1:
            raise StopIteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            support = self.support[index]
            time = self._restrict_to_epoch_array_fast(
                epocharray=support,
                time=self.time,
                copyover=True
                )
            eventtrain = EventArray(empty=True)
            exclude = ["_time", "_support"]
            attrs = (x for x in self.__attributes__ if x not in exclude)
            for attr in attrs:
                exec("eventtrain." + attr + " = self." + attr)
            eventtrain._time = time
            eventtrain._support = support
            eventtrain.loc = ItemGetter_loc(eventtrain)
            eventtrain.iloc = ItemGetter_iloc(eventtrain)
        self._index += 1
        return eventtrain

    def _epochslicer(self, idx):
        """Helper function to restrict object to EpochArray."""
        # if self.isempty:
        #     return self
        raise NotImplementedError

        if isinstance(idx, core.EpochArray):
            if idx.isempty:
                return EventArray(empty=True)
            support = self.support.intersect(
                    epoch=idx,
                    boundaries=True
                    ) # what if fs of slicing epoch is different?
            if support.isempty:
                return EventArray(empty=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                time = self._restrict_to_epoch_array_fast(
                    epocharray=support,
                    time=self.time,
                    copyover=True
                    )
                eventtrain = EventArray(empty=True)
                exclude = ["_time", "_support"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("eventtrain." + attr + " = self." + attr)
                eventtrain._time = time
                eventtrain._support = support
                eventtrain.loc = ItemGetter_loc(eventtrain)
                eventtrain.iloc = ItemGetter_iloc(eventtrain)
            return eventtrain
        elif isinstance(idx, int):
            eventtrain = EventArray(empty=True)
            exclude = ["_time", "_support"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("eventtrain." + attr + " = self." + attr)
                support = self.support[idx]
                eventtrain._support = support
            if (idx >= self.support.n_epochs) or idx < (-self.support.n_epochs):
                eventtrain.loc = ItemGetter_loc(eventtrain)
                eventtrain.iloc = ItemGetter_iloc(eventtrain)
                return eventtrain
            else:
                time = self._restrict_to_epoch_array_fast(
                        epocharray=support,
                        time=self.time,
                        copyover=True
                        )
                eventtrain._time = time
                eventtrain._support = support
                eventtrain.loc = ItemGetter_loc(eventtrain)
                eventtrain.iloc = ItemGetter_iloc(eventtrain)
                return eventtrain
        else:  # most likely slice indexing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    support = self.support[idx]
                    time = self._restrict_to_epoch_array_fast(
                        epocharray=support,
                        time=self.time,
                        copyover=True
                        )
                    eventtrain = EventArray(empty=True)
                    exclude = ["_time", "_support"]
                    attrs = (x for x in self.__attributes__ if x not in exclude)
                    for attr in attrs:
                        exec("eventtrain." + attr + " = self." + attr)
                    eventtrain._time = time
                    eventtrain._support = support
                    eventtrain.loc = ItemGetter_loc(eventtrain)
                    eventtrain.iloc = ItemGetter_iloc(eventtrain)
                return eventtrain
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))


    def __getitem__(self, idx):
        """EventArray index access.

        By default, this method is bound to EventArray.loc
        """
        return self.loc[idx]

    @property
    def isempty(self):
        """(bool) Empty EventArray."""
        try:
            return np.sum([len(evt) for evt in self.time]) == 0
        except TypeError:
            return True  # this happens when self.time == None

    @property
    def n_sources(self):
        """(int) The number of sources."""
        try:
            return utils.PrettyInt(len(self.time))
        except TypeError:
            return 0

    @property
    def n_active(self):
        """(int) The number of active sources.

        A source is considered active if it fired at least one event.
        """
        if self.isempty:
            return 0
        return utils.PrettyInt(np.count_nonzero(self.n_events))

    def _copy_without_data(self):
        """Return a copy of self, without event times."""
        raise NotImplementedError
        out = copy.copy(self) # shallow copy
        out._time = None
        out = copy.deepcopy(self) # just to be on the safe side, but at least now we are not copying the data!

        return out

    @staticmethod
    def _restrict_to_epoch_array_fast(epocharray, time, value, copyover=True):
        """Return time and values restricted to an EpochArray.

        This function assumes sorted event times, so that binary search can
        be used to quickly identify slices that should be kept in the
        restriction. It does not check every event time.

        Parameters
        ----------
        epocharray : EpochArray
        time : array-like
        values : array-like
        """
        raise NotImplementedError
        if epocharray.isempty:
            n_sources = len(time)
            time = np.zeros((n_sources,0))
            return time

        singlesource = len(time)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            time = copy.copy(time)

        # NOTE: this used to assume multiple sources for the enumeration to work
        for source, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                frm, to = np.searchsorted(st_time, (t_start, t_stop))
                indices.append((frm, to))
            indices = np.array(indices, ndmin=2)
            if np.diff(indices).sum() < len(st_time):
                warnings.warn(
                    'ignoring events outside of eventtrain support')
            if singlesource:
                time_list = []
                for start, stop in indices:
                    time_list.extend(st_time[start:stop])
                time = np.array(time_list, ndmin=2)
            else:
                # here we have to do some annoying conversion between
                # arrays and lists to fully support jagged array
                # mutation
                time_list = []
                for start, stop in indices:
                    time_list.extend(st_time[start:stop])
                time_ = time.tolist()
                time_[source] = np.array(time_list)
                time = np.array(time_)
        return time

    @staticmethod
    def _restrict_to_epoch_array(epocharray, time, copyover=True):
        """Return time restricted to an EpochArray.

        This function is quite slow, as it checks each event time for inclusion.
        It does this in a vectorized form, which is fast for small or moderately
        sized objects, but the memory penalty can be large, and it becomes very
        slow for large objects. Consequently, _restrict_to_epoch_array_fast
        should be used when possible.

        Parameters
        ----------
        epocharray : EpochArray
        time : array-like
        """
        raise NotImplementedError
        if epocharray.isempty:
            n_sources = len(time)
            time = np.zeros((n_sources,0))
            return time

        singlesource = len(time)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            time = copy.copy(time)

        # NOTE: this used to assume multiple sources for the enumeration to work
        for source, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                indices.append((st_time >= t_start) & (st_time < t_stop))
            indices = np.any(np.column_stack(indices), axis=1)
            if np.count_nonzero(indices) < len(st_time):
                warnings.warn(
                    'ignoring events outside of eventtrain support')
            if singlesource:
                time = np.array([time[0][indices]], ndmin=2)
            else:
                # here we have to do some annoying conversion between
                # arrays and lists to fully support jagged array
                # mutation
                time_ = time.tolist()
                time_[source] = np.array(time_[source])
                time_[source] = time_[source][indices]
                time = np.array(time_)
        return time

    def __repr__(self):
        raise NotImplementedError
        address_str = " at " + str(hex(id(self)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.isempty:
                return "<empty EventArray" + address_str + ">"
            if self.support.n_epochs > 1:
                epstr = " ({} segments)".format(self.support.n_epochs)
            else:
                epstr = ""
            if self.fs is not None:
                fsstr = " at %s Hz" % self.fs
            else:
                fsstr = ""
            if self.label is not None:
                labelstr = " from %s" % self.label
            else:
                labelstr = ""
            numstr = " %s sources" % self.n_sources
        return "<EventArray%s:%s%s>%s%s" % (address_str, numstr, epstr, fsstr, labelstr)

    @property
    def time(self):
        """Event times in seconds."""
        return self._time

    @property
    def n_events(self):
        """(np.array) The number of events in each source."""
        if self.isempty:
            return 0
        return np.array([len(source) for source in self.time])

    @property
    def issorted(self):
        """(bool) Sorted EventArray."""
        if self.isempty:
            return True
        return np.array(
            [utils.is_sorted(eventtrain) for eventtrain in self.time]
            ).all()

    def _reorder_sources_by_idx(self, neworder, inplace=False):
        """Reorder sources according to a specified order.

        neworder must be list-like, of size (n_sources,)

        Return
        ------
        out : reordered EventArray
        """
        raise NotImplementedError

        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._time, frm, to)
            out._source_ids[frm], out._source_ids[to] = out._source_ids[to], out._source_ids[frm]
            out._source_labels[frm], out._source_labels[to] = out._source_labels[to], out._source_labels[frm]
            # TODO: re-build source tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

    def reorder_sources_by_ids(self, neworder, *, inplace=False):
        """Reorder sources according to a specified order.

        neworder must be list-like, of size (n_sources,) and in terms of
        source_ids

        Return
        ------
        out : reordered EventArray
        """
        raise NotImplementedError
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        neworder = [self.source_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._time, frm, to)
            out._source_ids[frm], out._source_ids[to] = out._source_ids[to], out._source_ids[frm]
            out._source_labels[frm], out._source_labels[to] = out._source_labels[to], out._source_labels[frm]
            # TODO: re-build source tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

#----------------------------------------------------------------------#
#======================================================================#
