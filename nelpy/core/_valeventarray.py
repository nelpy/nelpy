# note:
# veva can be binned, and binning can be accommodative, or accumulative
# accomodative is averaging/interp while accumulative is mass-preserving

# __all__ = ['EventArray',
#            'BinnedEventArray',
#            'SpikeTrainArray',
#            'BinnedSpikeTrainArray']

__all__ = ['ValueEventArray']

# __all__ = ['BaseValueEventArray(ABC)',
#            'ValueEventArray(BaseValueEventArray)',
#            'MarkedSpikeTrainArray(ValueEventArray)',
#            'BinnedValueEventArray(ValueEventArray)',
#            'BinnedMarkedSpikeTrainArray(MarkedSpikeTrainArray)']

# __all__ = ['BaseStatefulEventArray(ABC)'
#            'StatefulEventArray(BaseStatefulEventArray)'
#            'BinnedStatefulEventArray(???)']

"""
Notes
-----

non-binned EVAs have multiple timeseries as abscissa
binned EVAs only have a single timeseries as abscissa

ISASA = Irregularly Sampled AnalogSignalArray

eva - not callable
    - fully binnable --> beva

beva - callable --> veva (really ISASA)
     - binnable
     - castable to ASA and back

veva - not callable
     - somewhat binnable --> bveva
     - make stateful --> sveva
     - .data; .values

bveva - callable --> veva (not exactly an ISASA anymore, due to each signal being ND)
      - castable to beva (2d matrix, instead of list of 2d matrices); not complete!

sveva - callable --> veva
      - somewhat binnable --> bveva
      - .data; .values; .states

Remarks: I planned on having a list of arrays, and a list of timeseries for the
         ValueEventArrays.

ValueEventArrays have data structure
    nSeries : nEvents x nValues [(i, j, k) --> i: j=f(i), k]
BinnedValueEventArrays have data structure
    nSeries x nBins x nValues [(i, j, k) --> i x j x k]
and each of these structures are wrapped in intervals[data] pseudo encapsulation

BaseValueEventArray (ABC)
  --- veva (base)
  --- bveva (base)
  --- sveva (veva)

"""

import warnings
import logging
import numpy as np
import copy
import numbers

from abc import ABC, abstractmethod

from .. import core
from .. import utils
from .. import version

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

class IntervalSeriesSlicer(object):
    def __init__(self):
        pass

    def __getitem__(self, *args):
        """intervals (e.g. epochs), series (e.g. units)"""
        # by default, keep all series
        seriesslice = slice(None, None, None)
        if isinstance(*args, int):
            intervalslice = args[0]
        elif isinstance(*args, core.IntervalArray):
            intervalslice = args[0]
        else:
            try:
                slices = np.s_[args]; slices = slices[0]
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

class ItemGetter_loc(object):
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
        intervalslice, seriesslice = IntervalSeriesSlicer()[idx]

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
                raise KeyError('series_id {} could not be found in BaseEventArray!'.format(start))
            try:
                if stop is None:
                    istop = self.obj.n_series
                else:
                    istop = self.obj._series_ids.index(stop) + 1
            except ValueError:
                raise KeyError('series_id {} could not be found in BaseEventArray!'.format(stop))
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
                    raise KeyError("series_id {} could not be found in BaseEventArray!".format(series))
                else:
                    series_idx_list.append(uidx)

        if not isinstance(series_idx_list, list):
            series_idx_list = list(series_idx_list)

        out = copy.copy(self.obj)
        try:
            out._data = out._data[[series_idx_list]]
            singleseries = len(out._data)==1
        except AttributeError:
            out._data = out._data[[series_idx_list]]
            singleseries = len(out._data)==1

        if singleseries:
            out._data = np.array([out._data[0]], ndmin=2)
        out._series_ids = list(np.atleast_1d(np.atleast_1d(out._series_ids)[[series_idx_list]]))
        # TODO: update tags
        if isinstance(intervalslice, slice):
            if intervalslice.start == None and intervalslice.stop == None and intervalslice.step == None:
                out.loc = ItemGetter_loc(out)
                out.iloc = ItemGetter_iloc(out)
                return out
        out = out._intervalslicer(intervalslice)
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
        """intervals, series"""
        intervalslice, seriesslice = IntervalSeriesSlicer()[idx]
        out = copy.copy(self.obj)
        if isinstance(seriesslice, int):
            seriesslice = [seriesslice]
        out._data = out._data[[seriesslice]]
        singleseries = len(out._data)==1
        if singleseries:
            out._data = np.array(out._data[[0]], ndmin=2)
        out._series_ids = list(np.atleast_1d(np.atleast_1d(out._series_ids)[seriesslice]))

        # TODO: update tags
        if isinstance(intervalslice, slice):
            if intervalslice.start == None and intervalslice.stop == None and intervalslice.step == None:
                out.loc = ItemGetter_loc(out)
                out.iloc = ItemGetter_iloc(out)
                return out
        out = out._intervalslicer(intervalslice)
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

########################################################################
# class BaseValueEventArray
########################################################################
class BaseValueEventArray(ABC):
    """Base class for ValueEventArray and BinnedValueEventArray.

    """

    __aliases__ = {}

    __attributes__ = ["_fs", "_series_ids"]

    def __init__(self, *, fs=None, series_ids=None, empty=False, abscissa=None, ordinate=None):

        self.__version__ = version.__version__
        self.type_name = self.__class__.__name__
        if abscissa is None:
            abscissa = core.Abscissa() #TODO: integrate into constructor?
        if ordinate is None:
            ordinate = core.Ordinate() #TODO: integrate into constructor?
        self._abscissa = abscissa
        self._ordinate = ordinate

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._abscissa.support = type(self._abscissa.support)(empty=True)
            self.loc = ItemGetter_loc(self)
            self.iloc = ItemGetter_iloc(self)
            return

        # set initial fs to None
        self._fs = None
        # then attempt to update the fs; this does input validation:
        self.fs = fs

        # WARNING! we need to ensure that self.n_series can work BEFORE
        # we can set self.series_ids or self.series_labels, since those
        # setters check that the lengths of the inputs are consistent
        # with self.n_series.

        # inherit series IDs if available, otherwise initialize to default
        if series_ids is None:
            series_ids = list(range(1,self.n_series + 1))

        series_ids = np.array(series_ids, ndmin=1)  # standardize series_ids

        self.series_ids = series_ids

        self.loc = ItemGetter_loc(self)
        self.iloc = ItemGetter_iloc(self)

    def __renew__(self):
        """Re-attach slicers and indexers."""
        self.loc = ItemGetter_loc(self)
        self.iloc = ItemGetter_iloc(self)

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        return "<BaseValueEventArray" + address_str + ">"

    @abstractmethod
    def partition(self, ds=None, n_intervals=None):
        """Returns a BaseEventArray whose support has been partitioned.

        # Irrespective of whether 'ds' or 'n_intervals' are used, the exact
        # underlying support is propagated, and the first and last points
        # of the supports are always included, even if this would cause
        # n_points or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Maximum duration (in seconds), for each interval.
        n_points : int, optional
            Number of intervals. If ds is None and n_intervals is None, then
            default is to use n_intervals = 100

        Returns
        -------
        out : BaseEventArray
            BaseEventArray that has been partitioned.
        """
        return

    @abstractmethod
    def isempty(self):
        """(bool) Empty BaseEventArray."""
        return

    @abstractmethod
    def n_series(self):
        """(int) The number of series."""
        return

    @abstractmethod
    def n_values(self):
        """(int) The number of values associated with each event series."""
        return

    @property
    def n_intervals(self):
        if self.isempty:
            return 0
        """(int) The number of underlying intervals."""
        return self._abscissa.support.n_intervals

    @property
    def series_ids(self):
        """Unit IDs contained in the BaseEventArray."""
        return self._series_ids

    @series_ids.setter
    def series_ids(self, val):
        if len(val) != self.n_series:
            raise TypeError("series_ids must be of length n_series")
        elif len(set(val)) < len(val):
            raise TypeError("duplicate series_ids are not allowed")
        else:
            try:
                # cast to int:
                series_ids = [int(id) for id in val]
            except TypeError:
                raise TypeError("series_ids must be int-like")
        self._series_ids = series_ids

    @property
    def support(self):
        """(nelpy.IntervalArray) The support of the underlying EventArray."""
        return self._abscissa.support

    @support.setter
    def support(self, val):
        """(nelpy.IntervalArray) The support of the underlying EventArray."""
        # modify support
        if isinstance(val, type(self._abscissa.support)):
            self._abscissa.support = val
        elif isinstance(val, (tuple, list)):
            prev_domain = self._abscissa.domain
            self._abscissa.support = type(self._abscissa.support)([val[0], val[1]])
            self._abscissa.domain = prev_domain
        else:
            raise TypeError('support must be of type {}'.format(str(type(self._abscissa.support))))
        # restrict data to new support
        self._data = self._restrict_to_interval_array_fast(
                intervalarray=self._abscissa.support,
                data=self.data,
                copyover=True
                )

    @property
    def domain(self):
        """(nelpy.IntervalArray) The domain of the underlying EventArray."""
        return self._abscissa.domain

    @domain.setter
    def domain(self, val):
        """(nelpy.IntervalArray) The domain of the underlying EventArray."""
        # modify domain
        if isinstance(val, type(self._abscissa.support)):
            self._abscissa.domain = val
        elif isinstance(val, (tuple, list)):
            self._abscissa.domain = type(self._abscissa.support)([val[0], val[1]])
        else:
            raise TypeError('support must be of type {}'.format(str(type(self._abscissa.support))))
        # restrict data to new support
        self._data = self._restrict_to_interval_array_fast(
                intervalarray=self._abscissa.support,
                data=self.data,
                copyover=True
                )

    @property
    def fs(self):
        """(float) Sampling rate."""
        return self._fs

    @fs.setter
    def fs(self, val):
        """(float) Sampling rate."""
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
        """Label pertaining to the source of the event series."""
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

    def _series_subset(self, series_list):
        """Return a BaseEventArray restricted to a subset of series.

        Parameters
        ----------
        series_list : array-like
            Array or list of series_ids.
        """
        return self.loc[:,series_list]

    def __setattr__(self, name, value):
        # https://stackoverflow.com/questions/4017572/how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
        name = self.__aliases__.get(name, name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        # https://stackoverflow.com/questions/4017572/how-can-i-make-an-alias-to-a-non-function-member-attribute-in-a-python-class
        if name == "aliases":
            raise AttributeError  # http://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
        name = self.__aliases__.get(name, name)
        #return getattr(self, name) #Causes infinite recursion on non-existent attribute
        return object.__getattribute__(self, name)

########################################################################
# class ValueEventArray
########################################################################
class ValueEventArray(BaseValueEventArray):
    """A multiseries eventarray with shared support.

    Parameters
    ----------
    data : array of np.array(dtype=np.float64) event datas in seconds.
        Array of length n_series, each entry with shape (n_data,)
    fs : float, optional
        Sampling rate in Hz. Default is 30,000
    support : EpochArray, optional
        EpochArray on which eventarrays are defined.
        Default is [0, last event] inclusive.
    label : str or None, optional
        Information pertaining to the source of the eventarray.
    cell_type : list (of length n_series) of str or other, optional
        Identified cell type indicator, e.g., 'pyr', 'int'.
    series_ids : list (of length n_series) of indices corresponding to
        curated data. If no series_ids are specified, then [1,...,n_series]
        will be used. WARNING! The first series will have index 1, not 0!
    meta : dict
        Metadata associated with eventarray.

    Attributes
    ----------
    data : array of np.array(dtype=np.float64) event datas in seconds.
        Array of length n_series, each entry with shape (n_data,)
    support : EpochArray on which eventarray is defined.
    n_events: np.array(dtype=np.int) of shape (n_series,)
        Number of events in each series.
    fs: float
        Sampling frequency (Hz).
    cell_types : np.array of str or other
        Identified cell type for each series.
    label : str or None
        Information pertaining to the source of the eventarray.
    meta : dict
        Metadata associated with eventseries.
    """

    __attributes__ = ["_data"]
    __attributes__.extend(BaseValueEventArray.__attributes__)
    def __init__(self, abscissa_vals=None, values=None, *, fs=None, support=None,
                 series_ids=None, empty=False, **kwargs):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._abscissa.support = type(self._abscissa.support)(empty=True)
            return

        # set default sampling rate
        if fs is None:
            fs = 30000
            logging.info("No sampling rate was specified! Assuming default of {} Hz.".format(fs))

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

        def is_single_series(data):
            """Returns True if data represents event datas from a single series.

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
                    logging.info("event datas input has too many layers!")
                    try:
                        if max(np.array(data).shape[:-1]) > 1:
            #                 singletons = True
                            return False
                    except ValueError:
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
            if is_single_series(data):
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

        def standardize_values_to_2d(data):
            data = standardize_to_2d(data)
            for ii, series in enumerate(data):
                if len(series.shape) == 2:
                    pass
                else:
                    for xx in series:
                        if len(np.atleast_1d(xx)) > 1:
                            raise ValueError('each series must have a fixed number of values; mismatch in series {}'.format(ii))
            return data

        abscissa_vals = standardize_to_2d(abscissa_vals)
        values = standardize_values_to_2d(values)

        data = []
        for a, v in zip(abscissa_vals, values):
            data.append(np.vstack((a, v.T)).T)
        data = np.array(data)

        #sort event series, but only if necessary:
        for ii, train in enumerate(abscissa_vals):
            if not utils.is_sorted(train):
                sortidx = np.argsort(train)
                data[ii] = (data[ii])[sortidx,:]

        kwargs["fs"] = fs
        kwargs["series_ids"] = series_ids

        self._data = data  # this is necessary so that
        # super() can determine self.n_series when initializing.

        # initialize super so that self.fs is set:
        super().__init__(**kwargs)

        # print(self.type_name, kwargs)

        # if only empty data were received AND no support, attach an
        # empty support:
        if np.sum([st.size for st in data]) == 0 and support is None:
            warnings.warn("no events; cannot automatically determine support")
            support = type(self._abscissa.support)(empty=True)

        # determine eventarray support:
        if support is None:
            self.support = type(self._abscissa.support)(np.array([self.first_event, self.last_event + 1/fs]))
        else:
            # restrict events to only those within the eventseries
            # array's support:
            self.support = support

        # TODO: if sorted, we may as well use the fast restrict here as well?
        data = self._restrict_to_interval_array_fast(
            intervalarray=self.support,
            data=data)

        self._data = data

    def partition(self, ds=None, n_intervals=None):
        """Returns a BaseEventArray whose support has been partitioned.

        # Irrespective of whether 'ds' or 'n_intervals' are used, the exact
        # underlying support is propagated, and the first and last points
        # of the supports are always included, even if this would cause
        # n_points or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Maximum duration (in seconds), for each interval.
        n_points : int, optional
            Number of intervals. If ds is None and n_intervals is None, then
            default is to use n_intervals = 100

        Returns
        -------
        out : BaseEventArray
            BaseEventArray that has been partitioned.
        """

        out = copy.copy(self)
        abscissa = copy.deepcopy(out._abscissa)
        abscissa.support = abscissa.support.partition(ds=ds, n_intervals=n_intervals)
        out._abscissa = abscissa
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)

        return out

    def _copy_without_data(self):
        """Return a copy of self, without event datas."""
        out = copy.copy(self) # shallow copy
        out._data = None
        out = copy.deepcopy(out) # just to be on the safe side, but at least now we are not copying the data!
        return out

    def copy(self):
        """Returns a copy of the EventArray."""
        newcopy = copy.deepcopy(self)
        newcopy.loc = ItemGetter_loc(newcopy)
        newcopy.iloc = ItemGetter_iloc(newcopy)
        return newcopy

    def __iter__(self):
        """EventArray iterator initialization."""
        self._index = 0
        return self

    def __next__(self):
        """EventArray iterator advancer."""
        index = self._index

        if index > self._abscissa.support.n_intervals - 1:
            raise StopIteration

        self._index += 1
        return self.loc[index]

    def _intervalslicer(self, idx):
        """Helper function to restrict object to EpochArray."""
        # if self.isempty:
        #     return self

        if isinstance(idx, core.IntervalArray):
            if idx.isempty:
                return type(self)(empty=True)
            support = self._abscissa.support.intersect(
                    interval=idx,
                    boundaries=True
                    ) # what if fs of slicing interval is different?
            if support.isempty:
                return type(self)(empty=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                data = self._restrict_to_interval_array_fast(
                    intervalarray=support,
                    data=self.data,
                    copyover=True
                    )
                eventarray = self._copy_without_data()
                eventarray._data = data
                eventarray._abscissa.support = support
                eventarray.loc = ItemGetter_loc(eventarray)
                eventarray.iloc = ItemGetter_iloc(eventarray)
            return eventarray
        elif isinstance(idx, int):
            eventarray = self._copy_without_data()
            support = self._abscissa.support[idx]
            eventarray._abscissa.support = support
            if (idx >= self._abscissa.support.n_intervals) or idx < (-self._abscissa.support.n_intervals):
                eventarray.loc = ItemGetter_loc(eventarray)
                eventarray.iloc = ItemGetter_iloc(eventarray)
                return eventarray
            else:
                data = self._restrict_to_interval_array_fast(
                        intervalarray=support,
                        data=self.data,
                        copyover=True
                        )
                eventarray._data = data
                eventarray._abscissa.support = support
                eventarray.loc = ItemGetter_loc(eventarray)
                eventarray.iloc = ItemGetter_iloc(eventarray)
                return eventarray
        else:  # most likely slice indexing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    support = self._abscissa.support[idx]
                    data = self._restrict_to_interval_array_fast(
                        intervalarray=support,
                        data=self.data,
                        copyover=True
                        )
                    eventarray = self._copy_without_data()
                    eventarray._data = data
                    eventarray._abscissa.support = support
                    eventarray.loc = ItemGetter_loc(eventarray)
                    eventarray.iloc = ItemGetter_iloc(eventarray)
                return eventarray
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))


    def __getitem__(self, idx):
        """EventArray index access.

        By default, this method is bound to ValueEventArray.loc
        """
        return self.loc[idx]

    @property
    def isempty(self):
        """(bool) Empty EventArray."""
        try:
            return np.sum([len(st) for st in self.data]) == 0
        except TypeError:
            return True  # this happens when self.data == None

    @property
    def n_series(self):
        """(int) The number of series."""
        try:
            return utils.PrettyInt(len(self.data))
        except TypeError:
            return 0

    @property
    def n_active(self):
        """(int) The number of active series.

        A series is considered active if it fired at least one event.
        """
        if self.isempty:
            return 0
        return utils.PrettyInt(np.count_nonzero(self.n_events))

    @property
    def events(self):
        events = []
        for series in self.data:
            events.append(series[:,0].squeeze())

        return events

    @property
    def values(self):
        values = []
        for series in self.data:
            values.append(series[:,1:].squeeze())

        return values

    def flatten(self, *, series_id=None):
        """Collapse events across series.

        Parameters
        ----------
        series_id: (int)
            (series) ID to assign to flattened event series, default is 0.
        """
        if self.n_series < 2:  # already flattened
            return self

        # default args:
        if series_id is None:
            series_id = 0

        flattened = self._copy_without_data()

        flattened._series_ids = [series_id]

        raise NotImplementedError
        alldatas = self.data[0]
        for series in range(1,self.n_series):
            alldatas = utils.linear_merge(alldatas, self.data[series])

        flattened._data = np.array(list(alldatas), ndmin=2)
        flattened.loc = ItemGetter_loc(flattened)
        flattened.iloc = ItemGetter_iloc(flattened)
        return flattened

    @staticmethod
    def _restrict_to_interval_array_fast(intervalarray, data, copyover=True):
        """Return data restricted to an IntervalArray.

        This function assumes sorted event datas, so that binary search can
        be used to quickly identify slices that should be kept in the
        restriction. It does not check every event data.

        Parameters
        ----------
        intervalarray : IntervalArray or EpochArray
        data : list or array-like, each element of size (n_events, n_values).
        """
        if intervalarray.isempty:
            n_series = len(data)
            data = np.zeros((n_series,0))
            return data

        singleseries = len(data)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            data = copy.copy(data)

        # NOTE: this used to assume multiple series for the enumeration to work
        for series, evt_data in enumerate(data):
            indices = []
            for epdata in intervalarray.data:
                t_start = epdata[0]
                t_stop = epdata[1]
                frm, to = np.searchsorted(evt_data[:,0], (t_start, t_stop))
                indices.append((frm, to))
            indices = np.array(indices, ndmin=2)
            if np.diff(indices).sum() < len(evt_data):
                logging.info('ignoring events outside of eventarray support')
            if singleseries:
                data_list = []
                for start, stop in indices:
                    data_list.extend(evt_data[start:stop])
                data = np.array(data_list, ndmin=2)
            else:
                # here we have to do some annoying conversion between
                # arrays and lists to fully support jagged array
                # mutation
                data_list = []
                for start, stop in indices:
                    data_list.extend(evt_data[start:stop])
                data_ = data.tolist()
                data_[series] = np.array(data_list)
                data = np.array(data_)
        return data

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.isempty:
                return "<empty " + self.type_name + address_str + ">"
            if self._abscissa.support.n_intervals > 1:
                epstr = " ({} segments)".format(self._abscissa.support.n_intervals)
            else:
                epstr = ""
            if self.fs is not None:
                fsstr = " at %s Hz" % self.fs
            else:
                fsstr = ""
            numstr = " %s series" % self.n_series # TODO # FIXME swap this with type specific label, e.g., 'units'
        return "<%s%s:%s%s>%s" % (self.type_name, address_str, numstr, epstr, fsstr)

    def bin(self, *, ds=None, method='accumulative'):
        """Return a binned eventarray."""
        raise NotImplementedError
        return BinnedValueEventArray(self, ds=ds, method=method)

    @property
    def data(self):
        """Event datas in seconds."""
        return self._data

    @property
    def n_events(self):
        """(np.array) The number of events in each series."""
        if self.isempty:
            return 0
        return np.array([len(series) for series in self.data])

    @property
    def n_values(self):
        """(int) The number of values associated with each event series."""
        if self.isempty:
            return 0
        n_values = []
        for series in self.data:
            n_values.append(series.squeeze().shape[1] - 1)
        return n_values

    @property
    def issorted(self):
        """(bool) Sorted EventArray."""
        if self.isempty:
            return True
        return np.array(
            [utils.is_sorted(eventarray[:,0]) for eventarray in self.data]
            ).all()

    def _reorder_series_by_idx(self, neworder, inplace=False):
        """Reorder series according to a specified order.

        neworder must be list-like, of size (n_series,)

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
            utils.swap_rows(out._data, frm, to)
            out._series_ids[frm], out._series_ids[to] = out._series_ids[to], out._series_ids[frm]
            # TODO: re-build series tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

    def reorder_series_by_ids(self, neworder, *, inplace=False):
        """Reorder series according to a specified order.

        neworder must be list-like, of size (n_series,) and in terms of
        series_ids

        Return
        ------
        out : reordered EventArray
        """
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        neworder = [self.series_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._data, frm, to)
            out._series_ids[frm], out._series_ids[to] = out._series_ids[to], out._series_ids[frm]
            # TODO: re-build series tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

    @property
    def first_event(self):
        """Returns the [time of the] first event across all series."""
        first = np.inf
        for series in self.data:
            if series[0,0] < first:
                first = series[0,0]
        return first

    @property
    def last_event(self):
        """Returns the [time of the] last event across all series."""
        last = -np.inf
        for series in self.data:
            if series[-1,0] > last:
                last = series[-1,0]
        return last

#----------------------------------------------------------------------#
#======================================================================#