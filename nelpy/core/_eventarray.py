__all__ = ['EventArray',
           'BinnedEventArray',
           'SpikeTrainArray',
           'BinnedSpikeTrainArray']

""" idea is to have abscissa and ordinate, and to use aliasing to have n_series,
    _series_subset, series_ids, (or trial_ids), and so on.

    What is the genericized class? EventArray? eventseries, eventcollection

    when event <==> spike, abscissa <==> data, eventseries <==> eventarray
         eventseries_id <==> series_id, eventseries_type <==> series, then we have a
         EventArray. (n_events, n_spikes)

    series ==> series (series, trial, DIO, ...)


    event rate (smooth; ds, sigma)
    series_id
    eventarray shift
    ISI
    PSTH
"""

import numpy as np
import copy
import numbers
import logging

from abc import ABC, abstractmethod

from .. import core
from .. import utils
from .. import version

from ..utils_.decorators import keyword_deprecation, keyword_equivalence
from . import _accessors

########################################################################
# class BaseEventArray
########################################################################
class BaseEventArray(ABC):
    """Base class for EventArray and BinnedEventArray.

    NOTE: This class can't really be instantiated, almost like a pseudo
    abstract class. In particular, during initialization it might fail
    because it checks the n_series of its derived classes to validate
    input to series_ids and series_labels. If NoneTypes are used, then you
    may actually succeed in creating an instance of this class, but it
    will be pretty useless.

    This docstring only applies to this base class, as subclasses may have
    different behavior. Therefore read the particular subclass' docstring
    for the most accurate information.

    Parameters
    ----------
    fs: float, optional
        Sampling rate / frequency (Hz).
    series_ids : list of int, optional
        Unit IDs
    series_labels : list of str, optional
        Labels corresponding to series. Default casts series_ids to str.
    series_tags : optional
        Tags correponding to series.
        NOTE: Currently we do not do any input validation so these can
        be any type. We also don't use these for anything yet.
    label : str, optional
        Information pertaining to the source of the event series.
        Default is None.
    empty : bool, optional
        Whether to create an empty class instance (no data).
        Default is False
    abscissa : optional
        Object for the abscissa (x-axis) coordinate
    ordinate : optional
        Object for the ordinate (y-axis) coordinate
    kwargs : optional
        Other keyword arguments. NOTE: Currently we do not do anything
        with these.

    Attributes
    ----------
    n_series : int
        Number of series in event series.
    issempty : bool
        Whether the class instance is empty (no data)
    series_ids : list of int
        Unit IDs
    series_labels : list of str
        Labels corresponding to series. Default casts series_ids to str.
    series_tags :
        Tags corresponding to series.
        NOTE: Currently we do not do any input validation so these can
        be any type. We also don't use these for anything yet.
    n_intervals : int
        The number of underlying intervals.
    support : nelpy.IntervalArray
        The support of the EventArray.
    fs: float
        Sampling frequency (Hz).
    label : str or None
        Information pertaining to the source of the event series.
    """

    __aliases__ = {}

    __attributes__ = ["_fs", "_series_ids", "_series_labels", "_series_tags", "_label"]

    def __init__(self, *, fs=None, series_ids=None, series_labels=None,
                 series_tags=None, label=None, empty=False, abscissa=None, ordinate=None, **kwargs):

        self.__version__ = version.__version__
        self.type_name = self.__class__.__name__
        if abscissa is None:
            abscissa = core.Abscissa() #TODO: integrate into constructor?
        if ordinate is None:
            ordinate = core.Ordinate() #TODO: integrate into constructor?
        self._abscissa = abscissa
        self._ordinate = ordinate

        series_label = kwargs.pop('series_label', None)
        if series_label is None:
            series_label = 'series'
        self._series_label = series_label

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._abscissa.support = type(self._abscissa.support)(empty=True)
            self.loc = _accessors.ItemGetterLoc(self)
            self.iloc = _accessors.ItemGetterIloc(self)
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

        # if series_labels is empty, default to series_ids
        if series_labels is None:
            series_labels = series_ids

        series_labels = np.array(series_labels, ndmin=1)  # standardize

        self.series_ids = series_ids
        self.series_labels = series_labels
        self._series_tags = series_tags  # no input validation yet
        self.label = label

        self.loc = _accessors.ItemGetterLoc(self)
        self.iloc = _accessors.ItemGetterIloc(self)

    def __renew__(self):
        """Re-attach slicers and indexers."""
        self.loc = _accessors.ItemGetterLoc(self)
        self.iloc = _accessors.ItemGetterIloc(self)

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        return "<BaseEventArray" + address_str + ">"

    @abstractmethod
    @keyword_equivalence(this_or_that={'n_intervals':'n_epochs'})
    def partition(self, ds=None, n_intervals=None):
        """Returns a BaseEventArray whose support has been partitioned.

        # Regardless of whether 'ds' or 'n_intervals' are used, the exact
        # underlying support is propagated, and the first and last points
        # of the supports are always included, even if this would cause
        # n_points or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Maximum duration (in seconds), for each interval.
        n_intervals : int, optional
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
    def series_labels(self):
        """Labels corresponding to series contained in the BaseEventArray."""
        if self._series_labels is None:
            logging.warning("series labels have not yet been specified")
            return self.series_ids
        return self._series_labels

    @series_labels.setter
    def series_labels(self, val):
        if len(val) != self.n_series:
            raise TypeError("labels must be of length n_series")
        else:
            try:
                # cast to str:
                labels = [str(label) for label in val]
            except TypeError:
                raise TypeError("labels must be string-like")
        self._series_labels = labels

    @property
    def series_tags(self):
        """Tags corresponding to series contained in the BaseEventArray"""
        if self._series_tags is None:
            logging.warning("series tags have not yet been specified")
        return self._series_tags

    @property
    def support(self):
        """(nelpy.IntervalArray) The support of the EventArray."""
        return self._abscissa.support

    @support.setter
    def support(self, val):
        """(nelpy.IntervalArray) The support of the EventArray."""
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
        self._restrict_to_interval(self._abscissa.support)

    @property
    def domain(self):
        """(nelpy.IntervalArray) The domain of the EventArray."""
        return self._abscissa.domain

    @domain.setter
    def domain(self, val):
        """(nelpy.IntervalArray) The domain of the EventArray."""
        # modify domain
        if isinstance(val, type(self._abscissa.support)):
            self._abscissa.domain = val
        elif isinstance(val, (tuple, list)):
            self._abscissa.domain = type(self._abscissa.support)([val[0], val[1]])
        else:
            raise TypeError('support must be of type {}'.format(str(type(self._abscissa.support))))
        # restrict data to new support
        self._restrict_to_interval(self._abscissa.support)

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
            logging.warning("label has not yet been specified")
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
# class EventArray
########################################################################
class EventArray(BaseEventArray):
    """A multiseries eventarray with shared support.

    Parameters
    ----------
    abscissa_vals : array of np.array(dtype=np.float64) event datas in seconds.
        Array of length n_series, each entry with shape (n_data,).
    fs : float, optional
        Sampling rate in Hz. Default is 30,000.
    support : IntervalArray, optional
        IntervalArray on which eventarrays are defined.
        Default is [0, last event] inclusive.
    series_ids : list of int, optional
        Unit IDs.
    series_labels : list of str, optional
        Labels corresponding to series. Default casts series_ids to str.
    series_tags : optional
        Tags correponding to series.
        NOTE: Currently we do not do any input validation so these can
        be any type. We also don't use these for anything yet.
    label : str or None, optional
        Information pertaining to the source of the eventarray.
    empty : bool, optional
        Whether an empty EventArray should be constructed (no data).
    assume_sorted : boolean, optional
        Whether the abscissa values should be treated as sorted (non-decreasing)
        or not. Significant overhead during RSASA object creation can be removed
        if this is True, but note that unsorted abscissa values will mess
        everything up.
        Default is False
    kwargs : optional
        Additional keyword arguments to forward along to the BaseEventArray
        constructor.

    Attributes
    ----------
    Note : Read the docstring for the BaseEventArray superclass for additional
    attributes that are defined there.
    isempty : bool
        Whether the EventArray is empty (no data).
    n_series : int
        The number of series.
    n_active : int
        The number of active series. A series is considered active if
        it fired at least one event.
    data : array of np.array(dtype=np.float64) event datas in seconds.
        Array of length n_series, each entry with shape (n_data,).
    n_events : np.ndarray
        The number of events in each series.
    issorted : bool
        Whether the data are sorted.
    first_event : np.float
        The time of the very first event, across all series.
    last_event : np.float
        The time of the very last event, across all series.
    """

    __attributes__ = ["_data"]
    __attributes__.extend(BaseEventArray.__attributes__)
    def __init__(self, abscissa_vals=None, *, fs=None, support=None,
                 series_ids=None, series_labels=None, series_tags=None,
                 label=None, empty=False, assume_sorted=None, **kwargs):

        if assume_sorted is None:
            assume_sorted = False

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
            logging.warning("No sampling rate was specified! Assuming default of {} Hz.".format(fs))

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

        data = standardize_to_2d(abscissa_vals)

        # If user said to assume the absicssa vals are sorted but they actually
        # aren't, then the mistake will get propagated down. The responsibility
        # therefore lies on the user whenever he/she uses assume_sorted=True
        # as a constructor argument
        for ii, train in enumerate(data):
            if not assume_sorted:
                # sort event series, but only if necessary
                if not utils.is_sorted(train):
                    data[ii] = np.sort(train)
            else:
                data[ii] = np.sort(train)

        kwargs["fs"] = fs
        kwargs["series_ids"] = series_ids
        kwargs["series_labels"] = series_labels
        kwargs["series_tags"] = series_tags
        kwargs["label"] = label

        self._data = data  # this is necessary so that
        # super() can determine self.n_series when initializing.

        # initialize super so that self.fs is set:
        super().__init__(**kwargs)

        # print(self.type_name, kwargs)

        # if only empty data were received AND no support, attach an
        # empty support:
        if np.sum([st.size for st in data]) == 0 and support is None:
            logging.warning("no events; cannot automatically determine support")
            support = type(self._abscissa.support)(empty=True)

        # determine eventarray support:
        if support is None:
            first_spk = np.nanmin(np.array([series[0] for series in data if len(series) !=0]))
            # BUG: if eventseries is empty np.array([]) then series[-1]
            # raises an error in the following:
            # FIX: list[-1] raises an IndexError for an empty list,
            # whereas list[-1:] returns an empty list.
            last_spk = np.nanmax(np.array([series[-1:] for series in data if len(series) !=0]))
            self.support = type(self._abscissa.support)(np.array([first_spk, last_spk + 1/fs]))
            # in the above, there's no reason to restrict to support
        else:
            # restrict events to only those within the eventseries
            # array's support:
            self.support = support

        # TODO: if sorted, we may as well use the fast restrict here as well?
        self._restrict_to_interval(self._abscissa.support, data=data)

    @keyword_equivalence(this_or_that={'n_intervals':'n_epochs'})
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

        out = self.copy()
        abscissa = copy.deepcopy(out._abscissa)
        abscissa.support = abscissa.support.partition(ds=ds, n_intervals=n_intervals)
        out._abscissa = abscissa
        out.__renew__()

        return out

    def _copy_without_data(self):
        """Return a copy of self, without event datas.
        Note: the support is left unchanged.
        """
        out = copy.copy(self) # shallow copy
        out._data = np.array(self.n_series*[None])
        out = copy.deepcopy(out) # just to be on the safe side, but at least now we are not copying the data!
        out.__renew__()
        return out

    def copy(self):
        """Returns a copy of the EventArray."""
        newcopy = copy.deepcopy(self)
        newcopy.__renew__()
        return newcopy

    def __iter__(self):
        """EventArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """EventArray iterator advancer."""
        index = self._index
        if index > self._abscissa.support.n_intervals - 1:
            raise StopIteration

        self._index += 1
        return self.loc[index]

    def __getitem__(self, idx):
        """EventArray index access.

        By default, this method is bound to EventArray.loc
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

    def flatten(self, *, series_id=None, series_label=None):
        """Collapse events across series.

        WARNING! series_tags are thrown away when flattening.

        Parameters
        ----------
        series_id: (int)
            (series) ID to assign to flattened event series, default is 0.
        series_label (str)
            (series) Label for event series, default is 'flattened'.
        """
        if self.n_series < 2:  # already flattened
            return self

        # default args:
        if series_id is None:
            series_id = 0
        if series_label is None:
            series_label = "flattened"

        flattened = self._copy_without_data()

        flattened._series_ids = [series_id]
        flattened._series_labels = [series_label]
        flattened._series_tags = None

        alldatas = self.data[0]
        for series in range(1,self.n_series):
            alldatas = utils.linear_merge(alldatas, self.data[series])

        flattened._data = np.array(list(alldatas), ndmin=2)
        flattened.__renew__()
        return flattened

    def _restrict(self, intervalslice, seriesslice, *, subseriesslice=None):

        self._restrict_to_series_subset(seriesslice)
        self._restrict_to_interval(intervalslice)
        return self

    def _restrict_to_series_subset(self, idx):

        # Warning: This function can mutate data

        # TODO: Update tags
        try:
            self._data = self._data[idx]
            singleseries = (len(self._data) == 1)
            if singleseries:
                self._data = np.array(self._data[0], ndmin=2)
            self._series_ids = list(np.atleast_1d(np.atleast_1d(self._series_ids)[idx]))
            self._series_labels = list(np.atleast_1d(np.atleast_1d(self._series_labels)[idx]))
        except AttributeError:
            self._data = self._data[idx]
            singleseries = (len(self._data) == 1)
            if singleseries:
                self._data = np.array(self._data[0], ndmin=2)
            self._series_ids = list(np.atleast_1d(np.atleast_1d(self._series_ids)[idx]))
            self._series_labels = list(np.atleast_1d(np.atleast_1d(self._series_labels)[idx]))
        except IndexError:
            raise IndexError("One of more indices were out of bounds for n_series with size {}"
                             .format(self.n_series))
        except Exception:
            raise TypeError("Unsupported indexing type {}".format(type(idx)))

        return self

    def _restrict_to_interval(self, intervalslice, *, data=None):
        """Return data restricted to an intervalarray.

        This function assumes sorted event datas, so that binary search can
        be used to quickly identify slices that should be kept in the
        restriction. It does not check every event data.

        Parameters
        ----------
        intervalarray : nelpy.IntervalArray
        """

        # Warning: this function can mutate data
        # This should be called from _restrict only. That's where
        # intervalarray is first checked against the support.
        # This function assumes that has happened already, so
        # every point in intervalarray is also in the support

        # NOTE: this used to assume multiple series for the enumeration to work

        if data is None:
            data = self._data

        if isinstance(intervalslice, slice):
            if intervalslice.start == None and intervalslice.stop == None and intervalslice.step == None:
                # no restriction on interval
                return self

        newintervals = self._abscissa.support[intervalslice].merge()
        if newintervals.isempty:
            logging.warning("Index resulted in empty interval array")
            return self.empty(inplace=True)

        issue_warning = False
        if not self.isempty:
            for series, evt_data in enumerate(data):
                indices = []
                for epdata in newintervals.data:
                    t_start = epdata[0]
                    t_stop = epdata[1]
                    frm, to = np.searchsorted(evt_data, (t_start, t_stop))
                    indices.append((frm, to))
                indices = np.array(indices, ndmin=2)
                if np.diff(indices).sum() < len(evt_data):
                    issue_warning = True
                singleseries = (len(self._data) == 1)
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
                    data_ = data.tolist()  # this creates copy
                    data_[series] = np.array(data_list)
                    data = utils.ragged_array(data_)
            self._data = data
            if issue_warning:
                logging.warning(
                        'ignoring events outside of eventarray support')

        self._abscissa.support = newintervals
        return self

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        logging.disable(logging.CRITICAL)
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
        if self.label is not None:
            labelstr = " from %s" % self.label
        else:
            labelstr = ""
        numstr = " %s %s" % (self.n_series, self._series_label)
        logging.disable(0)
        return "<%s%s:%s%s>%s%s" % (self.type_name, address_str, numstr, epstr, fsstr, labelstr)

    def bin(self, *, ds=None):
        """Return a binned eventarray."""
        return BinnedEventArray(self, ds=ds)

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
    def issorted(self):
        """(bool) Sorted EventArray."""
        if self.isempty:
            return True
        return np.array(
            [utils.is_sorted(eventarray) for eventarray in self.data]
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
            out = self.copy()

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._data, frm, to)
            out._series_ids[frm], out._series_ids[to] = out._series_ids[to], out._series_ids[frm]
            out._series_labels[frm], out._series_labels[to] = out._series_labels[to], out._series_labels[frm]
            # TODO: re-build series tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        out.__renew__()
        return out

    def reorder_series(self, neworder, *, inplace=False):
        """Reorder series according to a specified order.

        neworder must be list-like, of size (n_series,) and in terms of
        series_ids

        Return
        ------
        out : reordered EventArray
        """
        raise DeprecationWarning("reorder_series has been deprecated. Use reorder_series_by_id(x/s) instead!")

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
            out = self.copy()

        neworder = [self.series_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._data, frm, to)
            out._series_ids[frm], out._series_ids[to] = out._series_ids[to], out._series_ids[frm]
            out._series_labels[frm], out._series_labels[to] = out._series_labels[to], out._series_labels[frm]
            # TODO: re-build series tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        out.__renew__()
        return out

    def get_event_firing_order(self):
        """Returns a list of series_ids such that the series are ordered
        by when they first fire in the EventArray.

        Return
        ------
        firing_order : list of series_ids
        """

        first_events = [(ii, series[0]) for (ii, series) in enumerate(self.data) if len(series) !=0]
        first_events_series_ids = np.array(self.series_ids)[[fs[0] for fs in first_events]]
        first_events_datas = np.array([fs[1] for fs in first_events])
        sortorder = np.argsort(first_events_datas)
        first_events_series_ids = first_events_series_ids[sortorder]
        remaining_ids = list(set(self.series_ids) - set(first_events_series_ids))
        firing_order = list(first_events_series_ids)
        firing_order.extend(remaining_ids)

        return firing_order

    @property
    def first_event(self):
        """Returns the [time of the] first event across all series."""
        first = np.inf
        for series in self.data:
            if series[0] < first:
                first = series[0]
        return first

    @property
    def last_event(self):
        """Returns the [time of the] last event across all series."""
        last = -np.inf
        for series in self.data:
            if series[-1] > last:
                last = series[-1]
        return last

    def empty(self, *, inplace=False):
        """Remove data (but not metadata) from EventArray.

        Attributes 'data', and 'support' are both emptied.

        Note: n_series, series_ids, etc. are all preserved.
        """
        if not inplace:
            out = self._copy_without_data()
            out._abscissa.support = type(self._abscissa.support)(empty=True)
            return out
        out = self
        out._data = np.array(self.n_series*[None])
        out._abscissa.support = type(self._abscissa.support)(empty=True)
        out.__renew__()
        return out

########################################################################
# class BinnedEventArray
########################################################################
class BinnedEventArray(BaseEventArray):
    """BinnedEventArray.

    Parameters
    ----------
    eventarray : nelpy.EventArray or nelpy.RegularlySampledAnalogSignalArray
        Input data.
    ds : float
        The bin width, in seconds.
        Default is 0.0625 (62.5 ms)
    empty : bool, optional
        Whether an empty BinnedEventArray should be constructed (no data).
    fs : float, optional
        Sampling rate in Hz. If fs is passed as a parameter, then data
        is assumed to be in sample numbers instead of actual data.
    kwargs : optional
        Additional keyword arguments to forward along to the BaseEventArray
        constructor.

    Attributes
    ----------
    Note : Read the docstring for the BaseEventArray superclass for additional
    attributes that are defined there.
    isempty : bool
        Whether the BinnedEventArray is empty (no data).
    n_series : int
        The number of series.
    bin_centers : np.ndarray
        The bin centers, in seconds.
    event_centers : np.ndarray
        The centers of each event, in seconds.
    data : np.array, with shape (n_series, n_bins)
        Event counts in all bins.
    bins : np.ndarray
        The bin edges, in seconds.
    binnedSupport : np.ndarray, with shape (n_intervals, 2)
        The binned support of the BinnedEventArray (in
        bin IDs).
    lengths : np.ndarray
        Lengths of contiguous segments, in number of bins.
    eventarray : nelpy.EventArray
        The original eventarray associated with the binned data.
    n_bins : int
        The number of bins.
    ds : float
        Bin width, in seconds.
    n_active : int
        The number of active series. A series is considered active if
        it fired at least one event.
    n_active_per_bin : np.ndarray, with shape (n_bins, )
        Number of active series per data bin.
    n_events : np.ndarray
        The number of events in each series.
    support : nelpy.IntervalArray
        The support of the BinnedEventArray.
    """

    __attributes__ = ["_ds", "_bins", "_data", "_bin_centers",
                      "_binnedSupport", "_eventarray"]
    __attributes__.extend(BaseEventArray.__attributes__)

    def __init__(self, eventarray=None, *, ds=None, empty=False, **kwargs):

        super().__init__(empty=True)

        # if an empty object is requested, return it:
        if empty:
            # super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._abscissa.support = type(self._abscissa.support)(empty=True)
            self._event_centers = None
            return

        # handle casting other nelpy objects to BinnedEventArray:
        if isinstance(eventarray, core.RegularlySampledAnalogSignalArray):
            if eventarray.isempty:
                for attr in self.__attributes__:
                    exec("self." + attr + " = None")
                self._abscissa.support = type(eventarray._abscissa.support)(empty=True)
                self._event_centers = None
                return
            eventarray = eventarray.copy() # Note: this is a deep copy
            n_empty_epochs = np.sum(eventarray.support.lengths==0)
            if n_empty_epochs > 0:
                logging.warning("Detected {} empty epochs. Removing these in the cast object"
                              .format(n_empty_epochs))
                eventarray.support = eventarray.support._drop_empty_intervals()
            if not eventarray.support.ismerged:
                logging.warning("Detected overlapping epochs. Merging these in the cast object")
                eventarray.support = eventarray.support.merge()

            self._eventarray = None
            self._ds = 1/eventarray.fs
            self._series_labels = eventarray._series_labels
            self._bin_centers = eventarray.abscissa_vals
            tmp = np.insert(np.cumsum(eventarray.lengths),0,0)
            self._binnedSupport = np.array((tmp[:-1], tmp[1:]-1)).T
            self._abscissa.support = eventarray.support
            try:
                self._series_ids = (np.array(eventarray.series_labels).astype(int)).tolist()
            except (ValueError, TypeError):
                self._series_ids = (np.arange(eventarray.n_signals) + 1).tolist()
            self._data = eventarray._ydata_rowsig

            bins = []
            for starti, stopi in self._binnedSupport:
                bins_edges_in_interval = (self._bin_centers[starti:stopi+1] - self._ds/2).tolist()
                bins_edges_in_interval.append(self._bin_centers[stopi] + self._ds/2)
                bins.extend(bins_edges_in_interval)
            self._bins = np.array(bins)
            return

        if type(eventarray).__name__ == 'BinnedSpikeTrainArray':
            # old-style nelpy BinnedSpikeTrainArray object?
            try:
                self._eventarray = eventarray._spiketrainarray
                self._ds = eventarray.ds
                self._series_labels = eventarray.unit_labels
                self._bin_centers = eventarray.bin_centers
                self._binnedSupport = eventarray.binnedSupport
                try:
                    self._abscissa.support = core.EpochArray(eventarray.support.data)
                except AttributeError:
                    self._abscissa.support = core.EpochArray(eventarray.support.time)
                self._series_ids = eventarray.unit_ids
                self._data = eventarray.data
                return
            except Exception:
                pass

        if not isinstance(eventarray, EventArray):
            raise TypeError('eventarray must be a nelpy.EventArray object.')

        self._ds = None
        self._bin_centers = np.array([])
        self._event_centers = None

        logging.disable(logging.CRITICAL)
        kwargs = {'fs': eventarray.fs,
                    'series_ids': eventarray.series_ids,
                    'series_labels': eventarray.series_labels,
                    'series_tags': eventarray.series_tags,
                    'label': eventarray.label}
        logging.disable(0)

        # initialize super so that self.fs is set:
        self._data = np.zeros((eventarray.n_series,0))
            # the above is necessary so that super() can determine
            # self.n_series when initializing. self.data will
            # be updated later in __init__ to reflect subsequent changes
        super().__init__(**kwargs)

        if ds is None:
            logging.warning('no bin size was given, assuming 62.5 ms')
            ds = 0.0625

        self._eventarray = eventarray # TODO: remove this if we don't need it, or decide that it's too wasteful
        self._abscissa = copy.deepcopy(eventarray._abscissa)
        self.ds = ds

        self._bin_events(
            eventarray=eventarray,
            intervalArray=eventarray.support,
            ds=ds
            )

    def __mul__(self, other):
        """Overloaded * operator"""

        if isinstance(other, numbers.Number):
            neweva = self.copy()
            neweva._data = self.data * other
            return neweva
        elif isinstance(other, np.ndarray):
            neweva = self.copy()
            neweva._data = (self.data.T * other).T
            return neweva
        else:
            raise TypeError("unsupported operand type(s) for *: '{}' and '{}'".format(str(type(self)), str(type(other))))

    def __rmul__(self, other):
        """Overloaded * operator"""
        return self.__mul__(other)

    def __sub__(self, other):
        """Overloaded - operator"""
        if isinstance(other, numbers.Number):
            neweva = self.copy()
            neweva._data = self.data - other
            return neweva
        elif isinstance(other, np.ndarray):
            neweva = self.copy()
            neweva._data = (self.data.T - other).T
            return neweva
        else:
            raise TypeError("unsupported operand type(s) for -: '{}' and '{}'".format(str(type(self)), str(type(other))))

    def __add__(self, other):
        """Overloaded + operator"""

        if isinstance(other, numbers.Number):
            neweva = self.copy()
            neweva._data = self.data + other
            return neweva
        elif isinstance(other, np.ndarray):
            neweva = self.copy()
            neweva._data = (self.data.T + other).T
            return neweva
        elif isinstance(other, type(self)):

            #TODO: additional checks need to be done, e.g., same series ids...
            assert self.n_series == other.n_series
            support = self._abscissa.support + other.support

            newdata = []
            for series in range(self.n_series):
                newdata.append(np.append(self.data[series], other.data[series]))

            fs = self.fs
            if self.fs != other.fs:
                fs = None
            return type(self)(newdata, support=support, fs=fs)
        else:
            raise TypeError("unsupported operand type(s) for +: '{}' and '{}'".format(str(type(self)), str(type(other))))

    def __truediv__(self, other):
        """Overloaded / operator"""

        if isinstance(other, numbers.Number):
            neweva = self.copy()
            neweva._data = self.data / other
            return neweva
        elif isinstance(other, np.ndarray):
            neweva = self.copy()
            neweva._data = (self.data.T / other).T
            return neweva
        else:
            raise TypeError("unsupported operand type(s) for /: '{}' and '{}'".format(str(type(self)), str(type(other))))

    def median(self,*,axis=1):
        """Returns the median of each series in BinnedEventArray."""
        try:
            medians = np.nanmedian(self.data, axis=axis).squeeze()
            if medians.size == 1:
                return np.asscalar(medians)
            return medians
        except IndexError:
            raise IndexError("Empty BinnedEventArray; cannot calculate median.")

    def mean(self,*,axis=1):
        """Returns the mean of each series in BinnedEventArray."""
        try:
            means = np.nanmean(self.data, axis=axis).squeeze()
            if means.size == 1:
                return np.asscalar(means)
            return means
        except IndexError:
            raise IndexError("Empty BinnedEventArray; cannot calculate mean.")

    def std(self,*,axis=1):
        """Returns the standard deviation of each series in BinnedEventArray."""
        try:
            stds = np.nanstd(self.data,axis=axis).squeeze()
            if stds.size == 1:
                return np.asscalar(stds)
            return stds
        except IndexError:
            raise IndexError("Empty BinnedEventArray; cannot calculate standard deviation")

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
        std = out.std()
        std[std==0] = 1
        out._data = (out._data.T / std).T
        return out

    def standardize(self, inplace=False):
        """Standardize data (zero mean and unit std deviation)."""
        if inplace:
            out = self
        else:
            out = self.copy()
        out._data = (out._data.T - out.mean()).T
        std = out.std()
        std[std==0] = 1
        out._data = (out._data.T / std).T

        return out

    @keyword_equivalence(this_or_that={'n_intervals':'n_epochs'})
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

        partitioned = type(self)(core.RegularlySampledAnalogSignalArray(self).partition(ds=ds, n_intervals=n_intervals))
        # partitioned.loc = ItemGetter_loc(partitioned)
        # partitioned.iloc = ItemGetter_iloc(partitioned)
        return partitioned

        # raise NotImplementedError('workaround: cast to AnalogSignalArray, partition, and cast back to BinnedEventArray')

    def _copy_without_data(self):
        """Returns a copy of the BinnedEventArray, without data.
        Note: the support is left unchanged, but the binned_support is removed.
        """
        out = copy.copy(self) # shallow copy
        out._bin_centers = None
        out._binnedSupport = None
        out._bins = None
        out._data = np.zeros((self.n_series, 0))
        out._eventarray = out._eventarray._copy_without_data()
        out = copy.deepcopy(out) # just to be on the safe side, but at least now we are not copying the data!
        out.__renew__()
        return out

    def copy(self):
        """Returns a copy of the BinnedEventArray."""
        newcopy = copy.deepcopy(self)
        newcopy.__renew__()
        return newcopy

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty " + self.type_name + address_str + ">"
        ustr = " {} {}".format(self.n_series, self._series_label)
        if self._abscissa.support.n_intervals > 1:
            epstr = " ({} segments) in".format(self._abscissa.support.n_intervals)
        else:
            epstr = " in"
        if self.n_bins == 1:
            bstr = " {} bin of width {}".format(self.n_bins, utils.PrettyDuration(self.ds))
            dstr = ""
        else:
            bstr = " {} bins of width {}".format(self.n_bins, utils.PrettyDuration(self.ds))
            dstr = " for a total of {}".format(utils.PrettyDuration(self.n_bins*self.ds))
        return "<%s%s:%s%s%s>%s" % (self.type_name, address_str, ustr, epstr, bstr, dstr)

    def __iter__(self):
        """BinnedEventArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """BinnedEventArray iterator advancer."""
        index = self._index

        if index > self._abscissa.support.n_intervals - 1:
            raise StopIteration

        # TODO: return self.loc[index], and make sure that __getitem__ is updated
        logging.disable(logging.CRITICAL)
        support = self._abscissa.support[index]
        bsupport = self.binnedSupport[[index],:]

        binnedeventarray = type(self)(empty=True)
        exclude = ["_bins", "_data", "_support", "_bin_centers", "_binnedSupport"]
        attrs = (x for x in self.__attributes__ if x not in exclude)
        for attr in attrs:
            exec("binnedeventarray." + attr + " = self." + attr)
        binindices = np.insert(0, 1, np.cumsum(self.lengths + 1)) # indices of bins
        binstart = binindices[index]
        binstop = binindices[index+1]
        binnedeventarray._bins = self._bins[binstart:binstop]
        binnedeventarray._data = self._data[:,bsupport[0][0]:bsupport[0][1]+1]
        binnedeventarray._abscissa.support = support
        binnedeventarray._bin_centers = self._bin_centers[bsupport[0][0]:bsupport[0][1]+1]
        binnedeventarray._binnedSupport = bsupport - bsupport[0,0]
        logging.disable(0)
        self._index += 1
        binnedeventarray.__renew__()
        return binnedeventarray

    def empty(self, *, inplace=False):
        """Remove data (but not metadata) from BinnedEventArray.

        Attributes 'data', and 'support' 'binnedSupport' are all emptied.

        Note: n_series, series_ids, etc. are all preserved.
        """
        if not inplace:
            out = self._copy_without_data()
            out._abscissa.support = type(self._abscissa.support)(empty=True)
            return out
        out = self
        out._data = np.zeros((self.n_series, 0))
        out._abscissa.support = type(self._abscissa.support)(empty=True)
        out._binnedSupport = None
        out._bin_centers = None
        out._bins = None
        out._eventarray.empty(inplace=True)
        out.__renew__()
        return out

    def __getitem__(self, idx):
        """BinnedEventArray index access.

        By default, this method is bound to .loc
        """
        return self.loc[idx]

    def _restrict(self, intervalslice, seriesslice):
        # This function should be called only by an itemgetter
        # because it mutates data.
        # The itemgetter is responsible for creating copies
        # of objects

        self._restrict_to_series_subset(seriesslice)
        self._eventarray._restrict_to_series_subset(seriesslice)

        self._restrict_to_interval(intervalslice)
        self._eventarray._restrict_to_interval(intervalslice)
        return self

    def _restrict_to_series_subset(self, idx):

        # Warning: This function can mutate data

        if isinstance(idx, core.IntervalArray):
            raise IndexError("Slicing is [intervals, signal]; perhaps you have the order reversed?")

        # TODO: update tags
        try:
            self._data = np.atleast_2d(self.data[idx,:])
            self._series_ids = list(np.atleast_1d(np.atleast_1d(self._series_ids)[idx]))
            self._series_labels = list(np.atleast_1d(np.atleast_1d(self._series_labels)[idx]))
        except IndexError:
            raise IndexError("One of more indices were out of bounds for n_series with size {}"
                             .format(self.n_series))
        except Exception:
            raise TypeError("Unsupported indexing type {}".format(type(idx)))

    def _restrict_to_interval(self, intervalslice):

        # Warning: This function can mutate data. It should only be called from
        # _restrict

        if isinstance(intervalslice, slice):
            if (intervalslice.start == None and
                intervalslice.stop  == None and
                intervalslice.step  == None):
                # no restriction on interval
                return self

        newintervals = self._abscissa.support[intervalslice].merge()
        if newintervals.isempty:
            logging.warning("Index resulted in empty interval array")
            return self.empty(inplace=True)

        bcenter_inds = []
        bin_inds = []
        start = 0
        bsupport = np.zeros((newintervals.n_intervals, 2),
                            dtype=int)
        support_intervals = np.zeros((newintervals.n_intervals, 2))

        if not self.isempty:
            for ii, interval in enumerate(newintervals.data):

                a_start = interval[0]
                a_stop = interval[1]
                frm, to = np.searchsorted(self._bins, (a_start, a_stop))
                # If bin edges equal a_stop, they should still be included
                if self._bins[to] <= a_stop:
                    bin_inds.extend(np.arange(frm, to + 1, step=1))
                else:
                    bin_inds.extend(np.arange(frm, to, step=1))
                    to -= 1
                support_intervals[ii] = [self._bins[frm], self._bins[to]]

                lind, rind = np.searchsorted(self._bin_centers,
                                            (self._bins[frm], self._bins[to]))
                # We don't have to worry about an if-else block here unlike
                # for the bin_inds because the bin_centers can NEVER equal
                # the bins. Therefore we know every interval looks like
                # the following:
                #  first desired bin         last desired bin
                # |------------------|......|-------------------|
                #          ^                                         ^
                #          |                                         |
                #        lind                                      rind
                # Since arange is half-open, the indices we actually take
                # will be such that all bin centers fall within the desired
                # bin edges.
                bcenter_inds.extend(np.arange(lind, rind, step=1))

                bsupport[ii] = [start, start+(to-frm-1)]
                start += to - frm

            self._bins = self._bins[bin_inds]
            self._bin_centers = self._bin_centers[bcenter_inds]
            self._data = np.atleast_2d(self._data[:, bcenter_inds])
            self._binnedSupport = bsupport

        self._abscissa.support = type(self._abscissa.support)(support_intervals)

    @property
    def isempty(self):
        """(bool) Empty BinnedEventArray."""
        try:
            return len(self.bin_centers) == 0
        except TypeError:
            return True  # this happens when self.bin_centers == None

    @property
    def n_series(self):
        """(int) The number of series."""
        try:
            return utils.PrettyInt(self.data.shape[0])
        except AttributeError:
            return 0

    @property
    def centers(self):
        """(np.array) The bin centers (in seconds)."""
        logging.warning("centers is deprecated. Use bin_centers instead.")
        return self.bin_centers

    @property
    def _abscissa_vals(self):
        """(np.array) The bin centers (in seconds)."""
        return self._bin_centers

    @property
    def bin_centers(self):
        """(np.array) The bin centers (in seconds)."""
        return self._bin_centers

    @property
    def event_centers(self):
        """(np.array) The centers (in seconds) of each event."""
        if self._event_centers is None:
            raise NotImplementedError("event_centers not yet implemented")
            # self._event_centers = midpoints
        return self._event_centers

    @property
    def _midpoints(self):
        """(np.array) The centers (in index space) of all events.

        Example
        -------
        ax, img = npl.imagesc(bst.data) # data is of shape (n_series, n_bins)
        # then _midpoints correspond to the xvals at the center of
        # each event.
        ax.plot(bst.event_centers, np.repeat(1, self.n_intervals), marker='o', color='w')

        """
        if self._event_centers is None:
            midpoints = np.zeros(len(self.lengths))
            for idx, l in enumerate(self.lengths):
                midpoints[idx] = np.sum(self.lengths[:idx]) + l/2
            self._event_centers = midpoints
        return self._event_centers

    @property
    def data(self):
        """(np.array) Event counts in all bins, with shape (n_series, n_bins)."""
        return self._data

    @property
    def bins(self):
        """(np.array) The bin edges (in seconds)."""
        return self._bins

    @property
    def binnedSupport(self):
        """(np.array) The binned support of the BinnedEventArray (in
        bin IDs) of shape (n_intervals, 2).
        """
        return self._binnedSupport

    @property
    def lengths(self):
        """Lengths of contiguous segments, in number of bins."""
        if self.isempty:
            return 0
        return np.atleast_1d((self.binnedSupport[:,1] - self.binnedSupport[:,0] + 1).squeeze())

    @property
    def eventarray(self):
        """(nelpy.EventArray) The original EventArray associated with
        the binned data.
        """
        return self._eventarray

    @property
    def n_bins(self):
        """(int) The number of bins."""
        if self.isempty:
            return 0
        return utils.PrettyInt(len(self.bin_centers))

    @property
    def ds(self):
        """(float) Bin width in seconds."""
        return self._ds

    @ds.setter
    def ds(self, val):
        if self._ds is not None:
            raise AttributeError("can't set attribute")
        else:
            try:
                if val <= 0:
                    pass
            except:
                raise TypeError("bin width must be a scalar")
            if val <= 0:
                raise ValueError("bin width must be positive")
            self._ds = val

    @staticmethod
    def _get_bins_inside_interval(interval, ds):
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
                [0,2).

        Parameters
        ----------
        interval : IntervalArray
            IntervalArray containing a single interval with a start, and stop
        ds : float
            Time bin width, in seconds.

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
            logging.warning(
                "interval duration is less than bin size: ignoring...")
            return None, None

        n = int(np.floor(interval.length / ds)) # number of bins

        # linspace is better than arange for non-integral steps
        bins = np.linspace(interval.start, interval.start + n*ds, n+1)
        centers = bins[:-1] + (ds / 2)
        return bins, centers

    def _bin_events(self, eventarray, intervalArray, ds):
        """
        Docstring goes here. TBD. For use with bins that are contained
        wholly inside the intervals.

        """
        b = []  # bin list
        c = []  # centers list
        s = []  # data list
        for nn in range(eventarray.n_series):
            s.append([])
        left_edges = []
        right_edges = []
        counter = 0
        for interval in intervalArray:
            bins, centers = self._get_bins_inside_interval(interval, ds)
            if bins is not None:
                for uu, eventarraydatas in enumerate(eventarray.data):
                    event_counts, _ = np.histogram(
                        eventarraydatas,
                        bins=bins,
                        density=False,
                        range=(interval.start,interval.stop)
                        ) # TODO: is it faster to limit range, or to cut out events?
                    s[uu].extend(event_counts.tolist())
                left_edges.append(counter)
                counter += len(centers) - 1
                right_edges.append(counter)
                counter += 1
                b.extend(bins.tolist())
                c.extend(centers.tolist())
        self._bins = np.array(b)
        self._bin_centers = np.array(c)
        self._data = np.array(s)
        le = np.array(left_edges)
        le = le[:, np.newaxis]
        re = np.array(right_edges)
        re = re[:, np.newaxis]
        self._binnedSupport = np.hstack((le, re))
        support_starts = self.bins[np.insert(np.cumsum(self.lengths+1),0,0)[:-1]]
        support_stops = self.bins[np.insert(np.cumsum(self.lengths+1)-1,0,0)[1:]]
        supportdata = np.vstack([support_starts, support_stops]).T
        self._abscissa.support = type(self._abscissa.support)(supportdata) # set support to TRUE bin support

    @keyword_deprecation(replace_x_with_y={'bw':'truncate'})
    def smooth(self, *, sigma=None, inplace=False,  truncate=None, within_intervals=False):
        """Smooth BinnedEventArray by convolving with a Gaussian kernel.

        Smoothing is applied in data, and the same smoothing is applied
        to each series in a BinnedEventArray.

        Smoothing is applied within each interval.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of Gaussian kernel, in seconds. Default is 0.01 (10 ms)
        truncate : float, optional
            Bandwidth outside of which the filter value will be zero. Default is 4.0
        inplace : bool
            If True the data will be replaced with the smoothed data.
            Default is False.

        Returns
        -------
        out : BinnedEventArray
            New BinnedEventArray with smoothed data.
        """

        if truncate is None:
            truncate = 4
        if sigma is None:
            sigma = 0.01 # 10 ms default

        fs = 1 / self.ds

        return utils.gaussian_filter(self, fs=fs, sigma=sigma, truncate=truncate, inplace=inplace, within_intervals=within_intervals)

    @staticmethod
    def _smooth_array(arr, w=None):
        """Smooth an array by convolving a boxcar, row-wise.

        Parameters
        ----------
        w : int, optional
            Number of bins to include in boxcar window. Default is 10.

        Returns
        -------
        smoothed: array
            Smoothed array with same shape as arr.
        """

        if w is None:
            w = 10

        if w==1: # perform no smoothing
            return arr

        w = np.min((w, arr.shape[1]))

        smoothed = arr.astype(float) # copy array and cast to float
        window = np.ones((w,))/w

        # smooth per row
        for rowi, row in enumerate(smoothed):
            smoothed[rowi,:] = np.convolve(row, window, mode='same')

        if arr.shape[1] != smoothed.shape[1]:
            raise TypeError("Incompatible shape returned!")

        return smoothed

    @staticmethod
    def _rebin_array(arr, w):
        """Rebin an array of shape (n_signals, n_bins) into a
        coarser bin size.

        Parameters
        ----------
        arr : array
            Array with shape (n_signals, n_bins) to re-bin. A copy
            is returned.
        w : int
            Number of original bins to combine into each new bin.

        Returns
        -------
        out : array
            Bnned array with shape (n_signals, n_new_bins)
        bin_idx : array
            Array of shape (n_new_bins,) with the indices of the new
            binned array, relative to the original array.
        """
        cs = np.cumsum(arr, axis=1)
        binidx = np.arange(start=w, stop=cs.shape[1]+1, step=w) - 1

        rebinned = np.hstack((np.array(cs[:,w-1], ndmin=2).T, cs[:,binidx[1:]] - cs[:,binidx[:-1]]))
        # bins = bins[np.insert(binidx+1, 0, 0)]
        return rebinned, binidx

    def rebin(self, w=None):
        """Rebin the BinnedEventArray into a coarser bin size.

        Parameters
        ----------
        w : int, optional
            number of bins of width bst.ds to bin into new bin of
            width bst.ds*w. Default is w=1 (no re-binning).

        Returns
        -------
        out : BinnedEventArray
            New BinnedEventArray with coarser resolution.
        """

        if w is None:
            w = 1

        if not float(w).is_integer:
            raise ValueError("w has to be an integer!")

        w = int(w)

        bst = self
        return self._rebin_binnedeventarray(bst, w=w)

    @staticmethod
    def _rebin_binnedeventarray(bst, w=None):
        """Rebin a BinnedEventArray into a coarser bin size.

        Parameters
        ----------
        bst : BinnedEventArray
            BinnedEventArray to re-bin into a coarser resolution.
        w : int, optional
            number of bins of width bst.ds to bin into new bin of
            width bst.ds*w. Default is w=1 (no re-binning).

        Returns
        -------
        out : BinnedEventArray
            New BinnedEventArray with coarser resolution.

        # FFB! TODO: if w is longer than some event size,
        # an exception will occur. Handle it! Although I may already
        # implicitly do that.
        """

        if w is None:
            w = 1

        if w == 1:
            return bst

        edges = np.insert(np.cumsum(bst.lengths), 0, 0)
        newlengths = [0]
        binedges = np.insert(np.cumsum(bst.lengths+1), 0, 0)
        n_events = bst.support.n_intervals
        new_centers = []

        newdata = None

        for ii in range(n_events):
            data = bst.data[:,edges[ii]:edges[ii+1]]
            bins = bst.bins[binedges[ii]:binedges[ii+1]]

            datalen = data.shape[1]
            if w <= datalen:
                rebinned, binidx = bst._rebin_array(data, w=w)
                bins = bins[np.insert(binidx+1, 0, 0)]

                newlengths.append(rebinned.shape[1])

                if newdata is None:
                    newdata = rebinned
                    newbins = bins
                    newcenters = bins[:-1] + np.diff(bins) / 2
                    newsupport = np.array([bins[0], bins[-1]])
                else:
                    newdata = np.hstack((newdata, rebinned))
                    newbins = np.hstack((newbins, bins))
                    newcenters = np.hstack((newcenters, bins[:-1] + np.diff(bins) / 2))
                    newsupport = np.vstack((newsupport, np.array([bins[0], bins[-1]])))
            else:
                pass

        # assemble new binned event series array:
        newedges = np.cumsum(newlengths)
        newbst = bst._copy_without_data()
        abscissa = copy.copy(bst._abscissa)
        if newdata is not None:
            newbst._data = newdata
            newbst._abscissa = abscissa
            newbst._abscissa.support = type(bst.support)(newsupport)
            newbst._bins = newbins
            newbst._bin_centers = newcenters
            newbst._ds = bst.ds*w
            newbst._binnedSupport = np.array((newedges[:-1], newedges[1:]-1)).T
        else:
            logging.warning("No events are long enough to contain any bins of width {}".format(utils.PrettyDuration(ds)))
            newbst._data = None
            newbst._abscissa = abscissa
            newbst._abscissa.support = None
            newbst._binnedSupport = None
            newbst._bin_centers = None
            newbst._bins = None

        newbst.__renew__()

        return newbst

    def bst_from_indices(self, idx):
        """
        Return a BinnedEventArray from a list of indices.

        bst : BinnedEventArray
        idx : list of sample (bin) numbers with shape (n_intervals, 2) INCLUSIVE

        Example
        =======
        idx = [[10, 20]
            [25, 50]]
        bst_from_indices(bst, idx=idx)
        """

        idx = np.atleast_2d(idx)

        newbst = self._copy_without_data()
        ds = self.ds
        bin_centers_ = []
        bins_ = []
        binnedSupport_ = []
        support_ = []
        all_abscissa_vals = []

        n_preceding_bins = 0

        for frm, to in idx:
            idx_array = np.arange(frm, to+1).astype(int)
            all_abscissa_vals.append(idx_array)
            bin_centers = self.bin_centers[idx_array]
            bins = np.append(bin_centers - ds/2, bin_centers[-1] + ds/2)

            binnedSupport = [n_preceding_bins, n_preceding_bins + len(bins)-2]
            n_preceding_bins += len(bins)-1
            support = type(self._abscissa.support)((bins[0], bins[-1]))

            bin_centers_.append(bin_centers)
            bins_.append(bins)
            binnedSupport_.append(binnedSupport)
            support_.append(support)

        bin_centers = np.concatenate(bin_centers_)
        bins = np.concatenate(bins_)
        binnedSupport = np.array(binnedSupport_)
        support = np.sum(support_)
        all_abscissa_vals = np.concatenate(all_abscissa_vals)

        newbst._bin_centers = bin_centers
        newbst._bins = bins
        newbst._binnedSupport = binnedSupport
        newbst._abscissa.support = support
        newbst._data = newbst.data[:,all_abscissa_vals]

        newbst.__renew__()

        return newbst

    @property
    def n_active(self):
        """Number of active series.

        An active series is any series that fired at least one event.
        """
        if self.isempty:
            return 0
        return utils.PrettyInt(np.count_nonzero(self.n_events))

    @property
    def n_active_per_bin(self):
        """Number of active series per data bin with shape (n_bins,)."""
        if self.isempty:
            return 0
        # TODO: profile several alternatves. Could use data > 0, or
        # other numpy methods to get a more efficient implementation:
        return self.data.clip(max=1).sum(axis=0)

    @property
    def n_events(self):
        """(np.array) The number of events in each series."""
        if self.isempty:
            return 0
        return self.data.sum(axis=1)

    def flatten(self, *, series_id=None, series_label=None):
        """Collapse events across series.

        WARNING! series_tags are thrown away when flattening.

        Parameters
        ----------
        series_id: (int)
            (series) ID to assign to flattened event series, default is 0.
        series_label (str)
            (series) Label for event series, default is 'flattened'.
        """
        if self.n_series < 2:  # already flattened
            return self

        # default args:
        if series_id is None:
            series_id = 0
        if series_label is None:
            series_label = "flattened"

        binnedeventarray = self._copy_without_data()

        binnedeventarray._data = np.array(self.data.sum(axis=0), ndmin=2)

        binnedeventarray._bins = self.bins
        binnedeventarray._abscissa.support = self.support
        binnedeventarray._bin_centers = self.bin_centers
        binnedeventarray._binnedSupport = self.binnedSupport

        binnedeventarray._series_ids = [series_id]
        binnedeventarray._series_labels = [series_label]
        binnedeventarray._series_tags = None
        binnedeventarray.__renew__()

        return binnedeventarray

    @property
    def support(self):
        """(nelpy.IntervalArray) The support of the underlying BinnedEventArray."""
        return self._abscissa.support

    @support.setter
    def support(self, val):
        """(nelpy.IntervalArray) The support of the underlying BinnedEventArray."""
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
        self._restrict_to_interval(self._abscissa.support)

def legacySTAkwargs(**kwargs):
    """Provide support for legacy SpikeTrainArray
    kwargs. This function is primarily intended to be
    a helper for the new STA constructor, not for
    general-purpose use.

    kwarg: time        <==> timestamps <==> abscissa_vals
    kwarg: data        <==> ydata
    kwarg: unit_ids    <==> series_ids
    kwarg: unit_labels <==> series_labels
    kwarg: unit_tags   <==> series_tags

    Examples
    --------
    sta = nel.SpikeTrainArray(time=..., )
    sta = nel.SpikeTrainArray(timestamps=..., )
    sta = nel.SpikeTrainArray(abscissa_vals=..., )
    """

    def only_one_of(*args):
        num_non_null_args = 0
        out = None
        for arg in args:
            if arg is not None:
                num_non_null_args += 1
                out = arg
        if num_non_null_args > 1:
            raise ValueError ('multiple conflicting arguments received')
        return out

    # legacy STA constructor support for backward compatibility
    abscissa_vals = kwargs.pop('abscissa_vals', None)
    timestamps = kwargs.pop('timestamps', None)
    time = kwargs.pop('time', None)
    # only one of the above, otherwise raise exception
    abscissa_vals = only_one_of(abscissa_vals, timestamps, time)
    if abscissa_vals is not None:
        kwargs['abscissa_vals'] = abscissa_vals

    # Other legacy attributes
    series_ids = kwargs.pop('series_ids', None)
    unit_ids = kwargs.pop('unit_ids', None)
    series_ids = only_one_of(series_ids, unit_ids)
    kwargs['series_ids'] = series_ids

    series_labels = kwargs.pop('series_labels', None)
    unit_labels = kwargs.pop('unit_labels', None)
    series_labels = only_one_of(series_labels, unit_labels)
    kwargs['series_labels'] = series_labels

    series_tags = kwargs.pop('series_tags', None)
    unit_tags = kwargs.pop('unit_tags', None)
    series_tags = only_one_of(series_tags, unit_tags)
    kwargs['series_tags'] = series_tags

    return kwargs

########################################################################
# class SpikeTrainArray
########################################################################
class SpikeTrainArray(EventArray):
    """Custom SpikeTrainArray docstring with kwarg descriptions.

    TODO: add docstring here, using the aliases in the constructor.
    """

    # specify class-specific aliases:
    __aliases__ = {
        'time': 'data',
        '_time': '_data',
        'n_epochs': 'n_intervals',
        'n_units' : 'n_series',
        '_unit_subset' : '_series_subset', # requires kw change
        'get_event_firing_order' : 'get_spike_firing_order',
        'reorder_units_by_ids' : 'reorder_series_by_ids',
        'reorder_units' : 'reorder_series',
        '_reorder_units_by_idx' : '_reorder_series_by_idx',
        'n_spikes' : 'n_events',
        'unit_ids' : 'series_ids',
        'unit_labels': 'series_labels',
        'unit_tags': 'series_tags',
        '_unit_ids' : '_series_ids',
        '_unit_labels': '_series_labels',
        '_unit_tags': '_series_tags',
        'first_spike': 'first_event',
        'last_spike': 'last_event',
        }

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}

        series_label = kwargs.pop('series_label', None)
        if series_label is None:
            series_label = 'units'
        kwargs['series_label'] = series_label

        # legacy STA constructor support for backward compatibility
        kwargs = legacySTAkwargs(**kwargs)

        support = kwargs.get('support', None)
        if support is not None:
            abscissa = kwargs.get('abscissa', core.TemporalAbscissa(support=support))
        else:
            abscissa = kwargs.get('abscissa', core.TemporalAbscissa())
        ordinate = kwargs.get('ordinate', core.AnalogSignalArrayOrdinate())

        kwargs['abscissa'] = abscissa
        kwargs['ordinate'] = ordinate

        super().__init__(*args, **kwargs)

    # @keyword_equivalence(this_or_that={'n_intervals':'n_epochs'})
    # def partition(self, ds=None, n_intervals=None, n_epochs=None):
    #     if n_intervals is None:
    #         n_intervals = n_epochs
    #     kwargs = {'ds':ds, 'n_intervals': n_intervals}
    #     return super().partition(**kwargs)

    def bin(self, *, ds=None):
        """Return a BinnedSpikeTrainArray."""
        return BinnedSpikeTrainArray(self, ds=ds) # TODO #FIXME

########################################################################
# class BinnedSpikeTrainArray
########################################################################
class BinnedSpikeTrainArray(BinnedEventArray):
    """Custom SpikeTrainArray docstring with kwarg descriptions.

    TODO: add docstring here, using the aliases in the constructor.
    """

    # specify class-specific aliases:
    __aliases__ = {
        'time': 'data',
        '_time': '_data',
        'n_epochs': 'n_intervals',
        'n_units' : 'n_series',
        '_unit_subset' : '_series_subset', # requires kw change
        # 'get_event_firing_order' : 'get_spike_firing_order'
        'reorder_units_by_ids' : 'reorder_series_by_ids',
        'reorder_units' : 'reorder_series',
        '_reorder_units_by_idx' : '_reorder_series_by_idx',
        'n_spikes' : 'n_events',
        'unit_ids' : 'series_ids',
        'unit_labels': 'series_labels',
        'unit_tags': 'series_tags',
        '_unit_ids' : '_series_ids',
        '_unit_labels': '_series_labels',
        '_unit_tags': '_series_tags'
        }

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}

        support = kwargs.get('support', None)
        if support is not None:
            abscissa = kwargs.get('abscissa', core.TemporalAbscissa(support=support))
        else:
            abscissa = kwargs.get('abscissa', core.TemporalAbscissa())
        ordinate = kwargs.get('ordinate', core.AnalogSignalArrayOrdinate())

        kwargs['abscissa'] = abscissa
        kwargs['ordinate'] = ordinate

        super().__init__(*args, **kwargs)

    # def partition(self, ds=None, n_intervals=None, n_epochs=None):
    #     if n_intervals is None:
    #         n_intervals = n_epochs
    #     kwargs = {'ds':ds, 'n_intervals': n_intervals}
    #     return super().partition(**kwargs)
