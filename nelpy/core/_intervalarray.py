__all__ = ['IntervalArray', 'EpochArray', 'SpaceArray']

import logging
import numpy as np
import copy
import numbers
from numba import jit

from sys import float_info

from .. import formatters
from .. import utils
from .. import version

from ..utils_.decorators import keyword_equivalence

########################################################################
# class IntervalArray
########################################################################
class IntervalArray:
    """An array of intervals, where each interval has a start and stop.

    Parameters
    ----------
    data : np.array
        If shape (n_intervals, 1) or (n_intervals,), the start value for each
        interval (which then requires a length to be specified).
        If shape (n_intervals, 2), the start and stop values for each interval.
    length : np.array, float, or None, optional
        The length of the interval (in base units). If (float) then the same
        length is assumed for every interval.
    meta : dict, optional
        Metadata associated with spiketrain.
    domain : IntervalArray ??? This is pretty meta @-@

    Attributes
    ----------
    data : np.array
        The start and stop values for each interval. With shape (n_intervals, 2).
    """

    __aliases__ = {}
    __attributes__ = ["_data", "_meta", "_domain"]

    def __init__(self, data=None, *args, length=None,
                 meta=None, empty=False, domain=None, label=None):

        self.__version__ = version.__version__

        self.type_name = self.__class__.__name__
        self._interval_label = 'interval'
        self.formatter = formatters.ArbitraryFormatter
        self.base_unit = self.formatter.base_unit

        if len(args) > 1:
            raise TypeError("__init__() takes from 1 to 3 positional arguments but 4 were given")
        elif len(args) == 1:
            data = [data, args[0]]

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        data = np.squeeze(data)  # coerce data into np.array

        # all possible inputs:
        # 1. single interval, no length    --- OK
        # 2. single interval and length    --- ERR
        # 3. multiple intervals, no length --- OK
        # 4. multiple intervals and length --- ERR
        # 5. single scalar and length   --- OK
        # 6. scalar list and duratin list --- OK
        #
        # Q. won't np.squeeze make our life difficult?
        #
        # Strategy: determine if length was passed. If so, try to see
        # if data can be coerced into right shape. If not, raise
        # error.
        # If length was NOT passed, then do usual checks for intervals.

        if length is not None:  # assume we received scalar starts
            data = np.array(data, ndmin=1)
            length = np.squeeze(length).astype(float)
            if length.ndim == 0:
                length = length[..., np.newaxis]

            if data.ndim == 2 and length.ndim == 1:
                raise ValueError(
                    "length not allowed when using start and stop "
                    "values")

            if len(length) > 1:
                if data.ndim == 1 and data.shape[0] != length.shape[0]:
                    raise ValueError(
                        "must have same number of data and length "
                        "data"
                        )
            if data.ndim == 1 and length.ndim == 1:
                stop_interval = data + length
                data = np.hstack(
                    (data[..., np.newaxis], stop_interval[..., np.newaxis]))
        else:  # length was not specified, so assume we recived intervals

            # Note: if we have an empty array of data with no
            # dimension, then calling len(data) will return a
            # TypeError.
            try:
                # if no data were received, return an empty IntervalArray:
                if len(data) == 0:
                    self.__init__(empty=True)
                    return
            except TypeError:
                logging.warning("unsupported type ("
                    + str(type(data))
                    + "); creating empty {}".format(self.type_name))
                self.__init__(empty=True)
                return

            # Only one interval is given eg IntervalArray([3,5,6,10]) with no
            # length and more than two values:
            if data.ndim == 1 and len(data) > 2:  # we already know length is None
                raise TypeError(
                    "data of size (n_intervals, ) has to be accompanied by "
                    "a length")

            if data.ndim == 1:  # and length is None:
                data = np.array([data])

        if data.ndim > 2:
            raise ValueError("data must be a 1D or a 2D vector")

        try:
            if data[:, 0].shape[0] != data[:, 1].shape[0]:
                raise ValueError(
                    "must have the same number of start and stop values")
        except Exception:
            raise Exception("Unhandled {}.__init__ case.".format(self.type_name))

        # TODO: what if start == stop? what will this break? This situation
        # can arise automatically when slicing a spike train with one or no
        # spikes, for example in which case the automatically inferred support
        # is a delta dirac

        if data.ndim == 2 and np.any(data[:, 1] - data[:, 0] < 0):
            raise ValueError("start must be less than or equal to stop")

        # potentially assign domain
        self._domain = domain

        self._data = data
        self._meta = meta
        self.label = label

        if not self.issorted:
            self._sort()

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty " + self.type_name + address_str + ">"
        if self.n_intervals > 1:
            nstr = "%s %ss" % (self.n_intervals, self._interval_label)
        else:
            nstr = "1 %s" % self._interval_label
        dstr = "of length {}".format(self.formatter(self.length))
        return "<%s%s: %s> %s" % (self.type_name, address_str, nstr, dstr)

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

    def _copy_without_data(self):
        """Return a copy of self, without data."""
        out = copy.copy(self) # shallow copy
        out._data = np.zeros((self.n_intervals, 2))
        out = copy.deepcopy(out) # just to be on the safe side, but at least now we are not copying the data!
        return out

    def __iter__(self):
        """IntervalArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """IntervalArray iterator advancer."""
        index = self._index
        if index > self.n_intervals - 1:
            raise StopIteration

        intervalarray = self._copy_without_data()
        intervalarray._data = np.array([self.data[index, :]])
        self._index += 1
        return intervalarray

    def __getitem__(self, *idx):
        """IntervalArray index access.

        Accepts integers, slices, and IntervalArrays.
        """
        if self.isempty:
            return self

        idx = [ii for ii in idx]
        if len(idx) == 1 and not isinstance(idx[0], int):
            idx = idx[0]
        if isinstance(idx, tuple):
            idx = [ii for ii in idx]

        if isinstance(idx, type(self)):
            if idx.isempty:  # case 0:
                return type(self)(empty=True)
            return self.intersect(interval=idx)
        elif isinstance(idx, IntervalArray):
            raise TypeError("Error taking intersection. {} expected, but got {}".format(self.type_name, idx.type_name))
        else:
            try: # works for ints, lists, and slices
                out = self.copy()
                out._data = self.data[idx,:]
            except IndexError:
                raise IndexError('{} index out of range'.format(self.type_name))
            except Exception:
                raise TypeError(
                    'unsupported subscripting type {}'.format(type(idx)))
        return out

    def __add__(self, other):
        """add length to start and stop of each interval, or join two interval arrays without merging"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            return new.expand(other, direction='both')
        elif isinstance(other, type(self)):
            return self.join(other)
        else:
            raise TypeError("unsupported operand type(s) for +: {} and {}".format(str(type(self)), str(type(other))))

    def __sub__(self, other):
        """subtract length from start and stop of each interval"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            return new.shrink(other, direction='both')
        elif isinstance(other, type(self)):
            # A - B = A intersect ~B
            return self.intersect(~other)
        else:
            raise TypeError("unsupported operand type(s) for +: {} and {}".format(str(type(self)), str(type(other))))

    def __mul__(self, other):
        """expand (>1) or shrink (<1) interval lengths"""
        raise NotImplementedError("operator * not yet implemented")

    def __truediv__(self, other):
        """expand (>1) or shrink (>1) interval lengths"""
        raise NotImplementedError("operator / not yet implemented")

    def __lshift__(self, other):
        """shift data to left (<<)"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            new._data = new._data - other
            if new.domain.is_finite:
                new.domain._data = new.domain._data - other
            return new
        else:
            raise TypeError("unsupported operand type(s) for <<: {} and {}".format(str(type(self)), str(type(other))))

    def __rshift__(self, other):
        """shift data to right (>>)"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            new._data = new._data + other
            if new.domain.is_finite:
                new.domain._data = new.domain._data + other
            return new
        else:
            raise TypeError("unsupported operand type(s) for >>: {} and {}".format(str(type(self)), str(type(other))))

    def __and__(self, other):
        """intersection of interval arrays"""
        if isinstance(other, type(self)):
            new = copy.copy(self)
            return new.intersect(other, boundaries=True)
        else:
            raise TypeError("unsupported operand type(s) for &: {} and {}".format(str(type(self)), str(type(other))))

    def __or__(self, other):
        """join and merge interval array; set union"""
        if isinstance(other, type(self)):
            new = copy.copy(self)
            joined = new.join(other)
            union = joined.merge()
            return union
        else:
            raise TypeError("unsupported operand type(s) for |: {} and {}".format(str(type(self)), str(type(other))))

    def __invert__(self):
        """complement within self.domain"""
        return self.complement()

    def __bool__(self):
        """(bool) Empty IntervalArray"""
        return not self.isempty

    def remove_duplicates(self, inplace=False):
        """Remove duplicate intervals."""
        raise NotImplementedError

    @keyword_equivalence(this_or_that={'n_intervals':'n_epochs'})
    def partition(self, *, ds=None, n_intervals=None):
        """Returns an IntervalArray that has been partitioned.

        # Irrespective of whether 'ds' or 'n_intervals' are used, the exact
        # underlying support is propagated, and the first and last points
        # of the supports are always included, even if this would cause
        # n_points or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Maximum length, for each interval.
        n_points : int, optional
            Number of intervals. If ds is None and n_intervals is None, then
            default is to use n_intervals = 100

        Returns
        -------
        out : IntervalArray
            IntervalArray that has been partitioned.
        """

        if self.isempty:
            raise ValueError ("cannot parition an empty object in a meaningful way!")

        if ds is not None and n_intervals is not None:
            raise ValueError("ds and n_intervals cannot be used together")

        if n_intervals is not None:
            assert float(n_intervals).is_integer(), "n_intervals must be a positive integer!"
            assert n_intervals > 1, "n_intervals must be a positive integer > 1"
            # determine ds from number of desired points:
            ds = self.length / n_intervals

        if ds is None:
            # neither n_intervals nor ds was specified, so assume defaults:
            n_intervals = 100
            ds = self.length / n_intervals

        # build list of points at which to esplit the IntervalArray
        new_starts = []
        new_stops = []
        for start, stop in self.data:
            newxvals = utils.frange(start, stop, step=ds).tolist()
            # newxvals = np.arange(start, stop, step=ds).tolist()
            if newxvals[-1] + float_info.epsilon < stop:
                newxvals.append(stop)
            newxvals = np.asanyarray(newxvals)
            new_starts.extend(newxvals[:-1])
            new_stops.extend(newxvals[1:])

        # now make a new interval array:
        out = copy.copy(self)
        out._data = np.hstack(
                [np.array(new_starts)[..., np.newaxis],
                 np.array(new_stops)[..., np.newaxis]])
        return out

    @property
    def label(self):
        """Label describing the interval array."""
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

    def complement(self, domain=None):
        """Complement within domain.

        Parameters
        ----------
        domain : IntervalArray, optional
            IntervalArray specifying entire domain. Default is self.domain.

        Returns
        -------
        complement : IntervalArray
            IntervalArray containing all the nonzero intervals in the
            complement set.
        """

        if domain is None:
            domain = self.domain

        # make sure IntervalArray is sorted:
        if not self.issorted:
            self._sort()
        # check that IntervalArray is entirely contained within domain
        # if (self.start < domain.start) or (self.stop > domain.stop):
        #     raise ValueError("IntervalArray must be entirely contained within domain")

        # check that IntervalArray is fully merged, or merge it if necessary
        merged = self.merge()
        # build complement intervals
        starts = np.insert(merged.stops, 0 , domain.start)
        stops = np.append(merged.starts, domain.stop)
        newvalues = np.vstack([starts, stops]).T
        # remove intervals with zero length
        lengths = newvalues[:,1] - newvalues[:,0]
        newvalues = newvalues[lengths>0]
        complement = copy.copy(self)
        complement._data = newvalues

        if domain.n_intervals > 1:
            return complement[domain]
        try:
            complement._data[0,0] = np.max((complement._data[0,0], domain.start))
            complement._data[-1,-1] = np.min((complement._data[-1,-1], domain.stop))
        except IndexError: # complement is empty
            return type(self)(empty=True)
        return complement

    @property
    def domain(self):
        """domain (in base units) within which support is defined"""
        if self._domain is None:
            self._domain = type(self)([-np.inf, np.inf])
        return self._domain

    @domain.setter
    def domain(self, val):
        """domain (in base units) within which support is defined"""
        #TODO: add  input validation
        if isinstance(val, type(self)):
            self._domain = val
        elif isinstance(val, (tuple, list)):
            self._domain = type(self)([val[0], val[1]])

    @property
    def meta(self):
        """Meta data associated with IntervalArray."""
        if self._meta is None:
            logging.warning("meta data is not available")
        return self._meta

    @meta.setter
    def meta(self, val):
        self._meta = val

    @property
    def min(self):
        """Minimum bound of all intervals in IntervalArray."""
        return self.merge().start

    @property
    def max(self):
        """Maximum bound of all intervals in IntervalArray."""
        return self.merge().stop

    @property
    def data(self):
        """Interval values [start, stop) in base units."""
        return self._data

    @property
    def is_finite(self):
        """Is the interval [start, stop) finite."""
        return not(np.isinf(self.start) | np.isinf(self.stop))

    # @property
    # def _human_readable_posix_intervals(self):
    #     """Interval start and stop values in human readable POSIX time.

    #     This property is left private, because it has not been carefully
    #     vetted for public API release yet.
    #     """
    #     import datetime
    #     n_intervals_zfill = len(str(self.n_intervals))
    #     for ii, (start, stop) in enumerate(self.time):
    #         print('[ep ' + str(ii).zfill(n_intervals_zfill) + ']\t' +
    #               datetime.datetime.fromtimestamp(
    #                 int(start)).strftime('%Y-%m-%d %H:%M:%S') + ' -- ' +
    #               datetime.datetime.fromtimestamp(
    #                 int(stop)).strftime('%Y-%m-%d %H:%M:%S') + '\t(' +
    #               str(utils.PrettyDuration(stop-start)) + ')')

    @property
    def centers(self):
        """(np.array) The center of each interval."""
        if self.isempty:
            return []
        return np.mean(self.data, axis=1)

    @property
    def lengths(self):
        """(np.array) The length of each interval."""
        if self.isempty:
            return 0
        return self.data[:, 1] - self.data[:, 0]

    @property
    def range(self):
        """return IntervalArray containing range of current IntervalArray."""
        return type(self)([self.start, self.stop])

    @property
    def length(self):
        """(float) The total length of the [merged] interval array."""
        if self.isempty:
            return self.formatter(0)
        merged = self.merge()
        return self.formatter(np.array(merged.data[:, 1] - merged.data[:, 0]).sum())

    @property
    def starts(self):
        """(np.array) The start of each interval."""
        if self.isempty:
            return []
        return self.data[:, 0]

    @property
    def start(self):
        """(np.array) The start of the first interval."""
        if self.isempty:
            return []
        return self.data[:, 0][0]

    @property
    def stops(self):
        """(np.array) The stop of each interval."""
        if self.isempty:
            return []
        return self.data[:, 1]

    @property
    def stop(self):
        """(np.array) The stop of the last interval."""
        if self.isempty:
            return []
        return self.data[:, 1][-1]

    @property
    def n_intervals(self):
        """(int) The number of intervals."""
        if self.isempty:
            return 0
        return utils.PrettyInt(len(self.data[:, 0]))

    def __len__(self):
        """(int) The number of intervals."""
        return self.n_intervals

    @property
    def ismerged(self):
        """(bool) No overlapping intervals exist."""
        if self.isempty:
            return True
        if self.n_intervals == 1:
            return True
        if not self.issorted:
            self._sort()
        if not utils.is_sorted(self.stops):
            return False

        return np.all(self.data[1:,0] - self.data[:-1,1] > 0)

    def _ismerged(self, overlap=0.0):
        """(bool) No overlapping intervals with overlap >= overlap exist."""
        if self.isempty:
            return True
        if self.n_intervals == 1:
            return True
        if not self.issorted:
            self._sort()
        if not utils.is_sorted(self.stops):
            return False

        return np.all(self.data[1:,0] - self.data[:-1,1] > -overlap)

    @property
    def issorted(self):
        """(bool) Left edges of intervals are sorted in ascending order."""
        if self.isempty:
            return True
        return utils.is_sorted(self.starts)

    @property
    def isempty(self):
        """(bool) Empty IntervalArray."""
        try:
            return len(self.data) == 0
        except TypeError:
            return True  # this happens when self.data is None

    def copy(self):
        """(IntervalArray) Returns a copy of the current interval array."""
        newcopy = copy.deepcopy(self)
        return newcopy

    def _drop_empty_intervals(self):
        """Drops empty intervals. Not in-place, i.e. returns a copy."""
        keep_interval_ids = np.argwhere(self.lengths).squeeze().tolist()
        return self[keep_interval_ids]
    

    def intersect(self, interval, *, boundaries=True):
        """Returns intersection (overlap) between current IntervalArray (self) and
        other interval array ('interval').
        """

        if self.isempty or interval.isempty:
            logging.warning('interval intersection is empty')
            return type(self)(empty=True)

        new_intervals = []

        # Extract starts and stops and convert to np.array of float64 (for numba)
        interval_starts_a = np.array(self.starts, dtype=np.float64)
        interval_stops_a = np.array(self.stops, dtype=np.float64)
        if interval.data.ndim == 1:
            interval_starts_b = np.array([interval.data[0]], dtype=np.float64)
            interval_stops_b = np.array([interval.data[1]], dtype=np.float64)
        else:
            interval_starts_b = np.array(interval.data[:,0], dtype=np.float64)
            interval_stops_b = np.array(interval.data[:,1], dtype=np.float64)

        new_starts, new_stops = interval_intersect(
            interval_starts_a,
            interval_stops_a,
            interval_starts_b,
            interval_stops_b,
            boundaries,
        )

        for start, stop in zip(new_starts, new_stops):
            new_intervals.append([start, stop])

        # convert to np.array of float64
        new_intervals = np.array(new_intervals, dtype=np.float64)

        out = type(self)(new_intervals)
        out._domain = self.domain
        return out
            
    # def intersect(self, interval, *, boundaries=True):
    #     """Returns intersection (overlap) between current IntervalArray (self) and
    #        other interval array ('interval').
    #     """

    #     this = copy.deepcopy(self)
    #     new_intervals = []
    #     for epa in this:
    #         cand_ep_idx = np.argwhere((interval.starts < epa.stop) & (interval.stops > epa.start)).squeeze()
    #         if np.size(cand_ep_idx) > 0:
    #             for epb in interval[cand_ep_idx.tolist()]:
    #                 new_interval = self._intersect(epa, epb, boundaries=boundaries)
    #                 if not new_interval.isempty:
    #                     new_intervals.append([new_interval.start, new_interval.stop])
    #     out = type(self)(new_intervals)
    #     out._domain = self.domain
    #     return out

    # def _intersect(self, intervala, intervalb, *, boundaries=True, meta=None):
    #     """Finds intersection (overlap) between two sets of interval arrays.

    #     TODO: verify if this requires a merged IntervalArray to work properly?
    #     ISSUE_261: not fixed yet

    #     TODO: domains are not preserved yet! careful consideration is necessary.

    #     Parameters
    #     ----------
    #     interval : nelpy.IntervalArray
    #     boundaries : bool
    #         If True, limits start, stop to interval start and stop.
    #     meta : dict, optional
    #         New dictionary of meta data for interval ontersection.

    #     Returns
    #     -------
    #     intersect_intervals : nelpy.IntervalArray
    #     """
    #     if intervala.isempty or intervalb.isempty:
    #         logging.warning('interval intersection is empty')
    #         return type(self)(empty=True)

    #     new_starts = []
    #     new_stops = []
    #     interval_a = intervala.merge().copy()
    #     interval_b = intervalb.merge().copy()

    #     for aa in interval_a.data:
    #         for bb in interval_b.data:
    #             if (aa[0] <= bb[0] < aa[1]) and (aa[0] < bb[1] <= aa[1]):
    #                 new_starts.append(bb[0])
    #                 new_stops.append(bb[1])
    #             elif (aa[0] < bb[0] < aa[1]) and (aa[0] < bb[1] > aa[1]):
    #                 new_starts.append(bb[0])
    #                 if boundaries:
    #                     new_stops.append(aa[1])
    #                 else:
    #                     new_stops.append(bb[1])
    #             elif (aa[0] > bb[0] < aa[1]) and (aa[0] < bb[1] < aa[1]):
    #                 if boundaries:
    #                     new_starts.append(aa[0])
    #                 else:
    #                     new_starts.append(bb[0])
    #                 new_stops.append(bb[1])
    #             elif (aa[0] >= bb[0] < aa[1]) and (aa[0] < bb[1] >= aa[1]):
    #                 if boundaries:
    #                     new_starts.append(aa[0])
    #                     new_stops.append(aa[1])
    #                 else:
    #                     new_starts.append(bb[0])
    #                     new_stops.append(bb[1])

    #     if not boundaries:
    #         new_starts = np.unique(new_starts)
    #         new_stops = np.unique(new_stops)

    #     interval_a._data = np.hstack(
    #         [np.array(new_starts)[..., np.newaxis],
    #             np.array(new_stops)[..., np.newaxis]])

    #     return interval_a

    def merge(self, *, gap=0.0, overlap=0.0):
        """Merge intervals that are close or overlapping.

        if gap == 0 and overlap == 0:
            [a, b) U [b, c) = [a, c)
        if gap == None and overlap > 0:
            [a, b) U [b, c) = [a, b) U [b, c)
            [a, b + overlap) U [b, c) = [a, c)
            [a, b) U [b - overlap, c) = [a, c)
        if gap > 0 and overlap == None:
            [a, b) U [b, c) = [a, c)
            [a, b) U [b + gap, c) = [a, c)
            [a, b - gap) U [b, c) = [a, c)

        WARNING! Algorithm only works on SORTED intervals.

        Parameters
        ----------
        gap : float, optional
            Amount (in base units) to consider intervals close enough to merge.
            Defaults to 0.0 (no gap).
        Returns
        -------
        merged_intervals : nelpy.IntervalArray
        """

        if gap < 0:
            raise ValueError("gap cannot be negative")
        if overlap < 0:
            raise ValueError("overlap cannot be negative")

        if self.isempty:
            return self

        if (self.ismerged) and (gap==0.0):
            # already merged
            return self

        newintervalarray = copy.copy(self)

        if not newintervalarray.issorted:
            newintervalarray._sort()

        overlap_ = overlap

        while not newintervalarray._ismerged(overlap=overlap) or gap>0:
            stops = newintervalarray.stops[:-1] + gap
            starts = newintervalarray.starts[1:] + overlap_
            to_merge = (stops - starts) >= 0

            new_starts = [newintervalarray.starts[0]]
            new_stops = []

            next_stop = newintervalarray.stops[0]
            for i in range(newintervalarray.data.shape[0] - 1):
                this_stop = newintervalarray.stops[i]
                next_stop = max(next_stop, this_stop)
                if not to_merge[i]:
                    new_stops.append(next_stop)
                    new_starts.append(newintervalarray.starts[i + 1])

            new_stops.append(max(newintervalarray.stops[-1], next_stop))

            new_starts = np.array(new_starts)
            new_stops = np.array(new_stops)

            newintervalarray._data = np.vstack([new_starts, new_stops]).T

            # after one pass, all the gap offsets have been added, and
            # then we just need to keep merging...
            gap = 0.0
            overlap_ = 0.0

        return newintervalarray

    def expand(self, amount, direction='both'):
        """Expands interval by the given amount.
        Parameters
        ----------
        amount : float
            Amount (in base units) to expand each interval.
        direction : str
            Can be 'both', 'start', or 'stop'. This specifies
            which direction to resize interval.
        Returns
        -------
        expanded_intervals : nelpy.IntervalArray
        """
        if direction == 'both':
            resize_starts = self.data[:, 0] - amount
            resize_stops = self.data[:, 1] + amount
        elif direction == 'start':
            resize_starts = self.data[:, 0] - amount
            resize_stops = self.data[:, 1]
        elif direction == 'stop':
            resize_starts = self.data[:, 0]
            resize_stops = self.data[:, 1] + amount
        else:
            raise ValueError(
                "direction must be 'both', 'start', or 'stop'")

        newintervalarray = copy.copy(self)

        newintervalarray._data = np.hstack((
                resize_starts[..., np.newaxis],
                resize_stops[..., np.newaxis]
                ))

        return newintervalarray

    def shrink(self, amount, direction='both'):
        """Shrinks interval by the given amount.
        Parameters
        ----------
        amount : float
            Amount (in base units) to shrink each interval.
        direction : str
            Can be 'both', 'start', or 'stop'. This specifies
            which direction to resize interval.
        Returns
        -------
        shrinked_intervals : nelpy.IntervalArray
        """
        both_limit = min(self.lengths / 2)
        if amount > both_limit and direction == 'both':
            raise ValueError("shrink amount too large")

        single_limit = min(self.lengths)
        if amount > single_limit and direction != 'both':
            raise ValueError("shrink amount too large")

        return self.expand(-amount, direction)

    def join(self, interval, meta=None):
        """Combines [and merges] two sets of intervals. Intervals can have
        different sampling rates.

        Parameters
        ----------
        interval : nelpy.IntervalArray
        meta : dict, optional
            New meta data dictionary describing the joined intervals.

        Returns
        -------
        joined_intervals : nelpy.IntervalArray
        """

        if self.isempty:
            return interval
        if interval.isempty:
            return self

        newintervalarray = copy.copy(self)

        join_starts = np.concatenate(
            (self.data[:, 0], interval.data[:, 0]))
        join_stops = np.concatenate(
            (self.data[:, 1], interval.data[:, 1]))

        newintervalarray._data = np.hstack((
            join_starts[..., np.newaxis],
            join_stops[..., np.newaxis]
            ))
        if not newintervalarray.issorted:
            newintervalarray._sort()
        # if not newintervalarray.ismerged:
        #     newintervalarray = newintervalarray.merge()
        return newintervalarray

    def __contains__(self, value):
        """Checks whether value is in any interval.

        #TODO: add support for when value is an IntervalArray (could be easy with intersection)

        Parameters
        ----------
        intervals: nelpy.IntervalArray
        value: float or int

        Returns
        -------
        boolean

        """
        # TODO: consider vectorizing this loop, which should increase
        # speed, but also greatly increase memory? Alternatively, if we
        # could assume something about intervals being sorted, this can
        # also be made much faster than the current O(N)
        for start, stop in zip(self.starts, self.stops):
            if start <= value <= stop:
                return True
        return False

    def _sort(self):
        """Sort intervals by interval starts"""
        sort_idx = np.argsort(self.data[:, 0])
        self._data = self._data[sort_idx]

#----------------------------------------------------------------------#
#======================================================================#

class EpochArray(IntervalArray):
    """IntervalArray containing temporal intervals (epochs, in seconds)."""

    __aliases__ = {
        'time': 'data',
        '_time': '_data',
        'n_epochs': 'n_intervals',
        'duration': 'length',
        'durations': 'lengths',
        }

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}
        super().__init__(*args, **kwargs)

        self._interval_label = 'epoch'
        self.formatter = formatters.PrettyDuration
        self.base_unit = self.formatter.base_unit

    # # override some functions for API backwards compatibility:
    # @keyword_equivalence(this_or_that={'n_intervals':'n_epochs'})
    # def partition(self, *, ds=None, n_epochs=None, n_intervals=None):
    #     """Returns an EpochArray that has been partitioned.

    #     # Irrespective of whether 'ds' or 'n_epochs' are used, the exact
    #     # underlying support is propagated, and the first and last points
    #     # of the supports are always included, even if this would cause
    #     # n_points or ds to be violated.

    #     Parameters
    #     ----------
    #     ds : float, optional
    #         Maximum length, for each interval.
    #     n_epochs : int, optional
    #         Number of epochs / intervals. If ds is None and n_epochs is None,
    #         then default is to use n_epochs = 100

    #     Returns
    #     -------
    #     out : EpochArray
    #         EpochArray that has been partitioned.
    #     """

    #     if n_intervals is None:
    #         n_intervals = n_epochs
    #     kwargs = {'ds':ds, 'n_intervals': n_intervals}
    #     return super().partition(**kwargs)

class SpaceArray(IntervalArray):
    """IntervalArray containing spatial intervals (in cm)."""

    __aliases__ = {}

    def __init__(self, *args, **kwargs):
        # add class-specific aliases to existing aliases:
        self.__aliases__ = {**super().__aliases__, **self.__aliases__}
        super().__init__(*args, **kwargs)

        self.formatter = formatters.PrettySpace
        self.base_unit = self.formatter.base_unit

@jit(nopython=True)
def interval_intersect(
    interval_starts_a,
    interval_stops_a,
    interval_starts_b,
    interval_stops_b,
    boundaries=True,
):
    new_starts = []
    new_stops = []

    for start_a, stop_a in zip(interval_starts_a, interval_stops_a):
        for start_b, stop_b in zip(interval_starts_b, interval_stops_b):
            if start_a < stop_b and start_b < stop_a:
                new_start = (
                    max(start_a, start_b) if boundaries else min(start_a, start_b)
                )
                new_stop = min(stop_a, stop_b) if boundaries else max(stop_a, stop_b)
                new_starts.append(new_start)
                new_stops.append(new_stop)

    return np.array(new_starts), np.array(new_stops)