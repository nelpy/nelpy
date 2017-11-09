__all__ = ['EpochArray']

import warnings
import numpy as np
import copy
import numbers

from sys import float_info

# from ..utils import is_sorted, \
#                    PrettyDuration, \
#                    PrettyInt

from .. import utils

from ..utils_.decorators import deprecated

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

########################################################################
# class EpochArray
########################################################################
class EpochArray:
    """An array of epochs, where each epoch has a start and stop time.

    Parameters
    ----------
    time : np.array
        If shape (n_epochs, 1) or (n_epochs,), the start time for each
        epoch (which then requires a duration to be specified).
        If shape (n_epochs, 2), the start and stop times for each epoch.
    duration : np.array, float, or None, optional
        The length of the epoch. If (float) then the same duration is
        assumed for every epoch.
    meta : dict, optional
        Metadata associated with spiketrain.
    domain : EpochArray ??? This is pretty meta @-@

    Attributes
    ----------
    time : np.array
        The start and stop times for each epoch. With shape (n_epochs, 2).
    meta : dict
        Metadata associated with spiketrain.
    """

    __attributes__ = ["_time", "_meta", "_domain"]

    def __init__(self, time=None, *, duration=None,
                 meta=None, empty=False, domain=None, label=None):

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        time = np.squeeze(time)  # coerce time into np.array

        # all possible inputs:
        # 1. single epoch, no duration    --- OK
        # 2. single epoch and duration    --- ERR
        # 3. multiple epochs, no duration --- OK
        # 4. multiple epochs and duration --- ERR
        # 5. single scalar and duration   --- OK
        # 6. scalar list and duratin list --- OK
        #
        # Q. won't np.squeeze make our life difficult?
        #
        # Strategy: determine if duration was passed. If so, try to see
        # if time can be coerced into right shape. If not, raise
        # error.
        # If duration was NOT passed, then do usual checks for epochs.

        if duration is not None:  # assume we received scalar starts
            time = np.array(time, ndmin=1)
            duration = np.squeeze(duration).astype(float)
            if duration.ndim == 0:
                duration = duration[..., np.newaxis]

            if time.ndim == 2 and duration.ndim == 1:
                raise ValueError(
                    "duration not allowed when using start and stop "
                    "times")

            if len(duration) > 1:
                if time.ndim == 1 and time.shape[0] != duration.shape[0]:
                    raise ValueError(
                        "must have same number of time and duration "
                        "time"
                        )
            if time.ndim == 1 and duration.ndim == 1:
                stop_epoch = time + duration
                time = np.hstack(
                    (time[..., np.newaxis], stop_epoch[..., np.newaxis]))
        else:  # duration was not specified, so assume we recived epochs

            # Note: if we have an empty array of time with no
            # dimension, then calling len(time) will return a
            # TypeError.
            try:
                # if no time were received, return an empty EpochArray:
                if len(time) == 0:
                    self.__init__(empty=True)
                    return
            except TypeError:
                warnings.warn("unsupported type ("
                    + str(type(time))
                    + "); creating empty EpochArray")
                self.__init__(empty=True)
                return

            # Only one epoch is given eg EpochArray([3,5,6,10]) with no
            # duration and more than two values:
            if time.ndim == 1 and len(time) > 2:  # we already know duration is None
                raise TypeError(
                    "time of size (n_epochs, ) has to be accompanied by "
                    "a duration")

            if time.ndim == 1:  # and duration is None:
                time = np.array([time])

        if time.ndim > 2:
            raise ValueError("time must be a 1D or a 2D vector")

        try:
            if time[:, 0].shape[0] != time[:, 1].shape[0]:
                raise ValueError(
                    "must have the same number of start and stop times")
        except Exception:
            raise Exception("Unhandled EpochArray.__init__ case.")

        # TODO: what if start == stop? what will this break? This situation
        # can arise automatically when slicing a spike train with one or no
        # spikes, for example in which case the automatically inferred support
        # is a delta dirac

        if time.ndim == 2 and np.any(time[:, 1] - time[:, 0] < 0):
            raise ValueError("start must be less than or equal to stop")

        # potentially assign domain
        self._domain = domain

        self._time = time
        self._meta = meta
        self.label = label

        if not self.issorted:
            self._sort()

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty EpochArray" + address_str + ">"
        if self.n_epochs > 1:
            nstr = "%s epochs" % (self.n_epochs)
        else:
            nstr = "1 epoch"
        dstr = "of duration {}".format(utils.PrettyDuration(self.duration))
        return "<EpochArray%s: %s> %s" % (address_str, nstr, dstr)

    def __iter__(self):
        """EpochArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """EpochArray iterator advancer."""
        index = self._index
        if index > self.n_epochs - 1:
            raise StopIteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epocharray = EpochArray(empty=True)

            exclude = ["_time"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("epocharray." + attr + " = self." + attr)
            epocharray._time = np.array([self.time[index, :]])
        self._index += 1
        return epocharray

    def __getitem__(self, *idx):
        """EpochArray index access.

        Accepts integers, slices, and EpochArrays.
        """
        if self.isempty:
            return self

        idx = [ii for ii in idx]
        if len(idx) == 1 and not isinstance(idx[0], int):
            idx = idx[0]
        if isinstance(idx, tuple):
            idx = [ii for ii in idx]

        if isinstance(idx, EpochArray):
            if idx.isempty:  # case 0:
                return EpochArray(empty=True)
            return self.intersect(epoch=idx, boundaries=True)
        else:
            try: # works for ints, lists, and slices
                out = copy.copy(self)
                out._time = None
                out._time = self.time[idx,:]
            except IndexError:
                pass
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))
            finally:
                return out

    def __add__(self, other):
        """add duration to start and stop of each epoch, or join two epoch arrays without merging"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            return new.expand(other, direction='both')
        elif isinstance(other, EpochArray):
            return self.join(other)
        else:
            raise TypeError("unsupported operand type(s) for +: 'EpochArray' and {}".format(str(type(other))))

    def __sub__(self, other):
        """subtract duration from start and stop of each epoch"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            return new.shrink(other, direction='both')
        elif isinstance(other, EpochArray):
            # A - B = A intersect ~B
            return self.intersect(~other)
        else:
            raise TypeError("unsupported operand type(s) for +: 'EpochArray' and {}".format(str(type(other))))

    def __mul__(self, other):
        """expand (>1) or shrink (<1) epoch durations"""
        raise NotImplementedError("operator * not yet implemented")

    def __truediv__(self, other):
        """expand (>1) or shrink (>1) epoch durations"""
        raise NotImplementedError("operator / not yet implemented")

    def __lshift__(self, other):
        """shift time to left"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            new._time = new._time - other
            return new
        else:
            raise TypeError("unsupported operand type(s) for <<: 'EpochArray' and {}".format(str(type(other))))

    def __rshift__(self, other):
        """shift time to right"""
        if isinstance(other, numbers.Number):
            new = copy.copy(self)
            new._time = new._time + other
            return new
        else:
            raise TypeError("unsupported operand type(s) for >>: 'EpochArray' and {}".format(str(type(other))))

    def __and__(self, other):
        """intersection of epoch arrays"""
        if isinstance(other, EpochArray):
            new = copy.copy(self)
            return new.intersect(other, boundaries=True)
        else:
            raise TypeError("unsupported operand type(s) for &: 'EpochArray' and {}".format(str(type(other))))

    def __or__(self, other):
        """join and merge epoch array; set union"""
        if isinstance(other, EpochArray):
            new = copy.copy(self)
            return (new.join(other)).merge()
        else:
            raise TypeError("unsupported operand type(s) for |: 'EpochArray' and {}".format(str(type(other))))

    def __invert__(self):
        """complement within self.domain"""
        return self.complement()

    def __bool__(self):
        """(bool) Empty EventArray"""
        return not self.isempty

    def partition(self, *, ds=None, n_epochs=None):
        """Returns an EpochArray that has been partitioned.

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
        out : EpochArray
            EpochArray that has been partitioned.
        """

        if self.isempty:
            raise ValueError ("cannot parition an empty object in a meaningful way!")

        if ds is not None and n_epochs is not None:
            raise ValueError("ds and n_epochs cannot be used together")

        if n_epochs is not None:
            assert float(n_epochs).is_integer(), "n_epochs must be a positive integer!"
            assert n_epochs > 1, "n_epochs must be a positive integer > 1"
            # determine ds from number of desired points:
            ds = self.duration / n_epochs

        if ds is None:
            # neither n_epochs nor ds was specified, so assume defaults:
            n_epochs = 100
            ds = self.duration / n_epochs

        # build list of points at which to esplit the EpochArray
        new_starts = []
        new_stops = []
        for start, stop in self.time:
            newxvals = np.arange(start, stop, step=ds).tolist()
            if newxvals[-1] + float_info.epsilon < stop:
                newxvals.append(stop)
            newxvals = np.asanyarray(newxvals)
            new_starts.extend(newxvals[:-1])
            new_stops.extend(newxvals[1:])

        # now make a new epoch array:
        out = copy.copy(self)
        out._time = np.hstack(
                [np.array(new_starts)[..., np.newaxis],
                 np.array(new_stops)[..., np.newaxis]])
        return out

    @property
    def label(self):
        """Label describing the epoch array."""
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

    def complement(self, domain=None):
        """Complement within domain.

        Parameters
        ----------
        domain : EpochArray, optional
            EpochArray specifying entire domain. Default is self.domain.

        Returns
        -------
        complement : EpochArray
            EpochArray containing all the nonzero intervals in the
            complement set.
        """

        if domain is None:
            domain = self.domain

        # make sure EpochArray is sorted:
        if not self.issorted:
            self._sort()
        # check that EpochArray is entirely contained within domain
        # if (self.start < domain.start) or (self.stop > domain.stop):
        #     raise ValueError("EpochArray must be entirely contained within domain")

        # check that EpochArray is fully merged, or merge it if necessary
        merged = self.merge()
        # build complement intervals
        starts = np.insert(merged.stops, 0 , domain.start)
        stops = np.append(merged.starts, domain.stop)
        newtimes = np.vstack([starts, stops]).T
        # remove intervals with zero duration
        durations = newtimes[:,1] - newtimes[:,0]
        newtimes = newtimes[durations>0]
        complement = copy.copy(self)
        complement._time = newtimes
        return complement[domain]

    @property
    def domain(self):
        """domain (in seconds) within which support is defined"""
        if self._domain is None:
            return EpochArray([-np.inf, np.inf])
        return self._domain

    @domain.setter
    def domain(self, val):
        """domain (in seconds) within which support is defined"""
        #TODO: add  input validation
        if isinstance(val, EpochArray):
            self._domain = val
        elif isinstance(val, (tuple, list)):
            self._domain = EpochArray([val[0], val[1]])

    @property
    def meta(self):
        """Meta data associated with EpochArray."""
        if self._meta is None:
            warnings.warn("meta data is not available")
        return self._meta

    @meta.setter
    def meta(self, val):
        self._meta = val

    @property
    def time(self):
        """Epoch times [start, stop) in seconds."""
        return self._time

    @property
    def _human_readable_posix_epochs(self):
        """Epoch start and stop times in human readable POSIX time.

        This property is left private, because it has not been carefully
        vetted for public API release yet.
        """
        import datetime
        n_epochs_zfill = len(str(self.n_epochs))
        for ii, (start, stop) in enumerate(self.time):
            print('[ep ' + str(ii).zfill(n_epochs_zfill) + ']\t' +
                  datetime.datetime.fromtimestamp(
                    int(start)).strftime('%Y-%m-%d %H:%M:%S') + ' -- ' +
                  datetime.datetime.fromtimestamp(
                    int(stop)).strftime('%Y-%m-%d %H:%M:%S') + '\t(' +
                  str(utils.PrettyDuration(stop-start)) + ')')

    @property
    def centers(self):
        """(np.array) The center of each epoch."""
        if self.isempty:
            return []
        return np.mean(self.time, axis=1)

    @property
    def durations(self):
        """(np.array) The duration of each epoch."""
        if self.isempty:
            return 0
        return self.time[:, 1] - self.time[:, 0]

    @property
    def duration(self):
        """(float) The total duration of the epoch array."""
        if self.isempty:
            return utils.PrettyDuration(0)
        return utils.PrettyDuration(np.array(self.time[:, 1] - self.time[:, 0]).sum())

    @property
    def starts(self):
        """(np.array) The start of each epoch."""
        if self.isempty:
            return []
        return self.time[:, 0]

    @property
    def start(self):
        """(np.array) The start of the first epoch."""
        if self.isempty:
            return []
        return self.time[:, 0][0]

    @property
    def stops(self):
        """(np.array) The stop of each epoch."""
        if self.isempty:
            return []
        return self.time[:, 1]

    @property
    def stop(self):
        """(np.array) The stop of the last epoch."""
        if self.isempty:
            return []
        return self.time[:, 1][-1]

    @property
    def n_epochs(self):
        """(int) The number of epochs."""
        if self.isempty:
            return 0
        return utils.PrettyInt(len(self.time[:, 0]))

    def __len__(self):
        """(int) The number of epochs."""
        return self.n_epochs

    @property
    def ismerged(self):
        """(bool) No overlapping epochs exist."""
        if self.isempty:
            return True
        if self.n_epochs == 1:
            return True
        if not self.issorted:
            self._sort()
        if not utils.is_sorted(self.stops):
            return False

        return np.all(self.time[1:,0] - self.time[:-1,1] > 0)

    @property
    def issorted(self):
        """(bool) Left edges of epochs are sorted in ascending order."""
        if self.isempty:
            return True
        return utils.is_sorted(self.starts)

    @property
    def isempty(self):
        """(bool) Empty EpochArray."""
        try:
            return len(self.time) == 0
        except TypeError:
            return True  # this happens when self.time is None

    def copy(self):
        """(EpochArray) Returns a copy of the current epoch array."""
        newcopy = EpochArray(empty=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in self.__attributes__:
                exec("newcopy." + attr + " = self." + attr)
        return newcopy

    def intersect(self, epoch, *, boundaries=True, meta=None):
        """Finds intersection (overlap) between two sets of epoch arrays.
        Sampling rates can be different.

        TODO: verify if this requires a merged EpochArray to work properly?

        Parameters
        ----------
        epoch : nelpy.EpochArray
        boundaries : bool
            If True, limits start, stop to epoch start and stop.
        meta : dict, optional
            New dictionary of meta data for epoch ontersection.

        Returns
        -------
        intersect_epochs : nelpy.EpochArray
        """
        if self.isempty or epoch.isempty:
            warnings.warn('epoch intersection is empty')
            return EpochArray(empty=True)

        new_starts = []
        new_stops = []
        epoch_a = self.copy().merge()
        epoch_b = epoch.copy().merge()

        for aa in epoch_a.time:
            for bb in epoch_b.time:
                if (aa[0] <= bb[0] < aa[1]) and (aa[0] < bb[1] <= aa[1]):
                    new_starts.append(bb[0])
                    new_stops.append(bb[1])
                elif (aa[0] < bb[0] < aa[1]) and (aa[0] < bb[1] > aa[1]):
                    new_starts.append(bb[0])
                    if boundaries:
                        new_stops.append(aa[1])
                    else:
                        new_stops.append(bb[1])
                elif (aa[0] > bb[0] < aa[1]) and (aa[0] < bb[1] < aa[1]):
                    if boundaries:
                        new_starts.append(aa[0])
                    else:
                        new_starts.append(bb[0])
                    new_stops.append(bb[1])
                elif (aa[0] >= bb[0] < aa[1]) and (aa[0] < bb[1] >= aa[1]):
                    if boundaries:
                        new_starts.append(aa[0])
                        new_stops.append(aa[1])
                    else:
                        new_starts.append(bb[0])
                        new_stops.append(bb[1])

        if not boundaries:
            new_starts = np.unique(new_starts)
            new_stops = np.unique(new_stops)

        epoch_a._time = np.hstack(
            [np.array(new_starts)[..., np.newaxis],
                np.array(new_stops)[..., np.newaxis]])

        return epoch_a

    def merge(self, *, gap=0.0):
        """Merges epochs that are close or overlapping.

        WARNING! Algorithm only works on SORTED epochs.

        Parameters
        ----------
        gap : float, optional
            Amount (in time) to consider epochs close enough to merge.
            Defaults to 0.0 (no gap).
        Returns
        -------
        merged_epochs : nelpy.EpochArray
        """

        if gap < 0:
            raise ValueError("gap cannot be negative")

        if (self.ismerged) and (gap==0.0):
            # already merged
            return self

        newepocharray = copy.copy(self)

        if not newepocharray.issorted:
            newepocharray._sort()

        while not newepocharray.ismerged or gap>0:
            stops = newepocharray.stops[:-1] + gap
            starts = newepocharray.starts[1:]
            to_merge = (stops - starts) >= 0

            new_starts = [newepocharray.starts[0]]
            new_stops = []

            next_stop = newepocharray.stops[0]
            for i in range(newepocharray.time.shape[0] - 1):
                this_stop = newepocharray.stops[i]
                next_stop = max(next_stop, this_stop)
                if not to_merge[i]:
                    new_stops.append(next_stop)
                    new_starts.append(newepocharray.starts[i + 1])

            new_stops.append(newepocharray.stops[-1])

            new_starts = np.array(new_starts)
            new_stops = np.array(new_stops)

            newepocharray._time = np.vstack([new_starts, new_stops]).T

            # after one pass, all the gap offsets have been added, and
            # then we just need to keep merging...
            gap = 0.0

        return newepocharray

    def expand(self, amount, direction='both'):
        """Expands epoch by the given amount.
        Parameters
        ----------
        amount : float
            Amount (in time) to expand each epoch.
        direction : str
            Can be 'both', 'start', or 'stop'. This specifies
            which direction to resize epoch.
        Returns
        -------
        expanded_epochs : nelpy.EpochArray
        """
        if direction == 'both':
            resize_starts = self.time[:, 0] - amount
            resize_stops = self.time[:, 1] + amount
        elif direction == 'start':
            resize_starts = self.time[:, 0] - amount
            resize_stops = self.time[:, 1]
        elif direction == 'stop':
            resize_starts = self.time[:, 0]
            resize_stops = self.time[:, 1] + amount
        else:
            raise ValueError(
                "direction must be 'both', 'start', or 'stop'")

        newepocharray = copy.copy(self)

        newepocharray._time = np.hstack((
                resize_starts[..., np.newaxis],
                resize_stops[..., np.newaxis]
                ))

        return newepocharray

    def shrink(self, amount, direction='both'):
        """Shrinks epoch by the given amount.
        Parameters
        ----------
        amount : float
            Amount (in time) to shrink each epoch.
        direction : str
            Can be 'both', 'start', or 'stop'. This specifies
            which direction to resize epoch.
        Returns
        -------
        shrinked_epochs : nelpy.EpochArray
        """
        both_limit = min(self.durations / 2)
        if amount > both_limit and direction == 'both':
            raise ValueError("shrink amount too large")

        single_limit = min(self.durations)
        if amount > single_limit and direction != 'both':
            raise ValueError("shrink amount too large")

        return self.expand(-amount, direction)

    def join(self, epoch, meta=None):
        """Combines [and merges] two sets of epochs. Epochs can have
        different sampling rates.

        Parameters
        ----------
        epoch : nelpy.EpochArray
        meta : dict, optional
            New meta data dictionary describing the joined epochs.

        Returns
        -------
        joined_epochs : nelpy.EpochArray
        """

        if self.isempty:
            return epoch
        if epoch.isempty:
            return self

        newepocharray = copy.copy(self)

        join_starts = np.concatenate(
            (self.time[:, 0], epoch.time[:, 0]))
        join_stops = np.concatenate(
            (self.time[:, 1], epoch.time[:, 1]))

        newepocharray._time = np.hstack((
            join_starts[..., np.newaxis],
            join_stops[..., np.newaxis]
            ))
        if not newepocharray.issorted:
            newepocharray._sort()
        if not newepocharray.ismerged:
            newepocharray = newepocharray.merge()
        return newepocharray

    def contains(self, value):
        """Checks whether value is in any epoch.

        Parameters
        ----------
        epochs: nelpy.EpochArray
        value: float or int

        Returns
        -------
        boolean

        """
        # TODO: consider vectorizing this loop, which should increase
        # speed, but also greatly increase memory? Alternatively, if we
        # could assume something about epochs being sorted, this can
        # also be made much faster than the current O(N)
        for start, stop in zip(self.starts, self.stops):
            if start <= value <= stop:
                return True
        return False

    def _sort(self):
        """Sort epochs by epoch starts"""
        sort_idx = np.argsort(self.time[:, 0])
        self._time = self._time[sort_idx]

#----------------------------------------------------------------------#
#======================================================================#