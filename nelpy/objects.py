#encoding : utf-8
"""This module contains the core nelpy object definitions."""

__all__ = ['EventArray',
           'EpochArray',
           'AnalogSignalArray',
           'SpikeTrainArray',
           'BinnedSpikeTrainArray']

# TODO: how should we organize our modules so that nelpy.objects.np does
# not shpw up, for example? If I type nelpy.object.<tab> I only want the
# actual objects to appear in the list. I think I do this with __all__,
# but still haven't quite figured it out yet. __all__ seems to mostly be
# useful for when we want to do from xxx import * in the package
# __init__ method

import warnings
import numpy as np
import copy
import numbers

from scipy import interpolate
from sys import float_info
from collections import namedtuple
from scipy.ndimage.filters import gaussian_filter1d, gaussian_filter

# from shapely.geometry import Point
from abc import ABC, abstractmethod

from .utils import is_sorted, \
                   get_contiguous_segments, \
                   linear_merge, \
                   PrettyDuration, \
                   PrettyInt, \
                   swap_rows, \
                   gaussian_filter

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

########################################################################
# Helper functions
########################################################################

#----------------------------------------------------------------------#
#======================================================================#

# TODO: how should AnalogSignal handle its support? As an EpochArray?
# then what about binning issues? As explicit bin centers? As bin
# indices?
#
# it seems best to have a BinnedEpoch class, so that bin centers can be
# computed lazily when needed, and that duration etc. will be trivial,
# but we also really, REALLY need to be able to index AnalogSignal with
# a regular Epoch or EpochArray object, ...
#
# a nice finish to AnalogSignal would be to plot AnalogSignal on its
# support along with a smoothed version of it

### contested classes --- unclear whether these should exist or not:

# class Event
# class Epoch
# class Spike(Event)

def fsgetter(self):
    """(float) [generic getter] Sampling frequency."""
    if self._fs is None:
        warnings.warn("No sampling frequency has been specified!")
    return self._fs

def fssetter(self, val):
    """(float) [generic setter] Sampling frequency."""
    if self._fs == val:
        return
    try:
        if val <= 0:
            raise ValueError("sampling rate must be positive")
    except:
        raise TypeError("sampling rate must be a scalar")

    # if it is the first time that a sampling rate is set, do not
    # modify anything except for self._fs:
    if self._fs is None:
        pass
    else:
        warnings.warn(
            "Sampling frequency has been updated! This will "
            "modify the spike times."
            )
        self._time = self.tdata / val
    self._fs = val


########################################################################
# class SpikeTrain
########################################################################
class SpikeTrain(ABC):
    """Base class for SpikeTrainArray and BinnedSpikeTrainArray.

    NOTE: This class can't really be instantiated, almost like a pseudo
    abstract class. In particular, during initialization it might fail
    because it checks the n_units of its derived classes to validate
    input to unit_ids and unit_labels. If NoneTypes are used, then you
    may actually succeed in creating an instance of this class, but it
    will be pretty useless.

    Parameters
    ----------
    fs: float, optional
        Sampling rate / frequency (Hz).
    unit_ids : list of int, optional
        Unit IDs preferabbly in integers
    unit_labels : list of str, optional
        Labels corresponding to units. Default casts unit_ids to str.
    label : str or None, optional
        Information pertaining to the source of the spike train.


    Attributes
    ----------
    n_units : int
        Number of units in spike train.
    unit_ids : list of int
        Unit integer IDs.
    unit_labels : list of str
        Labels corresponding to units. Default casts unit_ids to str.
    unit_tags : dict of tags and corresponding unit_ids
        Tags corresponding to units.
    issempty
    **********
    support
    **********
    fs: float
        Sampling frequency (Hz).
    label : str or None
        Information pertaining to the source of the spike train.
    """

    __attributes__ = ["_fs", "_unit_ids", "_unit_labels", "_unit_tags", "_label"]

    def __init__(self, *, fs=None, unit_ids=None, unit_labels=None,
                 unit_tags=None, label=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = EpochArray(empty=True)
            return

        # set initial fs to None
        self._fs = None
        # then attempt to update the fs; this does input validation:
        self.fs = fs

        # WARNING! we need to ensure that self.n_units can work BEFORE
        # we can set self.unit_ids or self.unit_labels, since those
        # setters check that the lengths of the inputs are consistent
        # with self.n_units.

        # inherit unit IDs if available, otherwise initialize to default
        if unit_ids is None:
            unit_ids = list(range(1,self.n_units + 1))

        unit_ids = np.array(unit_ids, ndmin=1)  # standardize unit_ids

        # if unit_labels is empty, default to unit_ids
        if unit_labels is None:
            unit_labels = unit_ids

        unit_labels = np.array(unit_labels, ndmin=1)  # standardize

        self.unit_ids = unit_ids
        self.unit_labels = unit_labels
        self._unit_tags = unit_tags  # no input validation yet
        self.label = label

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        return "<base SpikeTrain" + address_str + ">"

    @abstractmethod
    def isempty(self):
        """(bool) Empty SpikeTrain."""
        return

    @abstractmethod
    def n_units(self):
        """(int) The number of units."""
        return

    @property
    def n_sequences(self):
        warnings.warn("n_sequences is deprecated---use n_epochs instead", DeprecationWarning)
        if self.isempty:
            return 0
        """(int) The number of sequences."""
        return self.support.n_epochs

    @property
    def n_epochs(self):
        if self.isempty:
            return 0
        """(int) The number of underlying epochs."""
        return self.support.n_epochs

    @property
    def unit_ids(self):
        """Unit IDs contained in the SpikeTrain."""
        return self._unit_ids

    @unit_ids.setter
    def unit_ids(self, val):
        if len(val) != self.n_units:
            # print(len(val))
            # print(self.n_units)
            raise TypeError("unit_ids must be of length n_units")
        elif len(set(val)) < len(val):
            raise TypeError("duplicate unit_ids are not allowed")
        else:
            try:
                # cast to int:
                unit_ids = [int(id) for id in val]
            except TypeError:
                raise TypeError("unit_ids must be int-like")
        self._unit_ids = unit_ids

    @property
    def unit_labels(self):
        """Labels corresponding to units contained in the SpikeTrain."""
        if self._unit_labels is None:
            warnings.warn("unit labels have not yet been specified")
        return self._unit_labels

    @unit_labels.setter
    def unit_labels(self, val):
        if len(val) != self.n_units:
            raise TypeError("labels must be of length n_units")
        else:
            try:
                # cast to str:
                labels = [str(label) for label in val]
            except TypeError:
                raise TypeError("labels must be string-like")
        self._unit_labels = labels

    @property
    def unit_tags(self):
        """Tags corresponding to units contained in the SpikeTrain"""
        if self._unit_tags is None:
            warnings.warn("unit tags have not yet been specified")
        return self._unit_tags

    @property
    def support(self):
        """(nelpy.EpochArray) The support of the underlying spiketrain
        (in seconds).
         """
        return self._support

    @property
    def fs(self):
        """(float) Sampling rate / frequency (Hz)."""
        return fsgetter(self)

    @fs.setter
    def fs(self, val):
        """(float) Sampling rate / frequency (Hz)."""
        fssetter(self, val)

    @property
    def label(self):
        """Label pertaining to the source of the spike train."""
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

    def _unit_subset(self, unit_list):
        """Return a SpikeTrain restricted to a subset of units.

        Parameters
        ----------
        unit_list : array-like
            Array or list of unit_ids.
        """
        unit_subset_ids = []
        for unit in unit_list:
            try:
                id = self.unit_ids.index(unit)
            except ValueError:
                warnings.warn("unit_id " + str(unit) + " not found in SpikeTrain; ignoring")
                pass
            else:
                unit_subset_ids.append(id)

        new_unit_ids = (np.asarray(self.unit_ids)[unit_subset_ids]).tolist()
        new_unit_labels = (np.asarray(self.unit_labels)[unit_subset_ids]).tolist()

        if isinstance(self, SpikeTrainArray):
            if len(unit_subset_ids) == 0:
                warnings.warn("no units remaining in requested unit subset")
                return SpikeTrainArray(empty=True)

            spiketrainarray = SpikeTrainArray(empty=True)
            exclude = ["_tdata", "_time", "unit_ids", "unit_labels"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("spiketrainarray." + attr + " = self." + attr)

            spiketrainarray._tdata = self.tdata[unit_subset_ids]
            spiketrainarray._time = self.time[unit_subset_ids]
            spiketrainarray._unit_ids = new_unit_ids
            spiketrainarray._unit_labels = new_unit_labels

            return spiketrainarray
        elif isinstance(self, BinnedSpikeTrainArray):
            if len(unit_subset_ids) == 0:
                warnings.warn("no units remaining in requested unit subset")
                return BinnedSpikeTrainArray(empty=True)

            binnedspiketrainarray = BinnedSpikeTrainArray(empty=True)
            exclude = ["_data", "unit_ids", "unit_labels"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("binnedspiketrainarray." + attr + " = self." + attr)

            binnedspiketrainarray._data = self.data[unit_subset_ids,:]
            binnedspiketrainarray._unit_ids = new_unit_ids
            binnedspiketrainarray._unit_labels = new_unit_labels

            return binnedspiketrainarray
        else:
            raise NotImplementedError(
            "SpikeTrain._unit_slice() not supported for this type yet!")

########################################################################
# class EpochArray
########################################################################
class EpochArray:
    """An array of epochs, where each epoch has a start and stop time.

    Parameters
    ----------
    tdata : np.array
        If shape (n_epochs, 1) or (n_epochs,), the start time for each
        epoch (which then requires a duration to be specified).
        If shape (n_epochs, 2), the start and stop times for each epoch.
    fs : float, optional
        Sampling rate in Hz. If fs is passed as a parameter, then time
        is assumed to be in sample numbers instead of actual time.
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
    tdata : np.array
        The start and stop tdata for each epoch. With shape (n_epochs, 2).
    fs: float
        Sampling frequency (Hz).
    meta : dict
        Metadata associated with spiketrain.
    """

    __attributes__ = ["_tdata", "_time", "_fs", "_meta", "_domain"]

    def __init__(self, tdata=None, *, fs=None, duration=None,
                 meta=None, empty=False, domain=None):

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        tdata = np.squeeze(tdata)  # coerce tdata into np.array

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
        # if tdata can be coerced into right shape. If not, raise
        # error.
        # If duration was NOT passed, then do usual checks for epochs.

        if duration is not None:  # assume we received scalar starts
            tdata = np.array(tdata, ndmin=1)
            duration = np.squeeze(duration).astype(float)
            if duration.ndim == 0:
                duration = duration[..., np.newaxis]

            if tdata.ndim == 2 and duration.ndim == 1:
                raise ValueError(
                    "duration not allowed when using start and stop "
                    "times")

            if len(duration) > 1:
                if tdata.ndim == 1 and tdata.shape[0] != duration.shape[0]:
                    raise ValueError(
                        "must have same number of time and duration "
                        "tdata"
                        )
            if tdata.ndim == 1 and duration.ndim == 1:
                stop_epoch = tdata + duration
                tdata = np.hstack(
                    (tdata[..., np.newaxis], stop_epoch[..., np.newaxis]))
        else:  # duration was not specified, so assume we recived epochs

            # Note: if we have an empty array of tdata with no
            # dimension, then calling len(tdata) will return a
            # TypeError.
            try:
                # if no tdata were received, return an empty EpochArray:
                if len(tdata) == 0:
                    return EpochArray(empty=True)
            except TypeError:
                warnings.warn("unsupported type ("
                    + str(type(tdata))
                    + "); creating empty EpochArray")
                return EpochArray(empty=True)

            # Only one epoch is given eg EpochArray([3,5,6,10]) with no
            # duration and more than two values:
            if tdata.ndim == 1 and len(tdata) > 2:  # we already know duration is None
                raise TypeError(
                    "tdata of size (n_epochs, ) has to be accompanied by "
                    "a duration")

            if tdata.ndim == 1:  # and duration is None:
                tdata = np.array([tdata])

        if tdata.ndim > 2:
            raise ValueError("tdata must be a 1D or a 2D vector")

        # set initial fs to None
        self._fs = None
        # then attempt to update the fs; this does input validation:
        self.fs = fs

        try:
            if tdata[:, 0].shape[0] != tdata[:, 1].shape[0]:
                raise ValueError(
                    "must have the same number of start and stop times")
        except Exception:
            raise Exception("Unhandled EpochArray.__init__ case.")

        # TODO: what if start == stop? what will this break? This situation
        # can arise automatically when slicing a spike train with one or no
        # spikes, for example in which case the automatically inferred support
        # is a delta dirac

        if tdata.ndim == 2 and np.any(tdata[:, 1] - tdata[:, 0] < 0):
            raise ValueError("start must be less than or equal to stop")

        # if a sampling rate was given, relate time to tdata using fs:
        if fs is not None:
            time = tdata / fs
        else:
            time = tdata

        # potentially assign domain
        self._domain = domain

        self._time = time
        self._tdata = tdata
        self._fs = fs
        self._meta = meta

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
        dstr = "of duration {}".format(PrettyDuration(self.duration))
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

            exclude = ["_tdata", "_time"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("epocharray." + attr + " = self." + attr)
            epocharray._tdata = np.array([self.tdata[index, :]])
            epocharray._time = np.array([self.time[index, :]])
        self._index += 1
        return epocharray

    def __getitem__(self, idx):
        """EpochArray index access.

        Accepts integers, slices, and EpochArrays.
        """
        if self.isempty:
            return self

        if isinstance(idx, EpochArray):
            # case #: (self, idx):
            # case 0: idx.isempty == True
            # case 1: (fs, fs) = (None, const)
            # case 2: (fs, fs) = (const, None)
            # case 3: (fs, fs) = (None, None)
            # case 4: (fs, fs) = (const, const)
            # case 5: (fs, fs) = (constA, constB)
            if idx.isempty:  # case 0:
                return EpochArray(empty=True)
            if idx.fs != self.fs:  # cases (1, 2, 5):
                epocharray = EpochArray(empty=True)
                epocharray._tdata = idx._tdata
                epocharray._time = idx._time
                epoch = self.intersect(epocharray, boundaries=True)
            else:  # cases (3, 4)
                epoch = self.intersect(
                    epoch=idx,
                    boundaries=True
                    )
            if epoch.isempty:
                return EpochArray(empty=True)
            return epoch
        elif isinstance(idx, int):
            epocharray = EpochArray(empty=True)
            exclude = ["_tdata", "_time"]
            attrs = (x for x in self.__attributes__ if x not in exclude)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("epocharray." + attr + " = self." + attr)
            try:
                epocharray._time = self.time[[idx], :]  # use np integer indexing! Cool!
                epocharray._tdata = self.tdata[[idx], :]
            except IndexError:
                # index is out of bounds, so return an empty EpochArray
                pass
            finally:
                return epocharray
        else:
            try:
                epocharray = EpochArray(empty=True)
                exclude = ["_tdata", "_time"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for attr in attrs:
                        exec("epocharray." + attr + " = self." + attr)
                epocharray._time = np.array([self.starts[idx],
                                             self.stops[idx]]).T
                epocharray._tdata = np.array([self._tdatastarts[idx],
                                              self._tdatastops[idx]]).T
                return epocharray
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))

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
            raise NotImplementedError("EpochArray subtraction not implemented yet :(")
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
            if self._fs is None:
                fs = 1
            else:
                fs = self._fs
            new = copy.copy(self)
            new._time = new._time - other
            new._tdata = new._tdata - other*fs
            return new
        else:
            raise TypeError("unsupported operand type(s) for <<: 'EpochArray' and {}".format(str(type(other))))

    def __rshift__(self, other):
        """shift time to right"""
        if isinstance(other, numbers.Number):
            if self._fs is None:
                fs = 1
            else:
                fs = self._fs
            new = copy.copy(self)
            new._time = new._time + other
            new._tdata = new._tdata + other*fs
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
        """join and merge epoch arrays"""
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
        if (self.start < domain.start) or (self.stop > domain.stop):
            raise ValueError("EpochArray must be entirely contained within domain")
        # check that EpochArray is fully merged, or merge it if necessary
        merged = self.merge()
        # build complement intervals
        starts = np.insert(merged.stops,0 , domain.start)
        stops = np.append(merged.starts, domain.stop)
        newtimes = np.vstack([starts, stops]).T
        # remove intervals with zero duration
        durations = newtimes[:,1] - newtimes[:,0]
        newtimes = newtimes[durations>0]
        complement = copy.copy(self)
        complement._time = newtimes
        if self._fs is None:
            fs = 1
        else:
            fs = self._fs
        complement._tdata = newtimes * fs
        return complement

    @property
    def domain(self):
        """domain (in seconds) within which support is defined"""
        if self._domain is None:
            return EpochArray([-np.inf, np.inf], fs=1)
        return self._domain

    @domain.setter
    def domain(self, val):
        """domain (in seconds) within which support is defined"""
        #TODO: add  input validation
        if isinstance(val, EpochArray):
            self._domain = val
        elif isinstance(val, (tuple, list)):
            self._domain = EpochArray([val[0], val[1]], fs=1)

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
    def fs(self):
        """(float) Sampling frequency."""
        return fsgetter(self)

    @fs.setter
    def fs(self, val):
        """(float) Sampling frequency."""
        fssetter(self, val)

    @property
    def tdata(self):
        """Epochs [start, stop] in sample numbers (default fs=1 Hz)."""
        return self._tdata

    @property
    def time(self):
        """Epoch times [start, stop] in seconds."""
        return self._time

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
            return PrettyDuration(0)
        return PrettyDuration(np.array(self.time[:, 1] - self.time[:, 0]).sum())

    @property
    def starts(self):
        """(np.array) The start of each epoch."""
        if self.isempty:
            return []
        return self.time[:, 0]

    @property
    def _tdatastarts(self):
        """(np.array) The start of each epoch, in tdata"""
        if self.isempty:
            return []
        return self.tdata[:, 0]

    @property
    def start(self):
        """(np.array) The start of the first epoch."""
        if self.isempty:
            return []
        return self.time[:, 0][0]

    @property
    def _tdatastart(self):
        """(np.array) The start of the first epoch, in tdata"""
        if self.isempty:
            return []
        return self.tdata[:, 0][0]

    @property
    def stops(self):
        """(np.array) The stop of each epoch."""
        if self.isempty:
            return []
        return self.time[:, 1]

    @property
    def _tdatastops(self):
        """(np.array) The stop of each epoch, in tdata"""
        if self.isempty:
            return []
        return self.tdata[:, 1]

    @property
    def stop(self):
        """(np.array) The stop of the last epoch."""
        if self.isempty:
            return []
        return self.time[:, 1][-1]

    @property
    def _tdatastop(self):
        """(np.array) The stop of the first epoch, in tdata"""
        return self.tdata[:, 0][0]

    @property
    def n_epochs(self):
        """(int) The number of epochs."""
        if self.isempty:
            return 0
        return PrettyInt(len(self.time[:, 0]))

    def __len__(self):
        """(int) The number of epochs."""
        return self.n_epochs

    @property
    def ismerged(self):
        """(bool) No overlapping epochs exist."""
        if self.isempty:
            return True
        if not self.issorted:
            self._sort()
        if not is_sorted(self.stops):
            return False

        return np.all(self.time[1:,0] - self.time[:-1,1] >= 0)

    @property
    def issorted(self):
        """(bool) Left edges of epochs are sorted in ascending order."""
        if self.isempty:
            return True
        return is_sorted(self.starts)

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
            # TODO: copy everything except time? Wouldn't rest get
            # lost anyway due to no samples ==> return EpochArray(empty)?
            return EpochArray([], duration=[], meta=meta)

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

        epocharray = EpochArray(empty=True)
        exclude = ["_tdata", "_time", "_fs"]
        attrs = (x for x in self.__attributes__ if x not in exclude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("epocharray." + attr + " = self." + attr)

        # case 1: (fs, fs) = (None, const)
        # case 2: (fs, fs) = (const, None)
        # case 3: (fs, fs) = (None, None)
        # case 4: (fs, fs) = (const, const)
        # case 5: (fs, fs) = (constA, constB)

        if self.fs != epoch.fs or self.fs is None:  # cases (1, 2, 3, 5)
            warnings.warn(
                "sampling rates are different; intersecting along "
                "time only and throwing away fs"
                )
            epocharray._time = np.hstack(
                [np.array(new_starts)[..., np.newaxis],
                 np.array(new_stops)[..., np.newaxis]])
            epocharray._tdata = epocharray._time
            epocharray._fs = None
        else:  # case (4, )
            epocharray._time = np.hstack(
                [np.array(new_starts)[..., np.newaxis],
                 np.array(new_stops)[..., np.newaxis]])
            epocharray._tdata = epocharray._time*self.fs
            epocharray._fs = self.fs

        return epocharray

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
        if (self.ismerged) and (gap==0.0):
            "yeah, not gonna merge..."
            return self

        if gap < 0:
            raise ValueError("gap cannot be negative")

        newepocharray = copy.copy(self)

        fs = newepocharray.fs
        if fs is None:
            fs = 1

        gap = gap * fs

        if not newepocharray.issorted:
            newepocharray._sort()

        while not newepocharray.ismerged or gap>0:
            stops = newepocharray._tdatastops[:-1] + gap
            starts = newepocharray._tdatastarts[1:]
            to_merge = (stops - starts) >= 0

            new_starts = [newepocharray._tdatastarts[0]]
            new_stops = []

            next_stop = newepocharray._tdatastops[0]
            for i in range(newepocharray.time.shape[0] - 1):
                this_stop = newepocharray._tdatastops[i]
                next_stop = max(next_stop, this_stop)
                if not to_merge[i]:
                    new_stops.append(next_stop)
                    new_starts.append(newepocharray._tdatastarts[i + 1])

            new_stops.append(newepocharray._tdatastops[-1])

            new_starts = np.array(new_starts)
            new_stops = np.array(new_stops)

            newepocharray._time = np.vstack([new_starts, new_stops]).T
            newepocharray._tdata = newepocharray._time * fs

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
        fs = newepocharray.fs
        if fs is None:
            fs = 1
        newepocharray._time = np.hstack((
                resize_starts[..., np.newaxis],
                resize_stops[..., np.newaxis]
                ))
        newepocharray._tdata = newepocharray._time * fs

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

        if self.fs != epoch.fs:
            warnings.warn(
                "sampling rates are different; joining along time "
                "only and throwing away fs"
                )
            join_starts = np.concatenate(
                (self.time[:, 0], epoch.time[:, 0]))
            join_stops = np.concatenate(
                (self.time[:, 1], epoch.time[:, 1]))
            #TODO: calling merge() just once misses some instances.
            # I haven't looked carefully enough to know which edge cases
            # these are...
            # merge() should therefore be checked!
            # return EpochArray(join_starts, fs=None,
            # duration=join_stops - join_starts, meta=meta).merge()
            # .merge()
            return EpochArray(
                join_starts,
                fs=None,
                duration=join_stops - join_starts,
                meta=meta
                )
        else:
            join_starts = np.concatenate(
                (self.tdata[:, 0], epoch.tdata[:, 0]))
            join_stops = np.concatenate(
                (self.tdata[:, 1], epoch.tdata[:, 1]))

        # return EpochArray(join_starts, fs=self.fs, duration=
        # join_stops - join_starts, meta=meta).merge().merge()
        return EpochArray(
            join_starts,
            fs=self.fs,
            duration=join_stops - join_starts,
            meta=meta
            )

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
        self._tdata = self._tdata[sort_idx]
#----------------------------------------------------------------------#
#======================================================================#


########################################################################
# class EventArray
########################################################################
class EventArray:
    """Class description.

    Parameters
    ----------
    tdata : np.array
        If shape (n_epochs, 1) or (n_epochs,), the start time for each
        epoch.
        If shape (n_epochs, 2), the start and stop times for each epoch.
    fs : float, optional
        Sampling rate in Hz. If fs is passed as a parameter, then time
        is assumed to be in sample numbers instead of actual time.

    Attributes
    ----------
    time : np.array
        The start and stop times for each epoch. With shape (n_epochs, 2).
    """

    def __init__(self, *, tdata, fs=None, duration=None, meta=None):

        # if no tdata were received, return an empty EpochArray:
        if len(tdata) == 0:
            self._tdata = np.array([])
            self._time = np.array([])
            self._fs = None
            self._meta = None
            return

        self._fs = fs
        self._meta = meta

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty EventArray" + address_str + ">"
        # return "<EventArray: %s> %s" % (nstr, dstr)
        return "<EventArray" + address_str + ">"

    def __getitem__(self, idx):
        raise NotImplementedError(
            'EventArray.__getitem__ not implemented yet')

    @property
    def isempty(self):
        """(bool) Empty EventArray."""
        raise NotImplementedError(
            'EventArray.isempty not implemented yet')
#----------------------------------------------------------------------#
#======================================================================#


########################################################################
# class AnalogSignalArray
########################################################################
class AnalogSignalArray:
    """Continuous analog signal(s) with regular sampling rates and same
    support. NOTE: ydata that is not equal dimensionality will NOT work
    and error/warning messages may/may not be sent out. Also, in this
    current rendition, I am assuming tdata is the exact same for all
    signals passed through. As such, tdata is expected to be single
    dimensional.

    Parameters
    ----------
    ydata : np.array(dtype=np.float,dimension=N)
    tdata : np.array(dtype=np.float,dimension=N), optional
        if fs is provided tdata is assumed to be sample numbers
        else it is assumed to be time but tdata can be a non time
        variable. Additionally, if tdata is time it is assumed to be
        sampled regularly in order to generate epochs. Irregular
        sampling rates can be corrected with operations on the support.
    fs : float, optional
        Sampling rate in Hz. If fs is passed as a parameter, then time
        is assumed to be in sample numbers instead of actual time. See
        fs_meta parameter below if sampling rate is to be stored as
        metadata and not used for calculations. See fs_acquisition if 
        timestamps are stored at a different rate than what was sampled
        and marked by the system.
    fs_acquisition : float, optional
        Optional to store sampling rate in Hz of the acquisition system.
        This should be used when tdata is passed in timestamps associated
        with the acquisition system but is stored in step sizes that are
        of a different sampling rate. E.g. times could be stamped at 
        30kHz but stored in a decimated fashion at 3kHz so instead of 
        1,2,3,4,5,6,7,8,9,10,11,...,20,21,22... it would be 1,10,20,30... 
        In cases like this fs_acquisiton would be 10 times higher than fs. 
        Additionally, fs_acquisition as opposed to fs will be used to 
        calculate time if it is changed from the default None. See 
        notebook of AnalogSignalArray uses. 
    fs_meta : float, optional
        Optional sampling rate storage. The true sampling rate if tdata
        is time can be stored here. The above parameter, fs, must be left
        blank if tdata is time and not sample numbers. This will not be
        used for any calculations. Just to store in AnalogSignalArray as
        a value.
    support : EpochArray, optional
        EpochArray array on which LFP is defined.
        Default is [0, last spike] inclusive.
    step : int
        specifies step size of samples passed as tdata if fs is given,
        default is None. If not passed it is inferred by the minimum
        difference in between samples of tdata passed in (based on if FS
        is passed). e.g. decimated data would have sample numbers every
        ten samples so step=10
    merge_sample_gap : float, optional
        Optional merging of gaps between support epochs. If epochs are within
        a certain amount of time, gap, they will be merged as one epoch. Example
        use case is when there is a dropped sample
    empty : bool
        Return an empty AnalogSignalArray if true else false. Default
        set to false.

    Attributes
    ----------
    ydata : np.array
        With shape (n_ydata,N).
    tdata : np.array
        With shape (n_tdata,N).
    time : np.array
        With shape (n_tdata,N).
    fs : float, scalar, optional
        See Parameters
    fs_acquisition : float, scalar, optional
    fs_meta : float, scalar, optional
        See Paramters
    step : int
        See Parameters
    support : EpochArray, optional
        See Parameters
    interp : array of interpolation objects from scipy.interpolate

        See Parameters
    """
    __attributes__ = ['_ydata', '_tdata', '_time', '_fs', '_support', \
                      '_interp', '_fs_meta', '_step', '_fs_acquisition']
    def __init__(self, ydata, *, tdata=None, fs=None, fs_acquisition=None, fs_meta = None,
                 step=None, merge_sample_gap=0, support=None, empty=False):

        if(empty):
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = EpochArray(empty=True)
            return
        #check if single AnalogSignal or multiple AnalogSignals in array
        #and standardize ydata to 2D
        ydata = np.squeeze(ydata).astype(float)
        try:
            if(ydata.shape[0] == ydata.size):
                ydata = np.array(ydata,ndmin=2).astype(float)

        except ValueError:
            raise TypeError("Unsupported type! integer or floating point expected")
        self._step = step
        self._fs_meta = fs_meta

        # set initial fs to None
        self._fs = None
        # then attempt to update the fs; this does input validation:
        self.fs = fs

        #set fs_acquisition
        self._fs_acquisition = None
        if(fs_acquisition is not None):
            try:
                if(fs_acquisition > 0):
                    self._fs_acquisition = fs_acquisition
                else:
                    raise ValueError("fs_acquisition must be positive")
            except TypeError:
                raise TypeError("fs_acquisition expected to be a scalar")

        # Note; if we have an empty array of ydata with no dimension,
        # then calling len(ydata) will return a TypeError
        try:
            # if no ydata are given return empty AnalogSignal
            if ydata.size == 0:
                self.__init__([],empty=True)
                return
        except TypeError:
            warnings.warn("unsupported type; creating empty AnalogSignalArray")
            self.__init__([],empty=True)
            return

        # Note: if both tdata and ydata are given and dimensionality does not
        # match, then TypeError!
        if(tdata is not None):
            tdata = np.squeeze(tdata).astype(float)
            if(tdata.shape[0] != ydata.shape[1]):
                self.__init__([],empty=True)
                raise TypeError("tdata and ydata size mismatch!")

        self._ydata = ydata
        # Note: time will be None if this is not a time series and fs isn't
        # specified set xtime to None.
        self._time = None

        # Alright, let's handle all the possible parameter cases!
        if tdata is not None:
            if fs is not None:
                if(self._fs_acquisition is not None):
                    time = tdata / self._fs_acquisition
                else:
                    time = tdata / self._fs
                self._tdata = tdata
                if support is not None:
                    # tdata, fs, support passed properly
                    self._time = time
                    # print(self._ydata)
                    self._restrict_to_epoch_array(epocharray=support)
                    # print(self._ydata)
                    if(self.support.isempty):
                        warnings.warn("Support is empty. Empty AnalogSignalArray returned")
                        exclude = ['_support','_ydata']
                        attrs = (x for x in self.__attributes__ if x not in exclude)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            for attr in attrs:
                                exec("self." + attr + " = None")

                # tdata, fs and no support
                else:
                    warnings.warn("creating support with given tdata and sampling rate, fs!")
                    self._time = time
                    if self._step is None:
                        self._step = np.min(np.diff(tdata))
                    self._support = EpochArray(get_contiguous_segments(self._tdata,
                        step=self._step), fs=self._fs_acquisition)
                    self._support = self._support.merge(gap=merge_sample_gap)
            else:
                time = tdata
                self._tdata = tdata
                # tdata and support
                if support is not None:
                    self._time = time
                    # print(self._ydata)
                    self._restrict_to_epoch_array(epocharray=support)
                    # print(self._ydata)
                    warnings.warn("support created with specified epoch array but no specified sampling rate")
                    if(self.support.isempty):
                        warnings.warn("Support is empty. Empty AnalogSignalArray returned")
                        exclude = ['_support','_ydata']
                        attrs = (x for x in self.__attributes__ if x not in exclude)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            for attr in attrs:
                                exec("self." + attr + " = None")
                # tdata
                else:
                    warnings.warn("support created with just tdata! no sampling rate specified so "\
                                    + " support is entire range of signal with Epochs separted by" \
                                    + " time step difference from first time to second time")
                    self._time = time
                    #infer step size if not given as minimum difference in samples passed
                    if self._step is None:
                        self._step = np.min(np.diff(tdata))
                    self._support = EpochArray(get_contiguous_segments(tdata,
                        step=self._step), fs=self._fs_acquisition)
                    #merge gaps in Epochs if requested
                    self._support = self._support.merge(gap=merge_sample_gap)
        else:
            tdata = np.arange(0, ydata.shape[1], 1)
            if fs is not None:
                if(self._fs_acquisition is not None):
                    time = tdata / self._fs_acquisition
                else:
                    time = tdata / self._fs
                # fs and support
                if support is not None:
                    self.__init__([],empty=True)
                    raise TypeError("tdata must be passed if support is specified")
                # just fs
                else:
                    self._time = time
                    warnings.warn("support created with given sampling rate, fs")
                    self._support = EpochArray(np.array([0, time[-1]]))
            else:
                # just support
                if support is not None:
                    self.__init__([],empty=True)
                    raise TypeError("tdata must be passed if support is "
                        +"specified")
                # just ydata
                else:
                    self._time = tdata
                    warnings.warn("support created with given ydata! support is entire signal")
                    self._support = EpochArray(np.array([0, tdata[-1]]))
            self._tdata = tdata

    def __mul__(self, other):
        """overloaded * operator."""
        if isinstance(other, numbers.Number):
            newasa = copy.copy(self)
            newasa._ydata = self._ydata * other
            return newasa
        else:
            raise TypeError("unsupported operand type(s) for *: 'AnalogSignalArray' and '{}'".format(str(type(other))))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        """overloaded / operator."""
        if isinstance(other, numbers.Number):
            newasa = copy.copy(self)
            newasa._ydata = self._ydata / other
            return newasa
        else:
            raise TypeError("unsupported operand type(s) for /: 'AnalogSignalArray' and '{}'".format(str(type(other))))

    def __len__(self):
        return self.n_epochs

    def add_signal(self, signal, label=None):
        """Docstring goes here.
        Basically we add a signal, and we add a label
        """
        # TODO: add functionality to check that supports are the same, etc.
        if isinstance(signal, AnalogSignalArray):
            signal = signal.ydata

        signal = np.squeeze(signal)
        if signal.ndim > 1:
            raise TypeError("Can only add one signal at a time!")
        if self._ydata.ndim==1:
            self._ydata = np.vstack([np.array(self._ydata, ndmin=2), np.array(signal, ndmin=2)])
        else:
            self._ydata = np.vstack([self._ydata, np.array(signal, ndmin=2)])
        print('labels not supported yet!')
        #TODO: add label support
        return self

    def _restrict_to_epoch_array(self, *, epocharray=None, update=True):
        """Restrict self._time and self._ydata to an EpochArray. If no
        EpochArray is specified, self._support is used.

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
                exclude = ['_support','_ydata']
                attrs = (x for x in self.__attributes__ if x not in exclude)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for attr in attrs:
                        exec("self." + attr + " = None")
                self._ydata = np.zeros([0,self._ydata.shape[0]])
                self._support = epocharray
                return
        except AttributeError:
            raise AttributeError("EpochArray expected")

        indices = []
        for eptime in epocharray.time:
            t_start = eptime[0]
            t_stop = eptime[1]
            indices.append((self.time >= t_start) & (self.time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)
        if np.count_nonzero(indices) < len(self._time):
            warnings.warn(
                'ignoring signal outside of support')
        try:
            self._ydata = self._ydata[:,indices]
        except IndexError:
            self._ydata = np.zeros([0,self._ydata.shape[0]])
        self._time = self._time[indices]
        self._tdata = self._tdata[indices]
        if update:
            self._support = epocharray

    def smooth(self, *, fs=None, sigma=None, bw=None, inplace=False):
        """Smooths the regularly sampled AnalogSignalArray with a Gaussian kernel.

        Smoothing is applied in time, and the same smoothing is applied to each
        signal in the AnalogSignalArray.

        Smoothing is applied within each epoch.

        Parameters
        ----------
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
        kwargs = {'inplace' : inplace,
                'fs' : fs,
                'sigma' : sigma,
                'bw' : bw}

        return gaussian_filter(self, **kwargs)

    @property
    def lengths(self):
        """(list) The number of samples in each epoch."""
        # TODO: make this faster and better!
        lengths = [segment.n_samples for segment in self]
        return np.asanyarray(lengths).squeeze()

    @property
    def n_signals(self):
        """(int) The number of signals."""
        try:
            return PrettyInt(self._ydata.shape[0])
        except AttributeError:
            return 0

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty AnalogSignal" + address_str + ">"
        if self.n_epochs > 1:
            epstr = " ({} segments)".format(self.n_epochs)
        else:
            epstr = ""
        try:
            if(self.n_signals > 0):
                nstr = " %s signals%s" % (self.n_signals, epstr)
        except IndexError:
            nstr = " 1 signal%s" % epstr
        dstr = " for a total of {}".format(PrettyDuration(self.support.duration))
        return "<AnalogSignalArray%s:%s>%s" % (address_str, nstr, dstr)

    @property
    def ydata(self):
        """(np.array N-Dimensional) ydata that was initially passed in but transposed
        """
        return self._ydata

    @property
    def tdata(self):
        """(np.array 1D) Either sample numbers or time depending on what was passed in
        """
        if self._tdata is None:
            warnings.warn("No tdata specified")
        return self._tdata

    @property
    def support(self):
        """(nelpy.EpochArray) The support of the underlying spiketrain
        (in seconds).
         """
        return self._support

    @property
    def step(self):
        """ steps per sample
        Example 1: sample_numbers = np.array([1,2,3,4,5,6]) #aka tdata
        Steps per sample in the above case would be 1

        Example 2: sample_numbers = np.array([1,3,5,7,9]) #aka tdata
        Steps per sample in Example 2 would be 2
        """
        return self._step

    @property
    def time(self):
        """(np.array 1D) Time calculated off sample numbers and frequency or time passed in
        """
        if self._time is None:
            warnings.warn("No time calculated. This should be due to no tdata specified")
        return self._time

    @property
    def fs(self):
        """(float) Sampling frequency."""
        return fsgetter(self)

    @fs.setter
    def fs(self, val):
        """(float) Sampling rate / frequency (Hz)."""
        fssetter(self, val)

    @property
    def isempty(self):
        """(bool) checks length of ydata input"""
        try:
            return len(self._ydata) == 0
        except TypeError: #TypeError should happen if _ydata = []
            return True

    @property
    def n_epochs(self):
        """(int) number of epochs in AnalogSignalArray"""
        return self._support.n_epochs

    @property
    def n_samples(self):
        """(int) number of time samples where signal is defined."""
        if self.isempty:
            return 0
        return PrettyInt(len(self.time))

    def __iter__(self):
        """AnalogSignal iterator initialization"""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """AnalogSignal iterator advancer."""
        index = self._index
        if index > self.n_epochs - 1:
            raise StopIteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epoch = EpochArray(empty=True)
            exclude = ["_tdata", "_time"]
            attrs = (x for x in self._support.__attributes__ if x not in exclude)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("epoch." + attr + " = self._support." + attr)
                try:
                    epoch._time = self._support.time[[index], :]  # use np integer indexing! Cool!
                    epoch._tdata = self._support.tdata[[index], :]
                except IndexError:
                    # index is out of bounds, so return an empty EpochArray
                    pass
            # epoch = EpochArray(
            #         np.array([self._support.tdata[index,:]])
            #     )
        self._index += 1

        asa = AnalogSignalArray([],empty=True)
        exclude = ['_interp','_support']
        attrs = (x for x in self.__attributes__ if x not in exclude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("asa." + attr + " = self." + attr)
        asa._restrict_to_epoch_array(epocharray=epoch)
        if(asa.support.isempty):
            warnings.warn("Support is empty. Empty AnalogSignalArray returned")
            asa = AnalogSignalArray([],empty=True)
        return asa

    def __getitem__(self, idx):
        """AnalogSignalArray index access.
        Parameters
        Parameters
        ----------
        idx : EpochArray, int, slice
            intersect passed epocharray with support,
            index particular a singular epoch or multiple epochs with slice
        """
        epoch = self._support[idx]
        if epoch is None:
            warnings.warn("Index resulted in empty epoch array")
            return AnalogSignalArray(empty=True)
        else:
            asa = AnalogSignalArray([],empty=True)
            exclude = ['_interp','_support']
            attrs = (x for x in self.__attributes__ if x not in exclude)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("asa." + attr + " = self." + attr)
            if(not asa.isempty):
                asa._restrict_to_epoch_array(epocharray=epoch)
            if(asa.support.isempty):
                        warnings.warn("Support is empty. Empty AnalogSignalArray returned")
                        asa = AnalogSignalArray([],empty=True)
            return asa

    def _subset(self, idx):
        asa = self.copy()
        asa._ydata = self._ydata[idx,:]
        return asa

    def copy(self):
        asa = AnalogSignalArray([], empty=True)
        exclude = ['_interp']
        attrs = (x for x in self.__attributes__ if x not in exclude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("asa." + attr + " = self." + attr)
        try:
            exec("asa._interp = self._interp")
        except AttributeError:
            pass
        return asa

    def mean(self,*,axis=1):
        """Returns the mean of each signal in AnalogSignalArray."""
        try:
            means = np.mean(self._ydata, axis=axis).squeeze()
            if means.size == 1:
                return np.asscalar(means)
            return means
        except IndexError:
            raise IndexError("Empty AnalogSignalArray cannot calculate mean")

    def std(self,*,axis=1):
        """Returns the standard deviation of each signal in AnalogSignalArray."""
        try:
            stds = np.std(self._ydata,axis=axis).squeeze()
            if stds.size == 1:
                return np.asscalar(stds)
            return stds
        except IndexError:
            raise IndexError("Empty AnalogSignalArray cannot calculate standard deviation")

    def max(self,*,axis=1):
        """Returns the maximum of each signal in AnalogSignalArray"""
        try:
            maxes = np.amax(self._ydata,axis=axis).squeeze()
            if maxes.size == 1:
                return np.asscalar(maxes)
            return maxes
        except ValueError:
            raise ValueError("Empty AnalogSignalArray cannot calculate maximum")

    def min(self,*,axis=1):
        """Returns the minimum of each signal in AnalogSignalArray"""
        try:
            mins = np.amin(self._ydata,axis=axis).squeeze()
            if mins.size == 1:
                return np.asscalar(mins)
            return mins
        except ValueError:
            raise ValueError("Empty AnalogSignalArray cannot calculate minimum")

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
        clipped_analogsignalarray : AnalogSignalArray
            AnalogSignalArray with the signal clipped with the elements of ydata, but where the values <
            min are replaced with min and the values > max are replaced
            with max.
        """
        new_ydata = np.clip(self._ydata, min, max)
        newasa = self.copy()
        newasa._ydata = new_ydata
        return newasa

    def trim(self, start, stop=None, *, fs=None):
        """Trim the AnalogSignalArray to a single time/sample interval.

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
        trim : AnalogSignalArray
            The AnalogSignalArray on the interval [start, stop].

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
        warnings.warn("AnalogSignalArray: Trim may not work!")
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
                        "unsupported input to AnalogSignalArray.trim()")
                stop = np.array(start[1], ndmin=1)
                start = np.array(start[0], ndmin=1)
                if len(start) != 1 or len(stop) != 1:
                    raise TypeError(
                        "start and stop must be scalar floats")
            except TypeError:
                raise TypeError(
                    "start and stop must be scalar floats")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            epoch = self._support.intersect(
                EpochArray(
                    [start, stop],
                    fs=fs))
            if not epoch.isempty:
                analogsignalarray = self[epoch]
            else:
                analogsignalarray = AnalogSignalArray([],empty=True)
        return analogsignalarray

    @property
    def _ydata_rowsig(self):
        """returns wide-format ydata s.t. each row is a signal."""
        return self._ydata

    @property
    def _ydata_colsig(self):
        """returns skinny-format ydata s.t. each column is a signal."""
        return self._ydata.T

    def _get_interp1d(self,* , kind='linear', copy=True, bounds_error=False,
                      fill_value=np.nan, assume_sorted=None):
        """returns a scipy interp1d object"""

        if assume_sorted is None:
            assume_sorted = is_sorted(self.time)

        if self.n_signals > 1:
            axis = 1
        else:
            axis = -1

        f = interpolate.interp1d(x=self.time,
                                 y=self._ydata_rowsig,
                                 kind=kind,
                                 axis=axis,
                                 copy=copy,
                                 bounds_error=bounds_error,
                                 fill_value=fill_value,
                                 assume_sorted=assume_sorted)
        return f

    def asarray(self,*, where=None, at=None, kind='linear', copy=True,
                bounds_error=False, fill_value=np.nan, assume_sorted=None,
                recalculate=False, store_interp=True, n_points=None,
                split_by_epoch=False):
        """returns a ydata_like array at requested points.

        Parameters
        ----------
        where : array_like or tuple, optional
            array corresponding to np where condition
            e.g., where=(ydata[1,:]>5) or tuple where=(speed>5,tspeed)
        at : array_like, optional
            Array of oints to evaluate array at. If none given, use
            self.tdata together with 'where' if applicable.
        n_points: int, optional
            Number of points to interplate at. These points will be
            distributed uniformly from self.support.start to stop.
        split_by_epoch: bool
            If True, separate arrays by epochs and return in a list.
        Returns
        -------
        out : (array, array)
            namedtuple tuple (xvals, yvals) of arrays, where xvals is an
            array of time points for which (interpolated) ydata are
            returned.
        """

        # TODO: implement splitting by epoch

        if split_by_epoch:
            raise NotImplementedError("split_by_epoch not yet implemented...")

        XYArray = namedtuple('XYArray', ['xvals', 'yvals'])

        if at is None and where is None and split_by_epoch is False and n_points is None:
            xyarray = XYArray(self.time, self._ydata_rowsig.squeeze())
            return xyarray

        if where is not None:
            assert at is None and n_points is None, "'where', 'at', and 'n_points' cannot be used at the same time"
            if isinstance(where, tuple):
                y = np.array(where[1]).squeeze()
                x = where[0]
                assert len(x) == len(y), "'where' condition and array must have same number of elements"
                at = y[x]
            else:
                x = np.asanyarray(where).squeeze()
                assert len(x) == len(self.time), "'where' condition must have same number of elements as self.time"
                at = self.time[x]
        elif at is not None:
            assert n_points is None, "'at' and 'n_points' cannot be used at the same time"
        else:
            at = np.linspace(self.support.start, self.support.stop, n_points)

        # if we made it this far, either at or where has been specified, and at is now well defined.

        kwargs = {'kind':kind,
                  'copy':copy,
                  'bounds_error':bounds_error,
                  'fill_value':fill_value,
                  'assume_sorted':assume_sorted}

        # retrieve an existing, or construct a new interpolation object
        if recalculate:
            interpobj = self._get_interp1d(**kwargs)
        else:
            try:
                interpobj = self._interp
                if interpobj is None:
                    interpobj = self._get_interp1d(**kwargs)
            except AttributeError: # does not exist yet
                interpobj = self._get_interp1d(**kwargs)

        # store interpolation object, if desired
        if store_interp:
            self._interp = interpobj

        # do the actual interpolation
        out = interpobj(at)

        # TODO: set all values outside of self.support to fill_value

        xyarray = XYArray(xvals=np.asanyarray(at), yvals=np.asanyarray(out).squeeze())
        return xyarray

    def simplify(self, *, ds=None, n_points=None):
        """Returns an AnalogSignalArray where the ydata has been
        simplified / subsampled.

        This function is primarily intended to be used for plotting and
        saving vector graphics without having too large file sizes as
        a result of too many points.

        Irrespective of whether 'ds' or 'n_points' are used, the exact
        underlying support is propagated, and the first and last points
        of the supports are always included, even if this would cause
        n_points or ds to be violated.

        Parameters
        ----------
        ds : float, optional
            Time (in seconds), in which to step points.
        n_points : int, optional
            Number of points at which to intepolate ydata. If ds is None
            and n_points is None, then default is to use n_points=5,000

        Returns
        -------
        out : AnalogSignalArray
            Copy of AnalogSignalArray where ydata is only stored at the
            new subset of points.
        """

        if self.isempty:
            return self

        if ds is not None and n_points is not None:
            raise ValueError("ds and n_points cannot be used together")

        if n_points is not None:
            assert float(n_points).is_integer(), "n_points must be a positive integer!"
            assert n_points > 1, "n_points must be a positive integer > 1"
            # determine ds from number of desired points:
            ds = self.support.duration / (n_points-1)

        if ds is None:
            # neither n_points nor ds was specified, so assume defaults:
            n_points = np.min((5000, 250+self.n_samples//2, self.n_samples))
            ds = self.support.duration / (n_points-1)

        # build list of points at which to evaluate the AnalogSignalArray
        at = []
        for start, stop in self.support.time:
            newxvals = np.arange(start, stop, step=ds).tolist()
            if newxvals[-1] + float_info.epsilon < stop:
                newxvals.append(stop)
            at.extend(newxvals)

        _, yvals = self.asarray(at=at, recalculate=True, store_interp=False)
        yvals = np.array(yvals, ndmin=2)

        # now make a new simplified ASA:
        asa = AnalogSignalArray([], empty=True)
        exclude = ['_interp', '_ydata', '_tdata', '_time']
        attrs = (x for x in self.__attributes__ if x not in exclude)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("asa." + attr + " = self." + attr)
        asa._tdata = np.asanyarray(at)
        asa._time = asa._tdata
        asa._ydata = yvals

        return asa

    def interp(self, event, *,store_interp=True):
        """Creates interpolate object if not created already via
        scipy.interpolate.interp1d

        Parameters
        ----------
        event : array-like elements upon which to interpolate values

        Returns
        -------
        interp_vals : np.array()
            numpy array of interpolated values in order of signals
            in AnalogSignalArray initially entered in constructor

        Examples
        --------
        >>> print("I will make examples soon :P")
        """
        warnings.warn("AnalogSignalArray.interp is deprecated.", DeprecationWarning)

        try:
            if(self._interp is not None):
                interp_vals = [interpObjectt(event) for interpObjectt in self._interp]
        except AttributeError:
            if(store_interp):
                self._interp = []
                for ydata in self._ydata:
                    self._interp.append(interpolate.interp1d(self._time, ydata))

                interp_vals = [interpObjectt(event) for interpObjectt in self._interp]
            else:
                tempInterpObj = []
                for ydata in self._ydata:
                    tempInterpObj.append(interpolate.interp1d(self._time, ydata))

                interp_vals = [interpObjectt(event) for interpObjectt in tempInterpObj]

        return np.asarray(interp_vals)

#----------------------------------------------------------------------#
#======================================================================#


########################################################################
# class SpikeTrainArray
########################################################################
class SpikeTrainArray(SpikeTrain):
    """A multiunit spiketrain array with shared support.

    Parameters
    ----------
    tdata : array (of length n_units) of np.array(dtype=np.float64)
        containing spike times in in seconds (unless fs=None).
    fs : float, optional
        Sampling rate in Hz. If fs is passed as a parameter, then time
        is assumed to be in sample numbers instead of actual time.
    support : EpochArray, optional
        EpochArray on which spiketrains are defined.
        Default is [0, last spike] inclusive.
    label : str or None, optional
        Information pertaining to the source of the spiketrain array.
    cell_type : list (of length n_units) of str or other, optional
        Identified cell type indicator, e.g., 'pyr', 'int'.
    unit_ids : list (of length n_units) of indices corresponding to
        curated data. If no unit_ids are specified, then [1,...,n_units]
        will be used. WARNING! The first unit will have index 1, not 0!
    meta : dict
        Metadata associated with spiketrain array.

    Attributes
    ----------
    time : array of np.array(dtype=np.float64) spike times in seconds.
        Array of length n_units, each entry with shape (n_tdata,)
    tdata : list of np.array(dtype=np.float64) spike times in tdata.
        Array of length n_units, each entry with shape (n_tdata,)
    support : EpochArray on which spiketrain array is defined.
    n_spikes: np.array(dtype=np.int) of shape (n_units,)
        Number of spikes in each unit.
    fs: float
        Sampling frequency (Hz).
    cell_types : np.array of str or other
        Identified cell type for each unit.
    label : str or None
        Information pertaining to the source of the spiketrain.
    meta : dict
        Metadata associated with spiketrain.
    """

    __attributes__ = ["_tdata", "_time", "_support"]
    __attributes__.extend(SpikeTrain.__attributes__)
    def __init__(self, tdata=None, *, fs=None, support=None,
                 unit_ids=None, unit_labels=None, unit_tags=None,
                 label=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = EpochArray(empty=True)
            return

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

        def is_single_unit(data):
            """Returns True if data represents spike times from a single unit.

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
                    warnings.warn("spike times input has too many layers!")
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
            if is_single_unit(data):
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

        tdata = standardize_to_2d(tdata)

        #sort spike trains, but only if necessary:
        for ii, train in enumerate(tdata):
            if not is_sorted(train):
                tdata[ii] = np.sort(train)

        kwargs = {"fs": fs,
                  "unit_ids": unit_ids,
                  "unit_labels": unit_labels,
                  "unit_tags": unit_tags,
                  "label": label}

        # initialize super so that self.fs is set:
        self._time = tdata  # this is necessary so that super() can
            # determine self.n_units when initializing. self.time will
            # be updated later in __init__ to reflect subsequent changes
        super().__init__(**kwargs)

        # if only empty tdata were received AND no support, attach an
        # empty support:
        if np.sum([st.size for st in tdata]) == 0 and support is None:
            warnings.warn("no spikes; cannot automatically determine support")
            support = EpochArray(empty=True)

        # if a sampling rate was given, relate time to tdata using fs:
        if fs is not None:
            time = tdata / fs
        else:
            time = tdata

        # determine spiketrain array support:
        if support is None:
            first_spk = np.array([unit[0] for unit in tdata if len(unit) !=0]).min()
            # BUG: if spiketrain is empty np.array([]) then unit[-1]
            # raises an error in the following:
            # FIX: list[-1] raises an IndexError for an empty list,
            # whereas list[-1:] returns an empty list.
            last_spk = np.array([unit[-1:] for unit in tdata if len(unit) !=0]).max()
            self._support = EpochArray(np.array([first_spk, last_spk]), fs=fs)
            # in the above, there's no reason to restrict to support
        else:
            # restrict spikes to only those within the spiketrain
            # array's support:
            self._support = support

            # if not support.isempty:
        time, tdata = self._restrict_to_epoch_array(
            epocharray=self._support,
            time=time,
            tdata=tdata)

        # if no tdata remain after restricting to the support, return
        # an empty SpikeTrainArray:
        # if np.sum([st.size for st in tdata]) == 0:
        #     print('wahoo!')
        #     return SpikeTrainArray(empty=True)

        # set self._tdata and self._time:
        self._time = time
        self._tdata = tdata

    def copy(self):
        """Returns a copy of the SpikeTrainArray."""
        newcopy = SpikeTrainArray(empty=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in self.__attributes__:
                exec("newcopy." + attr + " = self." + attr)
        return newcopy

    def __add__(self, other):
        """Overloaded + operator"""

        #TODO: additional checks need to be done, e.g., same unit ids...
        assert self.n_units == other.n_units
        support = self.support + other.support

        newdata = []
        for unit in range(self.n_units):
            newdata.append(np.append(self.time[unit], other.time[unit]))

        return SpikeTrainArray(newdata, support=support, fs=1)

    def __iter__(self):
        """SpikeTrainArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """SpikeTrainArray iterator advancer."""
        index = self._index
        if index > self.support.n_epochs - 1:
            raise StopIteration
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            support = self.support[index]
            time, tdata = self._restrict_to_epoch_array(
                epocharray=support,
                time=self.time,
                tdata=self.tdata,
                copy=True
                )
            spiketrain = SpikeTrainArray(empty=True)
            exclude = ["_tdata", "_time", "_support"]
            attrs = (x for x in self.__attributes__ if x not in exclude)
            for attr in attrs:
                exec("spiketrain." + attr + " = self." + attr)
            spiketrain._tdata = tdata
            spiketrain._time = time
            spiketrain._support = support
        self._index += 1
        return spiketrain

    def __getitem__(self, idx):
        """SpikeTrainArray index access."""
        # TODO: allow indexing of form sta[4,1:5] so that the STs of
        # epochs 1 to 5 (exlcusive) are returned, for neuron id 4.

        if self.isempty:
            return self

        if isinstance(idx, EpochArray):
            if idx.isempty:
                return SpikeTrainArray(empty=True)
            if idx.fs != self.support.fs:
                support = self.support.intersect(
                    epoch=EpochArray(idx.time, fs=None),
                    boundaries=True
                    )
            else:
                support = self.support.intersect(
                    epoch=idx,
                    boundaries=True
                    ) # what if fs of slicing epoch is different?
            if support.isempty:
                return SpikeTrainArray(empty=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                time, tdata = self._restrict_to_epoch_array(
                    epocharray=support,
                    time=self.time,
                    tdata=self.tdata,
                    copy=True
                    )
                spiketrain = SpikeTrainArray(empty=True)
                exclude = ["_tdata", "_time", "_support"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("spiketrain." + attr + " = self." + attr)
                spiketrain._tdata = tdata
                spiketrain._time = time
                spiketrain._support = support
            return spiketrain
        elif isinstance(idx, int):
            spiketrain = SpikeTrainArray(empty=True)
            exclude = ["_tdata", "_time", "_support"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("spiketrain." + attr + " = self." + attr)
                support = self.support[idx]
                spiketrain._support = support
            if (idx >= self.support.n_epochs) or idx < (-self.support.n_epochs):
                return spiketrain
            else:
                time, tdata = self._restrict_to_epoch_array(
                        epocharray=support,
                        time=self.time,
                        tdata=self.tdata,
                        copy=True
                        )
                spiketrain._tdata = tdata
                spiketrain._time = time
                spiketrain._support = support
                return spiketrain
        else:  # most likely slice indexing
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    support = self.support[idx]
                    time, tdata = self._restrict_to_epoch_array(
                        epocharray=support,
                        time=self.time,
                        tdata=self.tdata,
                        copy=True
                        )
                    spiketrain = SpikeTrainArray(empty=True)
                    exclude = ["_tdata", "_time", "_support"]
                    attrs = (x for x in self.__attributes__ if x not in exclude)
                    for attr in attrs:
                        exec("spiketrain." + attr + " = self." + attr)
                    spiketrain._tdata = tdata
                    spiketrain._time = time
                    spiketrain._support = support
                return spiketrain
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))

    @property
    def isempty(self):
        """(bool) Empty SpikeTrainArray."""
        try:
            return np.sum([len(st) for st in self.time]) == 0
        except TypeError:
            return True  # this happens when self.time == None

    @property
    def n_units(self):
        """(int) The number of units."""
        try:
            return PrettyInt(len(self.time))
        except TypeError:
            return 0

    @property
    def n_active(self):
        """(int) The number of active units.

        A unit is considered active if it fired at least one spike.
        """
        if self.isempty:
            return 0
        return PrettyInt(np.count_nonzero(self.n_spikes))

    def flatten(self, *, unit_id=None, unit_label=None):
        """Collapse spike trains across units.

        WARNING! unit_tags are thrown away when flattening.

        Parameters
        ----------
        unit_id: (int)
            (unit) ID to assign to flattened spike train, default is 0.
        unit_label (str)
            (unit) Label for spike train, default is 'flattened'.
        """
        if self.n_units < 2:  # already flattened
            return self

        # default args:
        if unit_id is None:
            unit_id = 0
        if unit_label is None:
            unit_label = "flattened"

        spiketrainarray = SpikeTrainArray(empty=True)

        exclude = ["_tdata", "_time", "unit_ids", "unit_labels", "unit_tags"]
        attrs = (x for x in self.__attributes__ if x not in exclude)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("spiketrainarray." + attr + " = self." + attr)
        spiketrainarray._unit_ids = [unit_id]
        spiketrainarray._unit_labels = [unit_label]
        spiketrainarray._unit_tags = None

        # TODO: here we linear merge twice; once for tdata and once for
        # time. This is unneccessary, and can be optimized. But flatten()
        # shouldn't be called often, so it's low priority,
        allspikes = self.tdata[0]
        for unit in range(1,self.n_units):
            allspikes = linear_merge(allspikes, self.tdata[unit])
        alltimes = self.time[0]
        for unit in range(1,self.n_units):
            alltimes = linear_merge(alltimes, self.time[unit])

        spiketrainarray._tdata = np.array(list(allspikes), ndmin=2)
        spiketrainarray._time = np.array(list(alltimes), ndmin=2)
        return spiketrainarray

    @staticmethod
    def _restrict_to_epoch_array(epocharray, time, tdata, copy=True):
        """Returns time and tdata restricted to an EpochArray.

        Parameters
        ----------
        epocharray : EpochArray
        """
        # Potential BUG: not sure if time and tdata point to same
        # object (like when only time was passed to __init__), then
        # doing tdata[unit] = ... followed by time[unit] = ... might
        # be applying the shrinking twice, no? We need a thorough test
        # for this! And I need to understand the shared memory 100%.

        if epocharray.isempty:
            n_units = len(tdata)
            time = np.zeros((n_units,0))
            tdata = np.zeros((n_units,0))
            return time, tdata

        singleunit = len(tdata)==1  # bool
        # TODO: upgrade this to use copy.copy or copy.deepcopy:
        if copy:
            time = time.copy()
            tdata = tdata.copy()
        # BUG: this assumes multiple units for the enumeration to work
        for unit, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                indices.append((st_time >= t_start) & (st_time <= t_stop))
            indices = np.any(np.column_stack(indices), axis=1)
            if np.count_nonzero(indices) < len(st_time):
                warnings.warn(
                    'ignoring spikes outside of spiketrain support')
            if singleunit:
                tdata = np.array([tdata[0,indices]], ndmin=2)
                time = np.array([time[0,indices]], ndmin=2)
            else:
                tdata[unit] = tdata[unit][indices]
                time[unit] = time[unit][indices]
        return time, tdata

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.isempty:
                return "<empty SpikeTrainArray" + address_str + ">"
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
            numstr = " %s units" % self.n_units
        return "<SpikeTrainArray%s:%s%s>%s%s" % (address_str, numstr, epstr, fsstr, labelstr)

    def bin(self, *, ds=None):
        """Return a binned spiketrain array."""
        return BinnedSpikeTrainArray(self, ds=ds)

    @property
    def tdata(self):
        """Spike times in sample numbers (default fs = 1 Hz)."""
        return self._tdata

    @property
    def time(self):
        """Spike times in seconds."""
        return self._time

    @property
    def n_spikes(self):
        """(np.array) The number of spikes in each unit."""
        if self.isempty:
            return 0
        return np.array([len(unit) for unit in self.time])

    @property
    def issorted(self):
        """(bool) Sorted SpikeTrainArray."""
        if self.isempty:
            return True
        return np.array(
            [is_sorted(spiketrain) for spiketrain in self.tdata]
            ).all()

    def reorder_units(self, neworder):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,)
        """
        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            swap_rows(self._time, frm, to)
            swap_rows(self._tdata, frm, to)
            self._unit_ids[frm], self._unit_ids[to] = self._unit_ids[to], self._unit_ids[frm]
            self._unit_labels[frm], self._unit_labels[to] = self._unit_labels[to], self._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
#----------------------------------------------------------------------#
#======================================================================#


########################################################################
# class BinnedSpikeTrainArray
########################################################################
class BinnedSpikeTrainArray(SpikeTrain):
    """Binned spiketrain array.

    Parameters
    ----------
    fs : float, optional
        Sampling rate in Hz. If fs is passed as a parameter, then time
        is assumed to be in sample numbers instead of actual time.

    Attributes
    ----------
    time : np.array
        The start and stop times for each epoch. With shape (n_epochs, 2).
    """

    __attributes__ = ["_ds", "_bins", "_data", "_bin_centers", "_support",
                      "_binnedSupport", "_spiketrainarray"]
    __attributes__.extend(SpikeTrain.__attributes__)

    def __init__(self, spiketrainarray=None, *, ds=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = EpochArray(empty=True)
            self._event_centers = None
            return

        if not isinstance(spiketrainarray, SpikeTrainArray):
            raise TypeError(
                'spiketrainarray must be a nelpy.SpikeTrainArray object.')

        self._ds = None
        self._bin_centers = np.array([])
        self._event_centers = None

        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kwargs = {"fs": spiketrainarray.fs,
                        "unit_ids": spiketrainarray.unit_ids,
                        "unit_labels": spiketrainarray.unit_labels,
                        "unit_tags": spiketrainarray.unit_tags,
                        "label": spiketrainarray.label}

        # initialize super so that self.fs is set:
        self._data = np.zeros((spiketrainarray.n_units,0))
            # the above is necessary so that super() can determine
            # self.n_units when initializing. self.time will
            # be updated later in __init__ to reflect subsequent changes
        super().__init__(**kwargs)

        if ds is None:
            warnings.warn('no bin size was given, assuming 62.5 ms')
            ds = 0.0625

        self._spiketrainarray = spiketrainarray # TODO: remove this if we don't need it, or decide that it's too wasteful
        # self._support = spiketrainarray.support
        self.ds = ds

        self._bin_spikes(
            spiketrainarray=spiketrainarray,
            epochArray=spiketrainarray.support,
            ds=ds
            )

    def copy(self):
        """Returns a copy of the BinnedSpikeTrainArray."""
        newcopy = BinnedSpikeTrainArray(empty=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in self.__attributes__:
                exec("newcopy." + attr + " = self." + attr)
        return newcopy

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty BinnedSpikeTrainArray" + address_str + ">"
        ustr = " {} units".format(self.n_units)
        if self.support.n_epochs > 1:
            epstr = " ({} segments) in".format(self.support.n_epochs)
        else:
            epstr = " in"
        if self.n_bins == 1:
            bstr = " {} bin of width {}".format(self.n_bins, PrettyDuration(self.ds))
            dstr = ""
        else:
            bstr = " {} bins of width {}".format(self.n_bins, PrettyDuration(self.ds))
            dstr = " for a total of {}".format(PrettyDuration(self.n_bins*self.ds))
        return "<BinnedSpikeTrainArray%s:%s%s%s>%s" % (address_str, ustr, epstr, bstr, dstr)

    def __iter__(self):
        """BinnedSpikeTrainArray iterator initialization."""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """BinnedSpikeTrainArray iterator advancer."""
        index = self._index

        if index > self.support.n_epochs - 1:
            raise StopIteration

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            support = self.support[index]
            bsupport = self.binnedSupport[[index],:]

            binnedspiketrain = BinnedSpikeTrainArray(empty=True)
            exclude = ["_bins", "_data", "_support", "_bin_centers", "_binnedSupport"]
            attrs = (x for x in self.__attributes__ if x not in exclude)
            for attr in attrs:
                exec("binnedspiketrain." + attr + " = self." + attr)
            binindices = np.insert(0, 1, np.cumsum(self.lengths + 1)) # indices of bins
            binstart = binindices[index]
            binstop = binindices[index+1]
            binnedspiketrain._bins = self._bins[binstart:binstop]
            binnedspiketrain._data = self._data[:,bsupport[0][0]:bsupport[0][1]+1]
            binnedspiketrain._support = support
            binnedspiketrain._bin_centers = self._bin_centers[bsupport[0][0]:bsupport[0][1]+1]
            binnedspiketrain._binnedSupport = bsupport - bsupport[0,0]
        self._index += 1
        return binnedspiketrain

    def __getitem__(self, idx):
        """BinnedSpikeTrainArray index access."""
        if self.isempty:
            return self
        if isinstance(idx, EpochArray):
            # need to determine if there is any proper subset in self.support intersect EpochArray
            # next, we need to identify all the bins that would fall within the EpochArray

            if idx.isempty:
                return BinnedSpikeTrainArray(empty=True)
            if idx.fs != self.support.fs:
                support = self.support.intersect(
                    epoch=EpochArray(idx.time, fs=None),
                    boundaries=True
                    )
            else:
                support = self.support.intersect(
                    epoch=idx,
                    boundaries=True
                    ) # what if fs of slicing epoch is different?
            if support.isempty:
                return BinnedSpikeTrainArray(empty=True)
            # next we need to determine the binnedSupport:

            raise NotImplementedError("EpochArray indexing for BinnedSpikeTrainArrays not supported yet")

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")

            #     time, tdata = self._restrict_to_epoch_array(
            #         epocharray=support,
            #         time=self.time,
            #         tdata=self.tdata,
            #         copy=True
            #         )
            #     spiketrain = SpikeTrainArray(empty=True)
            #     exclude = ["_tdata", "_time", "_support"]
            #     attrs = (x for x in self.__attributes__ if x not in exclude)
            #     for attr in attrs:
            #         exec("spiketrain." + attr + " = self." + attr)
            #     spiketrain._tdata = tdata
            #     spiketrain._time = time
            #     spiketrain._support = support
            # return spiketrain

        elif isinstance(idx, int):
            binnedspiketrain = BinnedSpikeTrainArray(empty=True)
            exclude = ["_data", "_bins", "_support", "_bin_centers", "_spiketrainarray", "_binnedSupport"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("binnedspiketrain." + attr + " = self." + attr)
            support = self.support[idx]
            binnedspiketrain._support = support
            if (idx >= self.support.n_epochs) or idx < (-self.support.n_epochs):
                return binnedspiketrain
            else:
                bsupport = self.binnedSupport[[idx],:]
                centers = self._bin_centers[bsupport[0,0]:bsupport[0,1]+1]
                binindices = np.insert(0, 1, np.cumsum(self.lengths + 1)) # indices of bins
                binstart = binindices[idx]
                binstop = binindices[idx+1]
                binnedspiketrain._data = self._data[:,bsupport[0,0]:bsupport[0,1]+1]
                binnedspiketrain._bins = self._bins[binstart:binstop]
                binnedspiketrain._binnedSupport = bsupport - bsupport[0,0]
                binnedspiketrain._bin_centers = centers
                return binnedspiketrain
        else:  # most likely a slice
            try:
                # have to be careful about re-indexing binnedSupport
                binnedspiketrain = BinnedSpikeTrainArray(empty=True)
                exclude = ["_data", "_bins", "_support", "_bin_centers", "_spiketrainarray", "_binnedSupport"]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    attrs = (x for x in self.__attributes__ if x not in exclude)
                    for attr in attrs:
                        exec("binnedspiketrain." + attr + " = self." + attr)
                support = self.support[idx]
                binnedspiketrain._support = support

                bsupport = self.binnedSupport[idx,:] # need to re-index!
                # now build a list of all elements in bsupport:
                ll = []
                for bs in bsupport:
                    ll.extend(np.arange(bs[0],bs[1]+1, step=1))
                binnedspiketrain._bin_centers = self._bin_centers[ll]
                binnedspiketrain._data = self._data[:,ll]

                lengths = self.lengths[[idx]]
                # lengths = bsupport[:,1] - bsupport[:,0]
                bsstarts = np.insert(np.cumsum(lengths),0,0)[:-1]
                bsends = np.cumsum(lengths) - 1
                binnedspiketrain._binnedSupport = np.vstack((bsstarts, bsends)).T

                binindices = np.insert(0, 1, np.cumsum(self.lengths + 1)) # indices of bins
                binstarts = binindices[idx]
                binstops = binindices[1:][idx]  # equivalent to binindices[idx + 1], but if idx is a slice, we can't add 1 to it
                ll = []
                for start, stop in zip(binstarts, binstops):
                    ll.extend(np.arange(start,stop,step=1))
                binnedspiketrain._bins = self._bins[ll]

                return binnedspiketrain
            except Exception:
                raise TypeError(
                    'unsupported indexing type {}'.format(type(idx)))

    @property
    def isempty(self):
        """(bool) Empty BinnedSpikeTrainArray."""
        try:
            return len(self.centers) == 0
        except TypeError:
            return True  # this happens when self.centers == None

    @property
    def n_units(self):
        """(int) The number of units."""
        try:
            return PrettyInt(self.data.shape[0])
        except AttributeError:
            return 0

    @property
    def centers(self):
        """(np.array) The bin centers (in seconds)."""
        warnings.warn("centers is deprecated. Use bin_centers instead.", DeprecationWarning)
        return self.bin_centers

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
        ax, img = npl.imagesc(bst.data) # data is of shape (n_units, n_bins)
        # then _midpoints correspond to the xvals at the center of
        # each event.
        ax.plot(bst.event_centers, np.repeat(1, self.n_epochs), marker='o', color='w')

        """
        if self._event_centers is None:
            midpoints = np.zeros(len(self.lengths))
            for idx, l in enumerate(self.lengths):
                midpoints[idx] = np.sum(self.lengths[:idx]) + l/2
            self._event_centers = midpoints
        return self._event_centers

    @property
    def data(self):
        """(np.array) The spike counts in all the bins.
        See also BinnedSpikeTrain.centers
        """
        return self._data

    @property
    def bins(self):
        """(np.array) The bin edges (in seconds)."""
        return self._bins

    @property
    def binnedSupport(self):
        """(np.array) The binned support of the binned spiketrain (in
        bin IDs) of shape (n_epochs, 2).
        """
        return self._binnedSupport

    @property
    def lengths(self):
        """Lenghts of contiguous segments, in number of bins."""
        return (self.binnedSupport[:,1] - self.binnedSupport[:,0] + 1).squeeze()

    @property
    def spiketrainarray(self):
        """(nelpy.SpikeTrain) The original spiketrain associated with
        the binned data.
        """
        return self._spiketrainarray

    @property
    def n_bins(self):
        """(int) The number of bins."""
        return PrettyInt(len(self.centers))

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
    def _get_bins_inside_epoch(epoch, ds):
        """(np.array) Return bin edges entirely contained inside an epoch.

        Bin edges always start at epoch.start, and continue for as many
        bins as would fit entirely inside the epoch.

        NOTE 1: there are (n+1) bin edges associated with n bins.

        WARNING: if an epoch is smaller than ds, then no bin will be
                associated with the particular epoch.

        NOTE 2: nelpy uses half-open intervals [a,b), but if the bin
                width divides b-a, then the bins will cover the entire
                range. For example, if epoch = [0,2) and ds = 1, then
                bins = [0,1,2], even though [0,2] is not contained in
                [0,2).

        Parameters
        ----------
        epoch : EpochArray
            EpochArray containing a single epoch with a start, and stop
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

        if epoch.duration < ds:
            warnings.warn(
                "epoch duration is less than bin size: ignoring...")
            return None, None

        n = int(np.floor(epoch.duration / ds)) # number of bins

        # linspace is better than arange for non-integral steps
        bins = np.linspace(epoch.start, epoch.start + n*ds, n+1)
        centers = bins[:-1] + (ds / 2)
        return bins, centers

    def _bin_spikes(self, spiketrainarray, epochArray, ds):
        """
        Docstring goes here. TBD. For use with bins that are contained
        wholly inside the epochs.

        """
        b = []  # bin list
        c = []  # centers list
        s = []  # data list
        for nn in range(spiketrainarray.n_units):
            s.append([])
        left_edges = []
        right_edges = []
        counter = 0
        for epoch in epochArray:
            bins, centers = self._get_bins_inside_epoch(epoch, ds)
            if bins is not None:
                for uu, spiketraintimes in enumerate(spiketrainarray.time):
                    spike_counts, _ = np.histogram(
                        spiketraintimes,
                        bins=bins,
                        density=False,
                        range=(epoch.start,epoch.stop)
                        ) # TODO: is it faster to limit range, or to cut out spikes?
                    s[uu].extend(spike_counts.tolist())
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
        self._support = EpochArray(supportdata, fs=1) # set support to TRUE bin support

    def smooth(self, *, sigma=None, inplace=False,  bw=None):
        """Smooth BinnedSpikeTrainArray by convolving with a Gaussian kernel.

        Smoothing is applied in time, and the same smoothing is applied
        to each unit in a BinnedSpikeTrainArray.

        Smoothing is applied within each epoch.

        Parameters
        ----------
        sigma : float, optional
            Standard deviation of Gaussian kernel, in seconds. Default is 0.01 (10 ms)
        bw : float, optional
            Bandwidth outside of which the filter value will be zero. Default is 4.0
        inplace : bool
            If True the data will be replaced with the smoothed data.
            Default is False.

        Returns
        -------
        out : BinnedSpikeTrainArray
            New BinnedSpikeTrainArray with smoothed data.
        """

        if bw is None:
            bw=4
        if sigma is None:
            sigma = 0.01 # 10 ms default

        fs = 1 / self.ds

        return gaussian_filter(self, fs=fs, sigma=sigma, inplace=inplace)

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
            Number of original bins to combine into each new bin.ABC

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
        """Rebin the BinnedSpikeTrainArray into a coarser bin size.

        Parameters
        ----------
        w : int, optional
            number of bins of width bst.ds to bin into new bin of
            width bst.ds*w. Default is w=1 (no re-binning).

        Returns
        -------
        out : BinnedSpikeTrainArray
            New BinnedSpikeTrainArray with coarser resolution.
        """

        if w is None:
            w = 1

        if not float(w).is_integer:
            raise ValueError("w has to be an integer!")

        w = int(w)

        bst = self
        return self._rebin_binnedspiketrain(bst, w=w)

    @staticmethod
    def _rebin_binnedspiketrain(bst, w=None):
        """Rebin a BinnedSpikeTrainArray into a coarser bin size.

        Parameters
        ----------
        bst : BinnedSpikeTrainArray
            BinnedSpikeTrainArray to re-bin into a coarser resolution.
        w : int, optional
            number of bins of width bst.ds to bin into new bin of
            width bst.ds*w. Default is w=1 (no re-binning).

        Returns
        -------
        out : BinnedSpikeTrainArray
            New BinnedSpikeTrainArray with coarser resolution.

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
        n_events = bst.support.n_epochs
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

        # assemble new binned spike train array:
        newedges = np.cumsum(newlengths)
        newbst = BinnedSpikeTrainArray(empty=True)
        if newdata is not None:
            newbst._data = newdata
            newbst._support = EpochArray(newsupport, fs=1)
            newbst._bins = newbins
            newbst._bin_centers = newcenters
            newbst._ds = bst.ds*w
            newbst._binnedSupport = np.array((newedges[:-1], newedges[1:]-1)).T
        else:
            warnings.warn("No events are long enough to contain any bins of width {}".format(PrettyDuration(ds)))

        return newbst

    def _bin_spikes_old(self, spiketrainarray, epochArray, ds):
        b = []  # bin list
        c = []  # centers list
        s = []  # data list
        for nn in range(spiketrainarray.n_units):
            s.append([])
        left_edges = []
        right_edges = []
        counter = 0
        for epoch in epochArray:
            bins, centers = self._get_bins_to_cover_epoch(epoch, ds)
            for uu, spiketraintimes in enumerate(spiketrainarray.time):
                spike_counts, _ = np.histogram(
                    spiketraintimes,
                    bins=bins,
                    density=False,
                    range=(epoch.start,epoch.stop)
                    ) # TODO: is it faster to limit range, or to cut out spikes?
                s[uu].extend(spike_counts.tolist())
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

    @property
    def n_active(self):
        """Number of active units.

        An active unit is any unit that fired at least one spike.
        """
        if self.isempty:
            return 0
        return PrettyInt(np.count_nonzero(self.n_spikes))

    @property
    def n_active_per_bin(self):
        """Number of active units per time bin with shape (n_bins,)."""
        if self.isempty:
            return 0
        # TODO: profile several alternatves. Could use data > 0, or
        # other numpy methods to get a more efficient implementation:
        return self.data.clip(max=1).sum(axis=0)

    @property
    def n_spikes(self):
        """(np.array) The number of spikes in each unit."""
        if self.isempty:
            return 0
        return self.data.sum(axis=1)

    def flatten(self, *, unit_id=None, unit_label=None):
        """Collapse spike trains across units.

        WARNING! unit_tags are thrown away when flattening.

        Parameters
        ----------
        unit_id: (int)
            (unit) ID to assign to flattened spike train, default is 0.
        unit_label (str)
            (unit) Label for spike train, default is 'flattened'.
        """
        if self.n_units < 2:  # already flattened
            return self

        # default args:
        if unit_id is None:
            unit_id = 0
        if unit_label is None:
            unit_label = "flattened"

        binnedspiketrainarray = BinnedSpikeTrainArray(empty=True)

        exclude = ["_data", "unit_ids", "unit_labels", "unit_tags"]
        attrs = (x for x in self.__attributes__ if x not in "_data")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("binnedspiketrainarray." + attr + " = self." + attr)
        binnedspiketrainarray._data = np.array(self.data.sum(axis=0), ndmin=2)
        binnedspiketrainarray._unit_ids = [unit_id]
        binnedspiketrainarray._unit_labels = [unit_label]
        binnedspiketrainarray._unit_tags = None
        return binnedspiketrainarray

#----------------------------------------------------------------------#
#======================================================================#
