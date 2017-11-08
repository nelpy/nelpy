__all__ = ['SpikeTrainArray',
           'BinnedSpikeTrainArray']

import warnings
import numpy as np
import copy

from abc import ABC, abstractmethod

# from ..utils import is_sorted, \
#                    linear_merge, \
#                    PrettyDuration, \
#                    PrettyInt, \
#                    swap_rows, \
#                    gaussian_filter

from .. import utils

from ..utils_.decorators import deprecated

from ._epocharray import EpochArray

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

class EpochUnitSlicer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """epochs, units"""
        # by default, keep all units
        unitslice = slice(None, None, None)
        if isinstance(*args, int):
            epochslice = args[0]
        elif isinstance(*args, EpochArray):
            epochslice = args[0]
        else:
            try:
                slices = np.s_[args]; slices = slices[0]
                if len(slices) > 2:
                    raise IndexError("only [epochs, units] slicing is supported at this time!")
                elif len(slices) == 2:
                    epochslice, unitslice = slices
                else:
                    epochslice = slices[0]
            except TypeError:
                # only epoch to slice:
                epochslice = slices

        return epochslice, unitslice

class ItemGetter_loc(object):
    """.loc is primarily label based (that is, unit_id based)

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
        """epochs, units"""
        epochslice, unitslice = self.obj._slicer[idx]

        # first convert unit slice into list
        if isinstance(unitslice, slice):
            start = unitslice.start
            stop = unitslice.stop
            istep = unitslice.step
            try:
                if start is None:
                    istart = 0
                else:
                    istart = self.obj._unit_ids.index(start)
            except ValueError:
                raise KeyError('unit_id {} could not be found in SpikeTrain!'.format(start))
            try:
                if stop is None:
                    istop = self.obj.n_units
                else:
                    istop = self.obj._unit_ids.index(stop) + 1
            except ValueError:
                raise KeyError('unit_id {} could not be found in SpikeTrain!'.format(stop))
            if istep is None:
                istep = 1
            if istep < 0:
                istop -=1
                istart -=1
                istart, istop = istop, istart
            unit_idx_list = list(range(istart, istop, istep))
        else:
            unit_idx_list = []
            unitslice = np.atleast_1d(unitslice)
            for unit in unitslice:
                try:
                    uidx = self.obj.unit_ids.index(unit)
                except ValueError:
                    raise KeyError("unit_id {} could not be found in SpikeTrain!".format(unit))
                else:
                    unit_idx_list.append(uidx)

        if not isinstance(unit_idx_list, list):
            unit_idx_list = list(unit_idx_list)
        out = copy.copy(self.obj)
        out._time = out._time[unit_idx_list]
        singleunit = len(out._time)==1
        if singleunit:
            out._time = np.array(out._time[0], ndmin=2)
        out._unit_ids = list(np.atleast_1d(np.atleast_1d(out._unit_ids)[unit_idx_list]))
        out._unit_labels = list(np.atleast_1d(np.atleast_1d(out._unit_labels)[unit_idx_list]))
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
        """epochs, units"""
        epochslice, unitslice = self.obj._slicer[idx]
        out = copy.copy(self.obj)
        if isinstance(unitslice, int):
            unitslice = [unitslice]
        out._time = out._time[unitslice]
        singleunit = len(out._time)==1
        if singleunit:
            out._time = np.array(out._time[0], ndmin=2)
        out._unit_ids = list(np.atleast_1d(np.atleast_1d(out._unit_ids)[unitslice]))
        out._unit_labels = list(np.atleast_1d(np.atleast_1d(out._unit_labels)[unitslice]))
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
            self._slicer = EpochUnitSlicer(self)
            self.loc = ItemGetter_loc(self)
            self.iloc = ItemGetter_iloc(self)
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

        self._slicer = EpochUnitSlicer(self)
        self.loc = ItemGetter_loc(self)
        self.iloc = ItemGetter_iloc(self)

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        return "<base SpikeTrain" + address_str + ">"

    def partition(self, ds=None, n_epochs=None):
        """Returns an SpikeTrain whose support has been partitioned.

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
        out : SpikeTrain
            SpikeTrain that has been partitioned.
        """

        out = copy.copy(self)
        out._support = out.support.partition(ds=ds, n_epochs=n_epochs)
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

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
            exclude = ["_time", "unit_ids", "unit_labels"]
            attrs = (x for x in self.__attributes__ if x not in exclude)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for attr in attrs:
                    exec("spiketrainarray." + attr + " = self." + attr)

            spiketrainarray._time = self.time[unit_subset_ids]
            spiketrainarray._unit_ids = new_unit_ids
            spiketrainarray._unit_labels = new_unit_labels
            spiketrainarray.loc = ItemGetter_loc(spiketrainarray)
            spiketrainarray.iloc = ItemGetter_iloc(spiketrainarray)

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
            binnedspiketrainarray.loc = ItemGetter_loc(binnedspiketrainarray)
            binnedspiketrainarray.iloc = ItemGetter_iloc(binnedspiketrainarray)

            return binnedspiketrainarray
        else:
            raise NotImplementedError(
            "SpikeTrain._unit_slice() not supported for this type yet!")


########################################################################
# class SpikeTrainArray
########################################################################
class SpikeTrainArray(SpikeTrain):
    """A multiunit spiketrain array with shared support.

    Parameters
    ----------
    time : array of np.array(dtype=np.float64) spike times in seconds.
        Array of length n_units, each entry with shape (n_time,)
    fs : float, optional
        Sampling rate in Hz. Default is 30,000
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
        Array of length n_units, each entry with shape (n_time,)
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

    __attributes__ = ["_time", "_support"]
    __attributes__.extend(SpikeTrain.__attributes__)
    def __init__(self, timestamps=None, *, fs=None, support=None,
                 unit_ids=None, unit_labels=None, unit_tags=None,
                 label=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = EpochArray(empty=True)
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

        time = standardize_to_2d(timestamps)

        #sort spike trains, but only if necessary:
        for ii, train in enumerate(time):
            if not utils.is_sorted(train):
                time[ii] = np.sort(train)

        kwargs = {"fs": fs,
                  "unit_ids": unit_ids,
                  "unit_labels": unit_labels,
                  "unit_tags": unit_tags,
                  "label": label}

        self._time = time  # this is necessary so that
        # super() can determine self.n_units when initializing.

        # initialize super so that self.fs is set:
        super().__init__(**kwargs)

        # if only empty time were received AND no support, attach an
        # empty support:
        if np.sum([st.size for st in time]) == 0 and support is None:
            warnings.warn("no spikes; cannot automatically determine support")
            support = EpochArray(empty=True)

        # determine spiketrain array support:
        if support is None:
            first_spk = np.array([unit[0] for unit in time if len(unit) !=0]).min()
            # BUG: if spiketrain is empty np.array([]) then unit[-1]
            # raises an error in the following:
            # FIX: list[-1] raises an IndexError for an empty list,
            # whereas list[-1:] returns an empty list.
            last_spk = np.array([unit[-1:] for unit in time if len(unit) !=0]).max()
            self._support = EpochArray(np.array([first_spk, last_spk + 1/fs]))
            # in the above, there's no reason to restrict to support
        else:
            # restrict spikes to only those within the spiketrain
            # array's support:
            self._support = support

        # TODO: if sorted, we may as well use the fast restrict here as well?
        time = self._restrict_to_epoch_array(
            epocharray=self._support,
            time=time)

        self._time = time

    def copy(self):
        """Returns a copy of the SpikeTrainArray."""
        newcopy = SpikeTrainArray(empty=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in self.__attributes__:
                exec("newcopy." + attr + " = self." + attr)
        newcopy.loc = ItemGetter_loc(newcopy)
        newcopy.iloc = ItemGetter_iloc(newcopy)
        return newcopy

    def __add__(self, other):
        """Overloaded + operator"""

        #TODO: additional checks need to be done, e.g., same unit ids...
        assert self.n_units == other.n_units
        support = self.support + other.support

        newdata = []
        for unit in range(self.n_units):
            newdata.append(np.append(self.time[unit], other.time[unit]))

        fs = self.fs
        if self.fs != other.fs:
            fs = None
        return SpikeTrainArray(newdata, support=support, fs=fs)

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
            time = self._restrict_to_epoch_array_fast(
                epocharray=support,
                time=self.time,
                copyover=True
                )
            spiketrain = SpikeTrainArray(empty=True)
            exclude = ["_time", "_support"]
            attrs = (x for x in self.__attributes__ if x not in exclude)
            for attr in attrs:
                exec("spiketrain." + attr + " = self." + attr)
            spiketrain._time = time
            spiketrain._support = support
            spiketrain.loc = ItemGetter_loc(spiketrain)
            spiketrain.iloc = ItemGetter_iloc(spiketrain)
        self._index += 1
        return spiketrain

    def _epochslicer(self, idx):
        """Helper function to restrict object to EpochArray."""
        # if self.isempty:
        #     return self

        if isinstance(idx, EpochArray):
            if idx.isempty:
                return SpikeTrainArray(empty=True)
            support = self.support.intersect(
                    epoch=idx,
                    boundaries=True
                    ) # what if fs of slicing epoch is different?
            if support.isempty:
                return SpikeTrainArray(empty=True)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                time = self._restrict_to_epoch_array_fast(
                    epocharray=support,
                    time=self.time,
                    copyover=True
                    )
                spiketrain = SpikeTrainArray(empty=True)
                exclude = ["_time", "_support"]
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("spiketrain." + attr + " = self." + attr)
                spiketrain._time = time
                spiketrain._support = support
                spiketrain.loc = ItemGetter_loc(spiketrain)
                spiketrain.iloc = ItemGetter_iloc(spiketrain)
            return spiketrain
        elif isinstance(idx, int):
            spiketrain = SpikeTrainArray(empty=True)
            exclude = ["_time", "_support"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                attrs = (x for x in self.__attributes__ if x not in exclude)
                for attr in attrs:
                    exec("spiketrain." + attr + " = self." + attr)
                support = self.support[idx]
                spiketrain._support = support
            if (idx >= self.support.n_epochs) or idx < (-self.support.n_epochs):
                spiketrain.loc = ItemGetter_loc(spiketrain)
                spiketrain.iloc = ItemGetter_iloc(spiketrain)
                return spiketrain
            else:
                time = self._restrict_to_epoch_array_fast(
                        epocharray=support,
                        time=self.time,
                        copyover=True
                        )
                spiketrain._time = time
                spiketrain._support = support
                spiketrain.loc = ItemGetter_loc(spiketrain)
                spiketrain.iloc = ItemGetter_iloc(spiketrain)
                return spiketrain
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
                    spiketrain = SpikeTrainArray(empty=True)
                    exclude = ["_time", "_support"]
                    attrs = (x for x in self.__attributes__ if x not in exclude)
                    for attr in attrs:
                        exec("spiketrain." + attr + " = self." + attr)
                    spiketrain._time = time
                    spiketrain._support = support
                    spiketrain.loc = ItemGetter_loc(spiketrain)
                    spiketrain.iloc = ItemGetter_iloc(spiketrain)
                return spiketrain
            except Exception:
                raise TypeError(
                    'unsupported subsctipting type {}'.format(type(idx)))


    def __getitem__(self, idx):
        """SpikeTrainArray index access.

        By default, this method is bound to SpikeTrainArray.loc
        """
        return self.loc[idx]

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
            return utils.PrettyInt(len(self.time))
        except TypeError:
            return 0

    @property
    def n_active(self):
        """(int) The number of active units.

        A unit is considered active if it fired at least one spike.
        """
        if self.isempty:
            return 0
        return utils.PrettyInt(np.count_nonzero(self.n_spikes))

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

        exclude = ["_time", "unit_ids", "unit_labels", "unit_tags"]
        attrs = (x for x in self.__attributes__ if x not in exclude)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for attr in attrs:
                exec("spiketrainarray." + attr + " = self." + attr)
        spiketrainarray._unit_ids = [unit_id]
        spiketrainarray._unit_labels = [unit_label]
        spiketrainarray._unit_tags = None

        alltimes = self.time[0]
        for unit in range(1,self.n_units):
            alltimes = utils.linear_merge(alltimes, self.time[unit])

        spiketrainarray._time = np.array(list(alltimes), ndmin=2)
        spiketrainarray.loc = ItemGetter_loc(spiketrainarray)
        spiketrainarray.iloc = ItemGetter_iloc(spiketrainarray)
        return spiketrainarray

    @staticmethod
    def _restrict_to_epoch_array_fast(epocharray, time, copyover=True):
        """Return time restricted to an EpochArray.

        This function assumes sorted spike times, so that binary search can
        be used to quickly identify slices that should be kept in the
        restriction. It does not check every spike time.

        Parameters
        ----------
        epocharray : EpochArray
        time : array-like
        """
        if epocharray.isempty:
            n_units = len(time)
            time = np.zeros((n_units,0))
            return time

        singleunit = len(time)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            time = copy.copy(time)

        # NOTE: this used to assume multiple units for the enumeration to work
        for unit, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                frm, to = np.searchsorted(st_time, (t_start, t_stop))
                indices.append((frm, to))
            indices = np.array(indices, ndmin=2)
            if np.diff(indices).sum() < len(st_time):
                warnings.warn(
                    'ignoring spikes outside of spiketrain support')
            if singleunit:
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
                time_[unit] = np.array(time_list)
                time = np.array(time_)
        return time

    @staticmethod
    def _restrict_to_epoch_array(epocharray, time, copyover=True):
        """Return time restricted to an EpochArray.

        This function is quite slow, as it checks each spike time for inclusion.
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
            n_units = len(time)
            time = np.zeros((n_units,0))
            return time

        singleunit = len(time)==1  # bool

        # TODO: is this copy even necessary?
        if copyover:
            time = copy.copy(time)

        # NOTE: this used to assume multiple units for the enumeration to work
        for unit, st_time in enumerate(time):
            indices = []
            for eptime in epocharray.time:
                t_start = eptime[0]
                t_stop = eptime[1]
                indices.append((st_time >= t_start) & (st_time < t_stop))
            indices = np.any(np.column_stack(indices), axis=1)
            if np.count_nonzero(indices) < len(st_time):
                warnings.warn(
                    'ignoring spikes outside of spiketrain support')
            if singleunit:
                time = np.array([time[0][indices]], ndmin=2)
            else:
                # here we have to do some annoying conversion between
                # arrays and lists to fully support jagged array
                # mutation
                time_ = time.tolist()
                time_[unit] = np.array(time_[unit])
                time_[unit] = time_[unit][indices]
                time = np.array(time_)
        return time

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
    def time(self):
        """Spike times in seconds."""
        return self._time

    @property
    @deprecated
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
            [utils.is_sorted(spiketrain) for spiketrain in self.time]
            ).all()

    def _reorder_units_by_idx(self, neworder, inplace=False):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,)

        Return
        ------
        out : reordered SpikeTrainArray
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
            out._unit_ids[frm], out._unit_ids[to] = out._unit_ids[to], out._unit_ids[frm]
            out._unit_labels[frm], out._unit_labels[to] = out._unit_labels[to], out._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

    def reorder_units(self, neworder, *, inplace=False):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,) and in terms of
        unit_ids

        Return
        ------
        out : reordered SpikeTrainArray
        """
        raise DeprecationWarning("reorder_units has been deprecated. Use reorder_units_by_id(x/s) instead!")

    def reorder_units_by_ids(self, neworder, *, inplace=False):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,) and in terms of
        unit_ids

        Return
        ------
        out : reordered SpikeTrainArray
        """
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        neworder = [self.unit_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._time, frm, to)
            out._unit_ids[frm], out._unit_ids[to] = out._unit_ids[to], out._unit_ids[frm]
            out._unit_labels[frm], out._unit_labels[to] = out._unit_labels[to], out._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out

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
        newcopy.loc = ItemGetter_loc(newcopy)
        newcopy.iloc = ItemGetter_iloc(newcopy)
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
            bstr = " {} bin of width {}".format(self.n_bins, utils.PrettyDuration(self.ds))
            dstr = ""
        else:
            bstr = " {} bins of width {}".format(self.n_bins, utils.PrettyDuration(self.ds))
            dstr = " for a total of {}".format(utils.PrettyDuration(self.n_bins*self.ds))
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
        binnedspiketrain.loc = ItemGetter_loc(binnedspiketrain)
        binnedspiketrain.iloc = ItemGetter_iloc(binnedspiketrain)
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
            support = self.support.intersect(
                    epoch=idx,
                    boundaries=True
                    ) # what if fs of slicing epoch is different?
            if support.isempty:
                return BinnedSpikeTrainArray(empty=True)
            # next we need to determine the binnedSupport:

            raise NotImplementedError("EpochArray indexing for BinnedSpikeTrainArrays not supported yet")

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
                binnedspiketrain.loc = ItemGetter_loc(binnedspiketrain)
                binnedspiketrain.iloc = ItemGetter_iloc(binnedspiketrain)
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
                binnedspiketrain.loc = ItemGetter_loc(binnedspiketrain)
                binnedspiketrain.iloc = ItemGetter_iloc(binnedspiketrain)
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
                binnedspiketrain.loc = ItemGetter_loc(binnedspiketrain)
                binnedspiketrain.iloc = ItemGetter_iloc(binnedspiketrain)

                return binnedspiketrain

            except IndexError:
                raise TypeError(
                    'index out of range')
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
            return utils.PrettyInt(self.data.shape[0])
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
        return np.atleast_1d((self.binnedSupport[:,1] - self.binnedSupport[:,0] + 1).squeeze())

    @property
    def spiketrainarray(self):
        """(nelpy.SpikeTrain) The original spiketrain associated with
        the binned data.
        """
        return self._spiketrainarray

    @property
    def n_bins(self):
        """(int) The number of bins."""
        return utils.PrettyInt(len(self.centers))

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
        self._support = EpochArray(supportdata) # set support to TRUE bin support

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

        return utils.gaussian_filter(self, fs=fs, sigma=sigma, inplace=inplace)

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
        newbst = copy.copy(bst)
        if newdata is not None:
            newbst._data = newdata
            newbst._support = EpochArray(newsupport)
            newbst._bins = newbins
            newbst._bin_centers = newcenters
            newbst._ds = bst.ds*w
            newbst._binnedSupport = np.array((newedges[:-1], newedges[1:]-1)).T
        else:
            warnings.warn("No events are long enough to contain any bins of width {}".format(utils.PrettyDuration(ds)))
            newbst._data = None
            newbst._support = None
            newbst._binnedSupport = None
            newbst._bin_centers = None
            newbst._bins = None

        newbst.loc = ItemGetter_loc(newbst)
        newbst.iloc = ItemGetter_iloc(newbst)

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
        return utils.PrettyInt(np.count_nonzero(self.n_spikes))

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
        binnedspiketrainarray.loc = ItemGetter_loc(binnedspiketrainarray)
        binnedspiketrainarray.iloc = ItemGetter_iloc(binnedspiketrainarray)
        return binnedspiketrainarray

#----------------------------------------------------------------------#
#======================================================================#
