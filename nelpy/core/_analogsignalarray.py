__all__ = ['AnalogSignalArray']

import warnings
import numpy as np
import copy
import numbers

from scipy import interpolate
from sys import float_info
from collections import namedtuple

from ..utils import is_sorted, \
                   get_contiguous_segments, \
                   PrettyDuration, \
                   PrettyInt, \
                   gaussian_filter

from ._epocharray import EpochArray

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

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
        As such, fs_acquisition as opposed to fs should be used to calculate
        time and should be changed from the default None. See notebook of
        AnalogSignalArray uses. Lastly, it is worth noting that fs_acquisition
        is set equal to fs if it is not set.
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
    calc_time : bool, optional
        Boolean to determine whether or not time will be calculated by scaling
        tdata by fs or fs_acquisition depending on which is passed in.
    labels : np.array(dtype=np.str,dimension=N)
        Labeling each one of the signals in AnalogSignalArray. By default this
        will be set to None. It is expected that all signals will be labeled if
        labels are passed in. If any signals are not labeled we will label them
        as Nones and if more labels are passed in than the number of signals
        given, the extras will be truncated. If we're nice (which we are for
        the most part), we will display a warning upon doing any of these
        things! :P Lastly, it is worth noting that most logical and type error
        checking for this is expected to be done by the user. Inputs are casted
        to string snad stored in a numpy array.
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
        See Parameters. fs_acquisition will be set to fs if fs_acquisition is
        None and fs is specified.
    fs_meta : float, scalar, optional
        See Paramters
    step : int
        See Parameters
    support : EpochArray, optional
        See Parameters
    labels : np.array
        See Parameters
    interp : array of interpolation objects from scipy.interpolate

        See Parameters
    """
    __attributes__ = ['_ydata', '_tdata', '_time', '_fs', '_support', \
                      '_interp', '_fs_meta', '_step', '_fs_acquisition',\
                      '_labels']
    def __init__(self, ydata, *, tdata=None, fs=None, fs_acquisition=None, fs_meta = None,
                 step=None, merge_sample_gap=0, support=None, calc_time = True,
                 in_memory=True, labels=None, empty=False):

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
        if(fs is not None):
            try:
                if(fs > 0):
                    self._fs = fs
                else:
                    raise ValueError("fs must be positive")
            except TypeError:
                raise TypeError("fs expected to be a scalar")

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
        else:
            self._fs_acquisition = self.fs

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
                # self.__init__([],empty=True)
                raise TypeError("tdata and ydata size mismatch! Note: ydata "
                                "is expected to have rows containing signals")

        self._ydata = ydata
        # Note: time will be None if this is not a time series and fs isn't
        # specified set xtime to None.
        self._time = None

        #handle labels
        if labels is not None:
            labels = np.asarray(labels,dtype=np.str)
            #label size doesn't match
            if labels.shape[0] > ydata.shape[0]:
                warnings.warn("More labels than ydata! labels are sliced to "
                              "size of ydata")
                labels = labels[0:ydata.shape[0]]
            elif labels.shape[0] < ydata.shape[0]:
                warnings.warn("Less labels than tdata! labels are filled with "
                              "None to match ydata shape")
                for i in range(labels.shape[0],ydata.shape[0]):
                    labels.append(None)
        self._labels = labels

        # Alright, let's handle all the possible parameter cases!
        if tdata is not None:
            if fs is not None:
                if(self._fs_acquisition is not None):
                    if(calc_time):
                        time = tdata / self._fs_acquisition
                        self._tdata = tdata
                    else:
                        time = tdata
                        self._tdata = time*self._fs_acquisition
                else:
                    self._fs_acquisition = self._fs
                    if calc_time:
                        time = tdata / self._fs
                        self._tdata = tdata
                    else:
                        time = tdata
                        self._tdata = time*self._fs_acquisition
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
                    warnings.warn("creating support with given tdata and "
                                  "sampling rate, fs (or fs_acq)!")
                    self._time = time
                    self._support = EpochArray(
                        get_contiguous_segments(
                            self.tdata,
                            step=self._step,
                            in_memory=in_memory),
                        fs=self._fs_acquisition)
                    if merge_sample_gap > 0:
                        self._support = self._support.merge(gap=merge_sample_gap)
            else:
                if(self._fs_acquisition is not None and calc_time):
                    warnings.warn("Why would you enter fs_acq but not fs? This feature may be removed"
                                    " but tdata has been scaled by fs_acq for now.")
                    time = tdata / self._fs_acquisition
                else:
                    if calc_time:
                        warnings.warn("No fs passed. time being set equal to "
                                      "tdata.")
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
                    self._support = EpochArray(
                        get_contiguous_segments(
                            self.tdata,
                            step=self._step,
                            in_memory=in_memory),
                        fs=self._fs_acquisition)
                    #merge gaps in Epochs if requested
                    if merge_sample_gap > 0:
                        self._support = self._support.merge(gap=merge_sample_gap)
        else:
            tdata = np.arange(0, ydata.shape[1], 1)
            if fs is not None:
                if(self._fs_acquisition is not None):
                    time = tdata / self._fs_acquisition
                else:
                    self._fs_acquisition = self._fs
                    time = tdata / self._fs
                # fs and support
                if support is not None:
                    self.__init__([],empty=True)
                    raise TypeError("tdata must be passed if support is specified")
                # just fs
                else:
                    self._time = time
                    warnings.warn("support created with given sampling rate, fs")
                    self._support = EpochArray(np.array([0, self._time[-1]]))
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
                    self._support = EpochArray(np.array([0, self._time[-1]]))
            self._tdata = tdata

        # finally, if still no fs has been set, estimate it:
        if self.fs is None:
            self._fs = self._estimate_fs()
        else:
            if np.abs(self.fs - self._estimate_fs()/self.fs) > 0.01:
                warnings.warn("estimated fs and provided fs differ by more than 1%")

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

    def _estimate_fs(self, data=None):
        """Estimate the sampling rate of the data."""
        if data is None:
            data = self.time
        return 1.0/np.median(np.diff(data))

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
        if label == None:
            warnings.warn("None label appended")
        np.append(self._labels,label)
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
                exclude = ['_support','_ydata','_fs','_fs_meta', '_step', \
                           '_fs_acquisition']
                attrs = (x for x in self.__attributes__ if x not in exclude)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    for attr in attrs:
                        exec("self." + attr + " = None")
                self._ydata = np.zeros([0,self._ydata.shape[0]])
                self._ydata[:] = np.NAN
                self._support = epocharray
                return
        except AttributeError:
            raise AttributeError("EpochArray expected")

        indices = []
        for eptime in epocharray.time:
            t_start = eptime[0]
            t_stop = eptime[1]
            indices.append((self._time >= t_start) & (self._time <= t_stop))
        indices = np.any(np.column_stack(indices), axis=1)
        if np.count_nonzero(indices) < len(self._time):
            warnings.warn(
                'ignoring signal outside of support')
        try:
            self._ydata = self._ydata[:,indices]
        except IndexError:
            self._ydata = np.zeros([0,self._ydata.shape[0]])
            self._ydata[:] = np.NAN
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
        """(nelpy.EpochArray) The support of the underlying AnalogSignalArray
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
        if self._fs is None:
            warnings.warn("No sampling frequency has been specified!")
        return self._fs

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

#----------------------------------------------------------------------#
#======================================================================#