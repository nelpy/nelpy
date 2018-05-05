"""Ring-like AnalogSignal/PositionArray

subsample, simplify, interpolation, smoothing, asarray, downsample, ... all have
to be handled in a special way when we have a ring-like environment.
"""

import copy
import numpy as np
from scipy import interpolate
from collections import namedtuple

from ..core import _analogsignalarray, _epocharray
from .. import utils

class RinglikeTrajectory(_analogsignalarray.AnalogSignalArray):

    __attributes__ = ['_track_length', '_is_wrapped'] # RinglikeTrajectory-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)
    def __init__(self, ydata=[], *, timestamps=None, fs=None, step=None,
                 merge_sample_gap=0, support=None, in_memory=True, labels=None,
                 track_length=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        # cast an AnalogSignalArray to a RinglikeTrajectory:
        if isinstance(ydata, _analogsignalarray.AnalogSignalArray):
            assert ydata.n_signals == 1, \
                "only 1D AnalogSignalArrays can be cast to RinglikeTrajectories!"
            self.__dict__ = copy.deepcopy(ydata.__dict__)
            self._track_length = None
            self._is_wrapped = None
            self.__renew__()
        else:
            kwargs = {"ydata": ydata,
                    "timestamps": timestamps,
                    "fs": fs,
                    "step": step,
                    "merge_sample_gap": merge_sample_gap,
                    "support": support,
                    "in_memory": in_memory,
                    "labels": labels}

            # initialize super:
            super().__init__(**kwargs)

            self._track_length = track_length
            self._is_wrapped = None # intialize to unknown (None) state

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty RinglikeTrajectory" + address_str + ">"
        if self.n_epochs > 1:
            epstr = ": {} segments".format(self.n_epochs)
        else:
            epstr = ""
        dstr = " for a total of {}".format(utils.PrettyDuration(self.support.duration))
        if self.is_1d:
            return "<1D RinglikeTrajectory%s%s>%s" % (address_str, epstr, dstr)
        raise TypeError ("RinglikeTrajectories must be 1D at this time!")

    @property
    def is_1d(self):
        try:
            return self.n_signals == 1
        except IndexError:
            return False

    @property
    def is_wrapped(self):
        if self._is_wrapped is None:
            if self.max() > self._track_length:
                self._is_wrapped = False
            else:
                self._is_wrapped = True
        return self._is_wrapped

    @property
    def track_length(self):
        """Length of the ringlike environment."""
        if not self._track_length:
            raise ValueError("Please initialize/set track_length first!")
        return self._track_length

    @track_length.setter
    def track_length(self, val):
        """Set the length of the ringlike environment"""
        # TODO: do data integrity cheking / validation
        self._track_length = val

    def _unwrap(self, arr):
        """Unwrap trajectory to winding distance."""
        lin = copy.deepcopy(arr.squeeze())
        for ii in range(1, len(lin)):
            if lin[ii] - lin[ii-1] >= self.track_length/2:
                lin[ii:] = lin[ii:] - self.track_length
            elif lin[ii] - lin[ii-1] < - self.track_length/2:
                lin[ii:] = lin[ii:] + self.track_length
        return np.atleast_2d(lin)

    def _wrap(self, arr):
        """Wrap trajectory around ring."""
        return arr % self.track_length

    def wrap(self):
        """Wrap trajectory around ring."""
        self._ydata = np.atleast_2d(self._wrap(self._ydata.squeeze()))
        self._is_wrapped = True
        # self._interp = None

    def unwrap(self):
        """Unwrap trajectory to winding distance."""
        self._ydata = np.atleast_2d(self._unwrap(self._ydata.squeeze()))
        self._is_wrapped = False
        # self._interp = None

    def _wraptimes(self):
        """Return timestamps when trajectory wraps around."""
        is_wrapped = self.is_wrapped
        if not is_wrapped:
            self.wrap()
        lin = copy.deepcopy(self.ydata.squeeze())
        wraptimes = []
        for ii in range(1, len(lin)):
            if lin[ii] - lin[ii-1] >= self.track_length/2:
                lin[ii:] = lin[ii:] - self.track_length
                wraptimes.append(self.time[ii])
            elif lin[ii] - lin[ii-1] < - self.track_length/2:
                lin[ii:] = lin[ii:] + self.track_length
                wraptimes.append(self.time[ii])
        if not is_wrapped:
            self.unwrap()
        return np.asarray(wraptimes)

    def shift(self, amount, *, inplace=False):
        """"""
        is_wrapped = self.is_wrapped
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)
        out.unwrap()
        out = out + amount
        if is_wrapped:
            out.wrap()
        return out

    def smooth(self, *, fs=None, sigma=None, bw=None, inplace=False):
        """Smooths the regularly sampled RinglikeTrajectory with a Gaussian kernel.

        Smoothing is applied in time, and the same smoothing is applied to each
        signal in the RinglikeTrajectory.

        Smoothing is applied within each epoch.

        Parameters
        ----------
        fs : float, optional
            Sampling rate (in Hz) of RinglikeTrajectory. If not provided, it will
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
        out : RinglikeTrajectory
            A RinglikeTrajectory with smoothed data is returned.
        """

        is_wrapped = self.is_wrapped

        kwargs = {'inplace' : inplace,
                'fs' : fs,
                'sigma' : sigma,
                'bw' : bw}

        if is_wrapped:
            self.unwrap()
        out = utils.gaussian_filter(self, **kwargs)
        out.__renew__()
        if is_wrapped:
            out.wrap()
            if not inplace:
                self.wrap()
        return out

    def _get_interp1d(self,* , kind='linear', copy=True, bounds_error=False,
                      fill_value=np.nan, assume_sorted=None):
        """returns a scipy interp1d object, extended to have values at all epoch
        boundaries!
        """

        if assume_sorted is None:
            assume_sorted = utils.is_sorted(self.time)

        if self.n_signals > 1:
            axis = 1
        else:
            axis = -1

        time = self.time
        yvals = self._unwrap(self._ydata_rowsig)
        lengths = self.lengths
        empty_epoch_ids = np.argwhere(lengths==0).squeeze().tolist()
        first_timestamps_per_epoch_idx = np.insert(np.cumsum(lengths[:-1]),0,0)
        first_timestamps_per_epoch_idx[empty_epoch_ids] = 0
        last_timestamps_per_epoch_idx = np.cumsum(lengths)-1
        last_timestamps_per_epoch_idx[empty_epoch_ids] = 0
        first_timestamps_per_epoch = self.time[first_timestamps_per_epoch_idx]
        last_timestamps_per_epoch = self.time[last_timestamps_per_epoch_idx]

        boundary_times = []
        boundary_vals = []
        for ii, (start, stop) in enumerate(self.support.time):
            if lengths[ii] == 0:
                continue
            if first_timestamps_per_epoch[ii] > start:
                boundary_times.append(start)
                boundary_vals.append(yvals[:,first_timestamps_per_epoch_idx[ii]])
                # print('adding {} at time {}'.format(yvals[:,first_timestamps_per_epoch_idx[ii]], start))
            if last_timestamps_per_epoch[ii] < stop:
                boundary_times.append(stop)
                boundary_vals.append(yvals[:,last_timestamps_per_epoch_idx[ii]])

        if boundary_times:
            insert_locs = np.searchsorted(time, boundary_times)
            time = np.insert(time, insert_locs, boundary_times)
            yvals = np.insert(yvals, insert_locs, np.array(boundary_vals).T, axis=1)

            time, unique_idx = np.unique(time, return_index=True)
            yvals = yvals[:,unique_idx]

        f = interpolate.interp1d(x=time,
                                 y=yvals,
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
            self.time together with 'where' if applicable.
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
        try:
            if self.is_wrapped:
                out = self._wrap(interpobj(at))
            else:
                out = interpobj(at)
        except SystemError:
            interpobj = self._get_interp1d(**kwargs)
            if store_interp:
                self._interp = interpobj
            if self.is_wrapped:
                out = self._wrap(interpobj(at))
            else:
                out = interpobj(at)

        # TODO: set all values outside of self.support to fill_value

        xyarray = XYArray(xvals=np.asanyarray(at), yvals=np.asanyarray(out).squeeze())
        return xyarray
