"""Position class for 1D and 2D position AnalogSignalArrays"""

import copy
import numpy as np

from ..core import _analogsignalarray, _epocharray
from .. import utils
from ..utils_.decorators import keyword_deprecation


class PositionArray(_analogsignalarray.AnalogSignalArray):

    __attributes__ = ["_kalmanfilter"]  # PositionArray-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)

    def __init__(
        self,
        data=[],
        *,
        timestamps=None,
        fs=None,
        step=None,
        merge_sample_gap=0,
        support=None,
        in_memory=True,
        labels=None,
        empty=False
    ):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        # cast an AnalogSignalArray to a PositionArray:
        if isinstance(data, _analogsignalarray.AnalogSignalArray):
            self.__dict__ = copy.deepcopy(data.__dict__)
            self.__renew__()
        else:
            kwargs = {
                "data": data,
                "timestamps": timestamps,
                "fs": fs,
                "step": step,
                "merge_sample_gap": merge_sample_gap,
                "support": support,
                "in_memory": in_memory,
                "labels": labels,
            }

            # initialize super:
            super().__init__(**kwargs)

        self._kalmanfilter = None

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty PositionArray" + address_str + ">"
        if self.n_epochs > 1:
            epstr = ": {} segments".format(self.n_epochs)
        else:
            epstr = ""
        dstr = " for a total of {}".format(utils.PrettyDuration(self.support.duration))
        if self.is_2d:
            return "<2D PositionArray%s%s>%s" % (address_str, epstr, dstr)
        return "<1D PositionArray%s%s>%s" % (address_str, epstr, dstr)

    @property
    def is_2d(self):
        try:
            return self.n_signals == 2
        except IndexError:
            return False

    @property
    def is_1d(self):
        try:
            return self.n_signals == 1
        except IndexError:
            return False

    @property
    def x(self):
        """return x-values, as numpy array."""
        return self.data[0, :]

    @property
    def y(self):
        """return y-values, as numpy array."""
        if self.is_2d:
            return self.data[1, :]
        raise ValueError(
            "PositionArray is not 2 dimensional, so y-values are undefined!"
        )

    @property
    def path_length(self):
        """Return the path length along the trajectory."""
        # raise NotImplementedError
        lengths = np.sqrt(np.sum(np.diff(self._data_colsig, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

    def get_speed(self, sigma_pos=None, simga_spd=None):
        """Return the speed, as an AnalogSignalArray."""
        # Idea is to smooth the position with a good default, and then to
        # compute the speed on the smoothed position. Optionally, then, the
        # speed should be smoothed as well.

        raise NotImplementedError
    
        out = self.copy()
        cum_lengths = np.insert(np.cumsum(self.lengths), 0, 0)

        # ensure that datatype is float
        out._data = out.data.astype(float)

        # now obtain the derivative for each epoch separately
        for idx in range(self.n_epochs):
            # if 1D:
            if self.n_signals == 1:
                if (cum_lengths[idx + 1] - cum_lengths[idx]) < 2:
                    # only single sample
                    out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = 0
                else:
                    out._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]] = (
                        np.gradient(
                            self._data[[0], cum_lengths[idx] : cum_lengths[idx + 1]],
                            axis=1,
                        )
                    )
            else:
                if (cum_lengths[idx + 1] - cum_lengths[idx]) < 2:
                    # only single sample
                    out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = 0
                else:
                    out._data[:, cum_lengths[idx] : cum_lengths[idx + 1]] = np.gradient(
                        self._data[:, cum_lengths[idx] : cum_lengths[idx + 1]], axis=1
                    )

    def direction(self):
        """Return the instantaneous direction estimate as an AnalogSignalArray."""
        # If 1D, then left/right or up/down or fwd/reverse or whatever might
        # make sense, so this signal could be binarized.
        # When 2D, then a continuous cartesian direction vector might make sense
        # (that is, (x,y)-components), or a polar coordinate system might be
        # better suited, with the direction signal being a continuous theta angle

        raise NotImplementedError

    def idealize(self, segments):
        """Project the position onto idealized segments."""
        # plan is to return a PositionArray constrained to the desired segments.

        raise NotImplementedError

    def linearize(self):
        """Linearize the position estimates."""

        # This is unclear how we should do it universally? Maybe have a few presets?

        raise NotImplementedError

    def bin(self, **kwargs):
        """Bin position into grid."""
        raise NotImplementedError

    @keyword_deprecation(replace_x_with_y={"bw": "truncate"})
    def smooth(
        self, *, fs=None, sigma=None, truncate=None, inplace=False, Kalman=False
    ):
        """Smooths the regularly sampled PositionArray with a Gaussian kernel.

        Smoothing is applied in time, and the same smoothing is applied to each
        signal in the PositionArray.

        Smoothing is applied within each epoch.

        Optionally, a Kalman smoother can be used by setting Kalman=True.

        Parameters
        ----------
        fs : float, optional
            Sampling rate (in Hz) of RinglikeTrajectory. If not provided, it will
            be obtained from asa.fs
        sigma : float, optional
            Standard deviation of Gaussian kernel, in seconds. Default is 0.05 (50 ms)
        truncate : float, optional
            Bandwidth outside of which the filter value will be zero. Default is 4.0
        inplace : bool
            If True the data will be replaced with the smoothed data.
            Default is False.
        Kalman : bool, optional
            If True, smooth with a Kalman smoother instead of a Gaussian kernel.
            Default is False.

        Returns
        -------
        out : PositionArray
            A PositionArray with smoothed data is returned.
        """

        if Kalman:
            position, speed = self._kalman_smoother()
            if inplace:
                out = self
            else:
                out = copy.deepcopy(self)
            out._data = position
            out.__renew__()
            return out

        kwargs = {"inplace": inplace, "fs": fs, "sigma": sigma, "truncate": truncate}

        out = utils.gaussian_filter(self, **kwargs)
        out.__renew__()
        return out

    def _kalman_smoother(self, Q=None, R=None, recompute=False, n_iter=None):
        """
        Use a Kalman smoother to estimate the trajectory of a particle moving
        with a constant velocity.

        Parameters
        ----------
        Q : float, optional
            Transition noise scalar: noise(Q) ~ N(0,Q). Default is 1.
        R : float, optional
            Observation noise scalar: noise(R) ~ N(0,R). Default is 10*self.fs.
            Larger values of R put higher trust in the model than in the [noisy]
            observations, leading to smoother estimates.
        recompute : bool, optional
            If True, recompute the filter parameters using EM. Default is False.
        n_iter : int, optional
            Number of iterations to use in EM when finding filter parameters.
            Default is 5.

        Returns
        ----------
        position : array with shape (ndim, n_samples), where ndim is 1D or 2D.
            Array of smoothed position estimates.
        speed : array with shape (ndim, n_samples), where ndim is 1D or 2D.
            Array of smoothed speed estimates.

        General information
        ===================

        see https://statweb.stanford.edu/~candes/acm116/Handouts/Kalman.pdf

        X(t) = [x1(t)
                x2(t)
                dx1(t)
                dx2(t)]

        F = [[1 0 1 0]
             [0 1 0 1]
             [0 0 1 0]
             [0 0 0 1]]

        H = [[1 0 0 0]
             [0 1 0 0]]

        X(t+1) = F X(t) + noise(Q)  {noise(Q) ~ N(0,Q)}
        Y(t) = H X(t) + noise(R)    {noise(R) ~ N(0,R)}

        ss = 4; # state size
        os = 2; # observation size
        F = [1 0 1 0; 0 1 0 1; 0 0 1 0; 0 0 0 1]
        H = [1 0 0 0; 0 1 0 0]
        Q = 1*eye(ss)  # transition_covariance
        R = 10*eye(os) # observation_covariance
        initx = [10 10 1 0].T
        initV = 10*eye(ss)

        NOTE: filtering vs smoothing...
            filter: compute (X_t | Y_0=y_0,...,Y_t=y_t); real-time
            smooth: compute (X_t | Y_0=y_0,...,Y_T=y_T), t < T; post-processing,
                given all data.

        NOTE: multi-epoch smoothing
        ===========================
            Currently each epoch is smoothed INDEPENDENTLY (and if parameters are
            recomputed, they are recomputed within each epoch). The alternative
            would have been to use masked arrays, but there are downsides to that
            approach, as well.

        NOTE: uniform sampling
        ======================
            The Kalman smoother assumes uniform sampling _within each epoch_. This
            is not checked explicitly.

        """

        # TODO: implement masking within epochs

        # >>> from numpy import ma
        # >>> X = ma.array([1,2,3])
        # >>> X[1] = ma.masked  # hide measurement at time step 1
        # >>> kf.em(X).smooth(X)

        from pykalman import KalmanFilter

        # assert self.n_epochs == 1, 'multi-epoch Kalman smoothing not supported yet!'

        if not n_iter:
            n_iter = 5

        if self.is_1d:
            ss = 2  # state size (x, dx)
            os = 1  # observation size, (x)
            # transition matrix:
            F = [[1, 1], [0, 1]]
            # observation matrix:
            H = [[1, 0]]
        elif self.is_2d:
            ss = 4  # state size (x, y, dx, dy)
            os = 2  # observation size, (x, y)
            # transition matrix:
            F = [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]]
            # observation matrix:
            H = [[1, 0, 0, 0], [0, 1, 0, 0]]
        else:
            raise ValueError("Only 1D or 2D PositionArrays supported!")

        if Q is None:
            Q = 1
        if R is None:
            R = 10 * self.fs  # TODO: maybe a better default is self.fs * 10 ???

        if self.is_1d:
            posdata = np.zeros((1, 0))
            speeddata = np.zeros((1, 0))
        elif self.is_2d:
            posdata = np.zeros((2, 0))
            speeddata = np.zeros((2, 0))

        for snippet in self:
            measurements = snippet.data.T

            if recompute:
                kf = KalmanFilter(transition_matrices=F, observation_matrices=H)
                kf = kf.em(measurements, n_iter=n_iter)
                snippet._kalmanfilter = kf
            else:
                if self._kalmanfilter is None:
                    kf = KalmanFilter(transition_matrices=F, observation_matrices=H)
                    self._kalmanfilter = kf
                self._kalmanfilter.transition_covariance = Q * np.eye(ss)
                self._kalmanfilter.observation_covariance = R * np.eye(os)

            if self.is_1d:
                self._kalmanfilter.initial_state_mean = np.append(snippet.data[0, 0], 0)
            elif self.is_2d:
                self._kalmanfilter.initial_state_mean = np.append(
                    snippet.data[:, 0], (0, 0)
                )

            kf = self._kalmanfilter

            smoothed_state_means, smoothed_state_covariances = kf.smooth(measurements)

            position = smoothed_state_means[:, :os].T
            speed = smoothed_state_means[:, os:].T

            posdata = np.hstack((posdata, position))
            speeddata = np.hstack((speeddata, speed))

        speeddata = self.fs * np.sqrt(np.sum(speeddata**2, axis=0))

        return posdata, speeddata
