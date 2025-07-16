import numpy as np
import logging
import copy

from scipy.special import logsumexp

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from .preprocessing import DataWindow
from . import core
from .plotting import _plot_ratemap

from .utils_.decorators import keyword_deprecation

"""
FiringRateEstimator(BaseEstimator) DRAFT SPECIFICATION
    X : BST / spike counts (or actual spikes?)
    y : firing rate (not used)
    z : position (response variable)
    mode = ['hist', 'glm-poisson', 'glm-binomial', 'glm', 'gvm', 'bars', 'gp']
    fit(X, y, z) : estimate model parameters, or collect histogram evidence
    y = predict(X, z) : this predicts the firing rate estimate for data
    score (X, y, z)

RateMap(BaseEstimator):
    X : position (state: discrete)
    y : firing rate(s)
    mode = ['continuous', 'discrete', 'circular']
    fit(X, y) : assign rates to X bins
    y = predict(X) : predicts, and then smooths, firing rates
    bst = synthesize(X) : predicts and smooths firing rate, and then synthesize spikes
    _bins
    _ratemap
    _mode

BayesianDecoder(BaseEstimator):
    X : BST / spike counts
    y : position
    fit(X, y) : fits the RateMap, and occupancy (and other params, like movement)
    y = predict(X) : predicts position from spike counts (also called decode)

"""


class KeywordError(Exception):
    """
    Exception raised for errors in keyword arguments.

    Parameters
    ----------
    message : str
        Explanation of the error.
    """
    def __init__(self, message):
        """
        Initialize the KeywordError.

        Parameters
        ----------
        message : str
            Explanation of the error.
        """
        self.message = message


class UnitSlicer(object):
    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, *args):
        """units ids"""
        # by default, keep all units
        unitslice = slice(None, None, None)
        if isinstance(*args, int):
            unitslice = args[0]
        else:
            slices = np.s_[args]
            slices = slices[0]
            unitslice = slices

        if isinstance(unitslice, slice):
            start = unitslice.start
            stop = unitslice.stop
            istep = unitslice.step

            try:
                if start is None:
                    istart = 0
                else:
                    istart = list(self.obj.unit_ids).index(start)
            except ValueError:
                raise KeyError(
                    "unit_id {} could not be found in RateMap!".format(start)
                )

            try:
                if stop is None:
                    istop = self.obj.n_units
                else:
                    istop = list(self.obj.unit_ids).index(stop) + 1
            except ValueError:
                raise KeyError("unit_id {} could not be found in RateMap!".format(stop))
            if istep is None:
                istep = 1
            if istep < 0:
                istop -= 1
                istart -= 1
                istart, istop = istop, istart
            unit_idx_list = list(range(istart, istop, istep))
        else:
            unit_idx_list = []
            unitslice = np.atleast_1d(unitslice)
            for unit in unitslice:
                try:
                    uidx = list(self.obj.unit_ids).index(unit)
                except ValueError:
                    raise KeyError(
                        "unit_id {} could not be found in RateMap!".format(unit)
                    )
                else:
                    unit_idx_list.append(uidx)

        return unit_idx_list


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
        """unit_ids"""
        unit_idx_list = self.obj._slicer[idx]

        return self.obj[unit_idx_list]


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
        unit_idx_list = idx
        if isinstance(idx, int):
            unit_idx_list = [idx]

        return self.obj[unit_idx_list]


class RateMap(BaseEstimator):
    """
    RateMap with persistent unit_ids and firing rates in Hz.

    This class estimates and stores firing rate maps for neural data, supporting both 1D and 2D spatial representations.

    Parameters
    ----------
    connectivity : {'continuous', 'discrete', 'circular'}, optional
        Defines how smoothing is applied. Default is 'continuous'.
        - 'continuous': Continuous smoothing.
        - 'discrete': No smoothing is applied.
        - 'circular': Circular smoothing (for angular variables).

    Attributes
    ----------
    connectivity : str
        Smoothing mode.
    ratemap_ : np.ndarray
        The estimated firing rate map.
    _unit_ids : np.ndarray
        Persistent unit IDs.
    _bins_x, _bins_y : np.ndarray
        Bin edges for each dimension.
    _bin_centers_x, _bin_centers_y : np.ndarray
        Bin centers for each dimension.
    _mask : np.ndarray
        Mask for valid regions.
    """
    def __init__(self, connectivity="continuous"):
        """
        Initialize a RateMap object.

        Parameters
        ----------
        connectivity : str, optional
            Defines how smoothing is applied. If 'discrete', then no smoothing is
            applied. Default is 'continuous'.
        """
        self.connectivity = connectivity
        self._slicer = UnitSlicer(self)
        self.loc = ItemGetter_loc(self)
        self.iloc = ItemGetter_iloc(self)
    def __repr__(self):
        """
        Return a string representation of the RateMap, including shape if fitted.

        Returns
        -------
        r : str
            String representation of the RateMap.
        """
        r = super().__repr__()
        if self._is_fitted():
            if self.is_1d:
                r += " with shape (n_units={}, n_bins_x={})".format(*self.shape)
            else:
                r += " with shape (n_units={}, n_bins_x={}, n_bins_y={})".format(
                    *self.shape
                )
        return r
    def fit(self, X, y, dt=1, unit_ids=None):
        """
        Fit firing rates to the provided data.

        Parameters
        ----------
        X : array-like, shape (n_bins,) or (n_bins_x, n_bins_y)
            Bin locations (centers) where ratemap is defined.
        y : array-like, shape (n_units, n_bins) or (n_units, n_bins_x, n_bins_y)
            Expected number of spikes in a temporal bin of width dt, for each of
            the predictor bins specified in X.
        dt : float, optional
            Temporal bin size with which firing rate y is defined. Default is 1.
        unit_ids : array-like, shape (n_units,), optional
            Persistent unit IDs that are used to associate units after
            permutation. If None, uses np.arange(n_units).

        Returns
        -------
        self : RateMap
            The fitted RateMap instance.
        """
        n_units, n_bins_x, n_bins_y = self._check_X_y(X, y)
        if n_bins_y > 0:
            # self.ratemap_ = np.zeros((n_units, n_bins_x, n_bins_y)) #FIXME
            self.ratemap_ = y / dt
            bin_centers_x = np.squeeze(X[:, 0])
            bin_centers_y = np.squeeze(X[:, 1])
            bin_dx = np.median(np.diff(bin_centers_x))
            bin_dy = np.median(np.diff(bin_centers_y))
            bins_x = np.insert(
                bin_centers_x[:-1] + np.diff(bin_centers_x) / 2,
                0,
                bin_centers_x[0] - bin_dx / 2,
            )
            bins_x = np.append(bins_x, bins_x[-1] + bin_dx)
            bins_y = np.insert(
                bin_centers_y[:-1] + np.diff(bin_centers_y) / 2,
                0,
                bin_centers_y[0] - bin_dy / 2,
            )
            bins_y = np.append(bins_y, bins_y[-1] + bin_dy)
            self._bins_x = bins_x
            self._bins_y = bins_y
            self._bin_centers_x = bin_centers_x
            self._bin_centers_y = X[:, 1]
        else:
            # self.ratemap_ = np.zeros((n_units, n_bins_x)) #FIXME
            self.ratemap_ = y / dt
            bin_centers_x = np.squeeze(X)
            bin_dx = np.median(np.diff(bin_centers_x))
            bins_x = np.insert(
                bin_centers_x[:-1] + np.diff(bin_centers_x) / 2,
                0,
                bin_centers_x[0] - bin_dx / 2,
            )
            bins_x = np.append(bins_x, bins_x[-1] + bin_dx)
            self._bins_x = bins_x
            self._bin_centers_x = bin_centers_x

        if unit_ids is not None:
            if len(unit_ids) != n_units:
                raise ValueError(
                    "'unit_ids' must have same number of elements as 'n_units'. {} != {}".format(
                        len(unit_ids), n_units
                    )
                )
            self._unit_ids = unit_ids
        else:
            self._unit_ids = np.arange(n_units)

    def predict(self, X):
        """
        Predict firing rates for the given bin locations.

        Parameters
        ----------
        X : array-like
            Bin locations to predict firing rates for.

        Returns
        -------
        rates : array-like
            Predicted firing rates.
        """
        check_is_fitted(self, "ratemap_")
        raise NotImplementedError

    def synthesize(self, X):
        """
        Generate synthetic spike data based on the ratemap.

        Parameters
        ----------
        X : array-like
            Bin locations to synthesize spikes for.

        Returns
        -------
        spikes : array-like
            Synthetic spike data.
        """
        check_is_fitted(self, "ratemap_")
        raise NotImplementedError

    def __len__(self):
        return self.n_units

    def __iter__(self):
        """TuningCurve1D iterator initialization"""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """TuningCurve1D iterator advancer."""
        index = self._index
        if index > self.n_units - 1:
            raise StopIteration
        out = copy.copy(self)
        out.ratemap_ = self.ratemap_[tuple([index])]
        out._unit_ids = self._unit_ids[index]
        self._index += 1
        return out

    def __getitem__(self, *idx):
        """
        Access RateMap units by index.

        Parameters
        ----------
        *idx : int, slice, or list
            Indices of units to access.

        Returns
        -------
        out : RateMap
            Subset RateMap with selected units.
        """
        idx = [ii for ii in idx]
        if len(idx) == 1 and not isinstance(idx[0], int):
            idx = idx[0]
        if isinstance(idx, tuple):
            idx = [ii for ii in idx]

        try:
            out = copy.copy(self)
            out.ratemap_ = self.ratemap_[tuple([idx])]
            out._unit_ids = list(np.array(out._unit_ids)[tuple([idx])])
            out._slicer = UnitSlicer(out)
            out.loc = ItemGetter_loc(out)
            out.iloc = ItemGetter_iloc(out)
            return out
        except Exception:
            raise TypeError("unsupported subsctipting type {}".format(type(idx)))

    def get_peak_firing_order_ids(self):
        """Get the unit_ids in order of peak firing location for 1D RateMaps.

        Returns
        -------
        unit_ids : array-like
            The permutaiton of unit_ids such that after reordering, the peak
            firing locations are ordered along the RateMap.
        """
        check_is_fitted(self, "ratemap_")
        if self.is_2d:
            raise NotImplementedError(
                "get_peak_firing_order_ids() only implemented for 1D RateMaps."
            )
        peakorder = np.argmax(self.ratemap_, axis=1).argsort()
        return np.array(self.unit_ids)[peakorder]

    def reorder_units_by_ids(self, unit_ids, inplace=False):
        """Permute the unit ordering.

        #TODO
        If no order is specified, and an ordering exists from fit(), then the
        data in X will automatically be permuted to match that registered during
        fit().

        Parameters
        ----------
        unit_ids : array-like, shape (n_units,)

        Returns
        -------
        out : reordered RateMap
        """

        def swap_units(arr, frm, to):
            """swap 'units' of a 3D np.array"""
            arr[(frm, to), :] = arr[(to, frm), :]

        self._validate_unit_ids(unit_ids)
        if len(unit_ids) != len(self._unit_ids):
            raise ValueError(
                "unit_ids must be a permutation of self.unit_ids, not a subset thereof."
            )

        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        neworder = [list(self.unit_ids).index(x) for x in unit_ids]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            swap_units(out.ratemap_, frm, to)
            out._unit_ids[frm], out._unit_ids[to] = (
                out._unit_ids[to],
                out._unit_ids[frm],
            )
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
        return out

    def _check_X_y(self, X, y):
        X = np.atleast_1d(X)
        y = np.atleast_2d(y)

        n_units = y.shape[0]
        n_bins_xy = y.shape[1]
        try:
            n_bins_yy = y.shape[2]
        except IndexError:
            n_bins_yy = 0

        n_bins_xx = X.shape[0]
        try:
            n_bins_yx = X.shape[1]
        except IndexError:
            n_bins_yx = 0

        assert n_units > 0, "n_units must be a positive integer!"
        assert n_bins_xx == n_bins_xy, "X and y must have the same n_bins_x"
        assert n_bins_yx == n_bins_yy, "X and y must have the same n_bins_y"

        n_bins_x = n_bins_xx
        n_bins_y = n_bins_yy

        return n_units, n_bins_x, n_bins_y

    def _validate_unit_ids(self, unit_ids):
        self._check_unit_ids_in_ratemap(unit_ids)

        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Duplicate unit_ids are not allowed.")

    def _check_unit_ids_in_ratemap(self, unit_ids):
        for unit_id in unit_ids:
            # NOTE: the check below allows for predict() to pass on only
            # a subset of the units that were used during fit! So we
            # could fit on 100 units, and then predict on only 10 of
            # them, if we wanted.
            if unit_id not in self.unit_ids:
                raise ValueError(
                    "unit_id {} was not present during fit(); aborting...".format(
                        unit_id
                    )
                )

    def _is_fitted(self):
        try:
            check_is_fitted(self, "ratemap_")
        except Exception:  # should really be except NotFitterError
            return False
        return True

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, val):
        self._connectivity = self._validate_connectivity(val)

    @staticmethod
    def _validate_connectivity(connectivity):
        connectivity = str(connectivity).strip().lower()
        options = ["continuous", "discrete", "circular"]
        if connectivity in options:
            return connectivity
        raise NotImplementedError(
            "connectivity '{}' is not supported yet!".format(str(connectivity))
        )

    @staticmethod
    def _units_from_X(X):
        """
        Get unit_ids from bst X, or generate them from ndarray X.

        Returns
        -------
        n_units :
        unit_ids :
        """
        raise NotImplementedError

    @property
    def T(self):
        """transpose the ratemap.
        Here we transpose the x and y dims, and return a new RateMap object.
        """
        if self.is_1d:
            return self
        out = copy.copy(self)
        out.ratemap_ = np.transpose(out.ratemap_, axes=(0, 2, 1))
        return out

    @property
    def shape(self):
        """
        RateMap.shape = (n_units, n_features_x, n_features_y)
            OR
        RateMap.shape = (n_units, n_features)
        """
        check_is_fitted(self, "ratemap_")
        return self.ratemap_.shape

    @property
    def is_1d(self):
        check_is_fitted(self, "ratemap_")
        if len(self.ratemap_.shape) == 2:
            return True
        return False

    @property
    def is_2d(self):
        check_is_fitted(self, "ratemap_")
        if len(self.ratemap_.shape) == 3:
            return True
        return False

    @property
    def n_units(self):
        check_is_fitted(self, "ratemap_")
        return self.ratemap_.shape[0]

    @property
    def unit_ids(self):
        check_is_fitted(self, "ratemap_")
        return self._unit_ids

    @property
    def n_bins(self):
        """(int) Number of external correlates (bins)."""
        check_is_fitted(self, "ratemap_")
        if self.is_2d:
            return self.n_bins_x * self.n_bins_y
        return self.n_bins_x

    @property
    def n_bins_x(self):
        """(int) Number of external correlates (bins)."""
        check_is_fitted(self, "ratemap_")
        return self.ratemap_.shape[1]

    @property
    def n_bins_y(self):
        """(int) Number of external correlates (bins)."""
        check_is_fitted(self, "ratemap_")
        if self.is_1d:
            raise ValueError("RateMap is 1D; no y bins are defined.")
        return self.ratemap_.shape[2]

    def max(self, axis=None, out=None):
        """
        maximum firing rate for each unit:
            RateMap.max()
        maximum firing rate across units:
            RateMap.max(axis=0)
        """
        check_is_fitted(self, "ratemap_")
        if axis is None:
            if self.is_2d:
                return self.ratemap_.max(axis=1, out=out).max(axis=1, out=out)
            else:
                return self.ratemap_.max(axis=1, out=out)
        return self.ratemap_.max(axis=axis, out=out)

    def min(self, axis=None, out=None):
        check_is_fitted(self, "ratemap_")
        if axis is None:
            if self.is_2d:
                return self.ratemap_.min(axis=1, out=out).min(axis=1, out=out)
            else:
                return self.ratemap_.min(axis=1, out=out)
        return self.ratemap_.min(axis=axis, out=out)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        check_is_fitted(self, "ratemap_")
        kwargs = {"dtype": dtype, "out": out, "keepdims": keepdims}
        if axis is None:
            if self.is_2d:
                return self.ratemap_.mean(axis=1, **kwargs).mean(axis=1, **kwargs)
            else:
                return self.ratemap_.mean(axis=1, **kwargs)
        return self.ratemap_.mean(axis=axis, **kwargs)

    @property
    def bins(self):
        if self.is_1d:
            return self._bins_x
        return np.vstack((self._bins_x, self._bins_y))

    @property
    def bins_x(self):
        return self._bins_x

    @property
    def bins_y(self):
        if self.is_2d:
            return self._bins_y
        else:
            raise ValueError("only valid for 2D RateMap() objects.")

    @property
    def bin_centers(self):
        if self.is_1d:
            return self._bin_centers_x
        return np.vstack((self._bin_centers_x, self._bin_centers_y))

    @property
    def bin_centers_x(self):
        return self._bin_centers_x

    @property
    def bin_centers_y(self):
        if self.is_2d:
            return self._bin_centers_y
        else:
            raise ValueError("only valid for 2D RateMap() objects.")

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        # TODO: mask validation
        raise NotImplementedError
        self._mask = val

    def plot(self, **kwargs):
        check_is_fitted(self, "ratemap_")
        if self.is_2d:
            raise NotImplementedError("plot() not yet implemented for 2D RateMaps.")
        pad = kwargs.pop("pad", None)
        _plot_ratemap(self, pad=pad, **kwargs)

    @keyword_deprecation(replace_x_with_y={"bw": "truncate"})
    def smooth(self, *, sigma=None, truncate=None, inplace=False, mode=None, cval=None):
        """Smooths the tuning curve with a Gaussian kernel.

        mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
            The mode parameter determines how the array borders are handled,
            where cval is the value when mode is equal to ‘constant’. Default is
            ‘reflect’
        truncate : float
            Truncate the filter at this many standard deviations. Default is 4.0.
        truncate : float, deprecated
            Truncate the filter at this many standard deviations. Default is 4.0.
        cval : scalar, optional
            Value to fill past edges of input if mode is ‘constant’. Default is 0.0
        """

        if sigma is None:
            sigma = 0.1  # in units of extern
        if truncate is None:
            truncate = 4
        if mode is None:
            mode = "reflect"
        if cval is None:
            cval = 0.0

        raise NotImplementedError


class BayesianDecoderTemp(BaseEstimator):
    """
    Bayesian decoder wrapper class.

    This class implements a Bayesian decoder for neural data, supporting various estimation modes.

    Parameters
    ----------
    rate_estimator : FiringRateEstimator, optional
        The firing rate estimator to use.
    w : any, optional
        Window parameter for decoding.
    ratemap : RateMap, optional
        Precomputed rate map.

    Attributes
    ----------
    rate_estimator : FiringRateEstimator
        The firing rate estimator.
    ratemap : RateMap
        The estimated or provided rate map.
    w : any
        Window parameter.
    """

    def __init__(self, rate_estimator=None, w=None, ratemap=None):
        self._rate_estimator = self._validate_rate_estimator(rate_estimator)
        self._ratemap = self._validate_ratemap(ratemap)
        self._w = self._validate_window(w)

    @property
    def rate_estimator(self):
        return self._rate_estimator

    @property
    def ratemap(self):
        return self._ratemap

    @property
    def w(self):
        return self._w

    @staticmethod
    def _validate_rate_estimator(rate_estimator):
        if rate_estimator is None:
            rate_estimator = FiringRateEstimator()
        elif not isinstance(rate_estimator, FiringRateEstimator):
            raise TypeError(
                "'rate_estimator' must be a nelpy FiringRateEstimator() type!"
            )
        return rate_estimator

    @staticmethod
    def _validate_ratemap(ratemap):
        if ratemap is None:
            ratemap = NDRateMap()
        elif not isinstance(ratemap, NDRateMap):
            raise TypeError("'ratemap' must be a nelpy RateMap() type!")
        return ratemap

    @staticmethod
    def _validate_window(w):
        if w is None:
            w = DataWindow(sum=True, bin_width=1)
        elif not isinstance(w, DataWindow):
            raise TypeError("w must be a nelpy DataWindow() type!")
        else:
            w = copy.copy(w)
        if w._sum is False:
            logging.warning(
                "BayesianDecoder requires DataWindow (w) to have sum=True; changing to True"
            )
            w._sum = True
        if w.bin_width is None:
            w.bin_width = 1
        return w

    def _check_X_dt(self, X, *, lengths=None, dt=None):

        if isinstance(X, core.BinnedEventArray):
            if dt is not None:
                logging.warning(
                    "A {} was passed in, so 'dt' will be ignored...".format(X.type_name)
                )
            dt = X.ds
            if self._w.bin_width != dt:
                raise ValueError(
                    "BayesianDecoder was fit with a bin_width of {}, but is being used to predict data with a bin_width of {}".format(
                        self.w.bin_width, dt
                    )
                )
            X, T = self.w.transform(X, lengths=lengths, sum=True)
        else:
            if dt is not None:
                if self._w.bin_width != dt:
                    raise ValueError(
                        "BayesianDecoder was fit with a bin_width of {}, but is being used to predict data with a bin_width of {}".format(
                            self.w.bin_width, dt
                        )
                    )
            else:
                dt = self._w.bin_width

        return X, dt

    def _check_X_y(self, X, y, *, method="score", lengths=None):

        if isinstance(X, core.BinnedEventArray):
            if method == "fit":
                self._w.bin_width = X.ds
                logging.info("Updating DataWindow.bin_width from training data.")
            else:
                if self._w.bin_width != X.ds:
                    raise ValueError(
                        "BayesianDecoder was fit with a bin_width of {}, but is being used to predict data with a bin_width of {}".format(
                            self.w.bin_width, X.ds
                        )
                    )

            X, T = self.w.transform(X, lengths=lengths, sum=True)

            if isinstance(y, core.RegularlySampledAnalogSignalArray):
                y = y(T).T

        if isinstance(y, core.RegularlySampledAnalogSignalArray):
            raise TypeError(
                "y can only be a RegularlySampledAnalogSignalArray if X is a BinnedEventArray."
            )

        assert len(X) == len(y), "X and y must have the same number of samples!"

        return X, y

    def _ratemap_permute_unit_order(self, unit_ids, inplace=False):
        """Permute the unit ordering.

        If no order is specified, and an ordering exists from fit(), then the
        data in X will automatically be permuted to match that registered during
        fit().

        Parameters
        ----------
        unit_ids : array-like, shape (n_units,)
        """
        unit_ids = self._check_unit_ids(unit_ids=unit_ids)
        if len(unit_ids) != len(self.unit_ids):
            raise ValueError(
                "To re-order (permute) units, 'unit_ids' must have the same length as self._unit_ids."
            )
        self._ratemap.reorder_units_by_ids(unit_ids, inplace=inplace)

    def _check_unit_ids(self, *, X=None, unit_ids=None, fit=False):
        """Check that unit_ids are valid (if provided), and return unit_ids.

        if calling from fit(), pass in fit=True, which will skip checks against
        self.ratemap, which doesn't exist before fitting...

        """

        def a_contains_b(a, b):
            """Returns True iff 'b' is a subset of 'a'."""
            for bb in b:
                if bb not in a:
                    logging.warning("{} was not found in set".format(bb))
                    return False
            return True

        if isinstance(X, core.BinnedEventArray):
            if unit_ids is not None:
                # unit_ids were passed in, even though it's also contained in X.unit_ids
                # 1. check that unit_ids are contained in the data:
                if not a_contains_b(X.series_ids, unit_ids):
                    raise ValueError("Some unit_ids were not contained in X!")
                # 2. check that unit_ids are contained in self (decoder ratemap)
                if not fit:
                    if not a_contains_b(self.unit_ids, unit_ids):
                        raise ValueError("Some unit_ids were not contained in ratemap!")
            else:
                # infer unit_ids from X
                unit_ids = X.series_ids
                # check that unit_ids are contained in self (decoder ratemap)
                if not fit:
                    if not a_contains_b(self.unit_ids, unit_ids):
                        raise ValueError(
                            "Some unit_ids from X were not contained in ratemap!"
                        )
        else:  # a non-nelpy X was passed, possibly X=None
            if unit_ids is not None:
                # 1. check that unit_ids are contained in self (decoder ratemap)
                if not fit:
                    if not a_contains_b(self.unit_ids, unit_ids):
                        raise ValueError("Some unit_ids were not contained in ratemap!")
            else:  # no unit_ids were passed, only a non-nelpy X
                if X is not None:
                    n_samples, n_units = X.shape
                    if not fit:
                        if n_units > self.n_units:
                            raise ValueError(
                                "X contains more units than decoder! {} > {}".format(
                                    n_units, self.n_units
                                )
                            )
                        unit_ids = self.unit_ids[:n_units]
                    else:
                        unit_ids = np.arange(n_units)
                else:
                    raise NotImplementedError("unexpected branch reached...")
        return unit_ids

    def _get_transformed_ratemap(self, unit_ids):
        # first, trim ratemap to subset of units
        ratemap = self.ratemap.loc[unit_ids]
        # then, permute the ratemap
        ratemap = ratemap.reorder_units_by_ids(
            unit_ids
        )  # maybe unneccessary, since .loc already permutes
        return ratemap

    def fit(
        self,
        X,
        y,
        *,
        lengths=None,
        dt=None,
        unit_ids=None,
        n_bins=None,
        sample_weight=None
    ):
        """Fit Gaussian Naive Bayes according to X, y

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
                OR
            nelpy.core.BinnedEventArray / BinnedSpikeTrainArray
                The number of spikes in each time bin for each neuron/unit.
        y : array-like, shape (n_samples, n_output_dims)
            Target values.
                OR
            nelpy.core.RegularlySampledAnalogSignalArray
                containing the target values corresponding to X.
            NOTE: If X is an array-like, then y must be an array-like.
        lengths : array-like, shape (n_epochs,), optional (default=None)
            Lengths (in samples) of contiguous segments in (X, y).
            .. versionadded:: x.xx
                BayesianDecoder does not yet support *lengths*.
        unit_ids : array-like, shape (n_units,), optional (default=None)
            Persistent unit IDs that are used to associate units after
            permutation. Unit IDs are inherited from nelpy.core.BinnedEventArray
            objects, or initialized to np.arange(n_units).
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).
            .. versionadded:: x.xx
               BayesianDecoder does not yet support fitting with *sample_weight*.
        Returns
        -------
        self : object

        """

        # TODO dt should probably come from datawindow specification, but may be overridden here!

        unit_ids = self._check_unit_ids(X=X, unit_ids=unit_ids, fit=True)

        # estimate the firing rate(s):
        self.rate_estimator.fit(X=X, y=y, dt=dt, n_bins=n_bins)

        # store the estimated firing rates as a rate map:
        bin_centers = self.rate_estimator.tc_.bin_centers  # temp code FIXME
        # bins = self.rate_estimator.tc_.bins  # temp code FIXME
        rates = self.rate_estimator.tc_.ratemap  # temp code FIXME
        # unit_ids = np.array(self.rate_estimator.tc_.unit_ids) #temp code FIXME
        self.ratemap.fit(X=bin_centers, y=rates, unit_ids=unit_ids)  # temp code FIXME

        X, y = self._check_X_y(
            X, y, method="fit", lengths=lengths
        )  # can I remove this? no; it sets the bin width... but maybe we should refactor...
        self.ratemap_ = self.ratemap.ratemap_

    def predict(
        self, X, *, output=None, mode="mean", lengths=None, unit_ids=None, dt=None
    ):
        # if output is 'asa', then return an ASA
        check_is_fitted(self, "ratemap_")
        unit_ids = self._check_unit_ids(X=X, unit_ids=unit_ids)
        ratemap = self._get_transformed_ratemap(unit_ids)
        X, dt = self._check_X_dt(X=X, lengths=lengths, dt=dt)

        posterior, mean_pth = decode_bayesian_memoryless_nd(
            X=X, ratemap=ratemap.ratemap_, dt=dt, bin_centers=ratemap.bin_centers
        )

        if output is not None:
            raise NotImplementedError("output mode not implemented yet")
        return posterior, mean_pth

    def predict_proba(self, X, *, lengths=None, unit_ids=None, dt=None):
        check_is_fitted(self, "ratemap_")
        raise NotImplementedError
        ratemap = self._get_transformed_ratemap(unit_ids)
        return self._predict_proba_from_ratemap(X, ratemap)

    def score(self, X, y, *, lengths=None, unit_ids=None, dt=None):

        # check that unit_ids are valid
        # THEN, transform X, y into standardized form (including trimming and permutation) and continue with scoring

        check_is_fitted(self, "ratemap_")
        unit_ids = self._check_unit_ids(X=X, unit_ids=unit_ids)
        ratemap = self._get_transformed_ratemap(unit_ids)
        # X = self._permute_unit_order(X)
        # X, y = self._check_X_y(X, y, method='score', unit_ids=unit_ids)

        raise NotImplementedError
        ratemap = self._get_transformed_ratemap(unit_ids)
        return self._score_from_ratemap(X, ratemap)

    def score_samples(self, X, y, *, lengths=None, unit_ids=None, dt=None):
        # X = self._permute_unit_order(X)
        check_is_fitted(self, "ratemap_")
        raise NotImplementedError

    @property
    def unit_ids(self):
        check_is_fitted(self, "ratemap_")
        return self.ratemap.unit_ids

    @property
    def n_units(self):
        check_is_fitted(self, "ratemap_")
        return len(self.unit_ids)


class FiringRateEstimator(BaseEstimator):
    """
    FiringRateEstimator
    Estimate the firing rate of a spike train.

    Parameters
    ----------
    mode : {'hist', 'glm-poisson', 'glm-binomial', 'glm', 'gvm', 'bars', 'gp'}, optional
        The estimation mode. Default is 'hist'.
        - 'hist': Histogram-based estimation.
        - 'glm-poisson': Generalized linear model with Poisson distribution.
        - 'glm-binomial': Generalized linear model with Binomial distribution.
        - 'glm': Generalized linear model.
        - 'gvm': Generalized von Mises.
        - 'bars': Bayesian adaptive regression splines.
        - 'gp': Gaussian process.

    Attributes
    ----------
    mode : str
        The estimation mode.
    tc_ : TuningCurve1D or TuningCurve2D
        The estimated tuning curve.
    """
    def __init__(self, mode="hist"):
        """
        Initialize a FiringRateEstimator.

        Parameters
        ----------
        mode : str, optional
            The estimation mode. Default is 'hist'.
        """
        if mode not in ["hist"]:
            raise NotImplementedError(
                "mode '{}' not supported / implemented yet!".format(mode)
            )
        self._mode = mode
    def _check_X_y_dt(self, X, y, lengths=None, dt=None, timestamps=None, n_bins=None):
        """
        Validate and standardize input data for fitting or prediction.

        Parameters
        ----------
        X : array-like or BinnedEventArray
            Input data.
        y : array-like or RegularlySampledAnalogSignalArray
            Target values.
        lengths : array-like, optional
            Lengths of intervals.
        dt : float, optional
            Temporal bin size.
        timestamps : array-like, optional
            Timestamps for the data.
        n_bins : int or array-like, optional
            Number of bins for discretization.

        Returns
        -------
        X : np.ndarray
            Standardized input data.
        y : np.ndarray
            Standardized target values.
        dt : float
            Temporal bin size.
        n_bins : int or array-like
            Number of bins for discretization.
        """
        if isinstance(X, core.BinnedEventArray):
            T = X.bin_centers
            if lengths is not None:
                logging.warning(
                    "'lengths' was passed in, but will be"
                    " overwritten by 'X's 'lengths' attribute"
                )
            if timestamps is not None:
                logging.warning(
                    "'timestamps' was passed in, but will be"
                    " overwritten by 'X's 'bin_centers' attribute"
                )
            if dt is not None:
                logging.warning(
                    "'dt' was passed in, but will be overwritten"
                    " by 'X's 'ds' attribute"
                )
            if isinstance(y, core.RegularlySampledAnalogSignalArray):
                y = y(T).T

            dt = X.ds
            lengths = X.lengths
            X = X.data.T
        elif isinstance(X, np.ndarray):
            if dt is None:
                raise ValueError(
                    "'dt' is a required argument when 'X' is passed in as a numpy array!"
                )
            if isinstance(y, core.RegularlySampledAnalogSignalArray):
                if timestamps is not None:
                    y = y(timestamps).T
                else:
                    raise ValueError(
                        "'timestamps' required when passing in 'X' as a numpy array and 'y' as a nelpy RegularlySampledAnalogSignalArray!"
                    )
        else:
            raise TypeError(
                "'X' should be either a nelpy BinnedEventArray, or a numpy array!"
            )

        n_samples, n_units = X.shape
        _, n_dims = y.shape
        print("{}-dimensional y passed in".format(n_dims))

        assert n_samples == len(y), (
            "'X' and 'y' must have the same number"
            " of samples! len(X)=={} but len(y)=={}".format(n_samples, len(y))
        )
        if n_bins is not None:
            n_bins = np.atleast_1d(n_bins)
            assert (
                len(n_bins) == n_dims
            ), "'n_bins' must have one entry for each dimension in 'y'!"

        return X, y, dt, n_bins
    def fit(
        self,
        X,
        y,
        lengths=None,
        dt=None,
        timestamps=None,
        unit_ids=None,
        n_bins=None,
        sample_weight=None,
    ):
        """
        Fit the firing rate estimator to the data.

        Parameters
        ----------
        X : array-like
            Input data.
        y : array-like
            Target values.
        lengths : array-like, optional
            Lengths of intervals.
        dt : float, optional
            Temporal bin size.
        timestamps : array-like, optional
            Timestamps for the data.
        unit_ids : array-like, optional
            Unit identifiers.
        n_bins : int or array-like, optional
            Number of bins for discretization.
        sample_weight : array-like, optional
            Weights for each sample.

        Returns
        -------
        self : FiringRateEstimator
            The fitted estimator.
        """
        X, y, dt, n_bins = self._check_X_y_dt(
            X=X, y=y, lengths=lengths, dt=dt, timestamps=timestamps, n_bins=n_bins
        )

        # 1. estimate mask
        # 2. estimate occupancy
        # 3. compute spikes histogram
        # 4. normalize spike histogram by occupancy
        # 5. apply mask

        # if y.n_signals == 1:
        #     self.tc_ = TuningCurve1D(bst=X, extern=y, n_extern=100, extmin=y.min(), extmax=y.max(), sigma=2.5, min_duration=0)
        # if y.n_signals == 2:
        #     xmin, ymin = y.min()
        #     xmax, ymax = y.max()
        #     self.tc_ = TuningCurve2D(bst=X, extern=y, ext_nx=50, ext_ny=50, ext_xmin=xmin, ext_xmax=xmax, ext_ymin=ymin, ext_ymax=ymax, sigma=2.5, min_duration=0)

    @property
    def mode(self):
        return self._mode

    def predict(self, X, lengths=None):
        """
        Predict firing rates for the given input data.

        Parameters
        ----------
        X : array-like
            Input data.
        lengths : array-like, optional
            Lengths of intervals.

        Returns
        -------
        rates : array-like
            Predicted firing rates.
        """
        raise NotImplementedError

    def predict_proba(self, X, lengths=None):
        """
        Predict firing rate probabilities for the given input data.

        Parameters
        ----------
        X : array-like
            Input data.
        lengths : array-like, optional
            Lengths of intervals.

        Returns
        -------
        probabilities : array-like
            Predicted probabilities.
        """
        raise NotImplementedError

    def score(self, X, y, lengths=None):
        """
        Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True values for X.
        lengths : array-like, optional
            Lengths of intervals.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        raise NotImplementedError

    def score_samples(self, X, y, lengths=None):
        """
        Return the per-sample accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like
            Test samples.
        y : array-like
            True values for X.
        lengths : array-like, optional
            Lengths of intervals.

        Returns
        -------
        scores : array-like
            Per-sample accuracy of self.predict(X) wrt. y.
        """
        raise NotImplementedError


# def decode_bayes_from_ratemap_1d(X, ratemap, dt, xmin, xmax, bin_centers):
#     """
#     X has been standardized to (n_samples, n_units), where each sample is a singleton window
#     """
#     n_samples, n_features = X.shape
#     n_units, n_xbins = ratemap.shape

#     assert n_features == n_units, "X has {} units, whereas ratemap has {}".format(n_features, n_units)

#     lfx = np.log(ratemap)
#     eterm = -ratemap.sum(axis=0)*dt

#     posterior = np.empty((n_xbins, n_samples))
#     posterior[:] = np.nan

#     # decode each sample / bin separately
#     for tt in range(n_samples):
#         obs = X[tt]
#         if obs.sum() > 0:
#             posterior[:,tt] = (np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx).sum(axis=0) + eterm

#     # normalize posterior:
#     posterior = np.exp(posterior - logsumexp(posterior, axis=0))

#     mode_pth = np.argmax(posterior, axis=0)*xmax/n_xbins
#     mode_pth = np.where(np.isnan(posterior.sum(axis=0)), np.nan, mode_pth)
#     mean_pth = (bin_centers * posterior.T).sum(axis=1)

#     return posterior, mode_pth, mean_pth


def decode_bayesian_memoryless_nd(X, *, ratemap, bin_centers, dt=1):
    """Memoryless Bayesian decoding (supports multidimensional decoding).

    Decode binned spike counts (e.g. from a BinnedSpikeTrainArray) to an
    external correlate (e.g. position), using a memoryless Bayesian decoder and
    a previously estimated ratemap.

    Parameters
    ----------
    X : numpy array with shape (n_samples, n_features),
        where the features are generally putative units / cells, and where
        each sample represents spike counts in a singleton data window.
    ratemap : array-like of shape (n_units, n_bins_d1, ..., n_bins_dN)
        Expected number of spikes for each unit, within each bin, along each
        dimension.
    bin_centers : array-like with shape (n_dims, ), where each element is also
        an array-like with shape (n_bins_dn, ) containing the bin centers for
        the particular dimension.
    dt : float, optional (default=1)
        Temporal bin width corresponding to X, in seconds.

        NOTE: generally it is assumed that ratemap will be given in Hz (that is,
        it has dt=1). If ratemap has a different unit, then dt might have to be
        adjusted to compensate for this. This can get tricky / confusing, so the
        recommended approach is always to construct ratemap with dt=1, and then
        to use the data-specific dt here when decoding.

    Returns
    -------
    posterior : numpy array of shape (n_samples, n_bins_d1, ..., n_bins_dN)
        Posterior probabilities for each voxel.
    expected_pth : numpy array of shape (n_samples, n_dims)
        Expected (posterior-averaged) decoded trajectory.
    """

    def tile_obs(obs, *n_bins):
        n_units = len(obs)
        out = np.zeros((n_units, *n_bins))
        for unit in range(n_units):
            out[unit, :] = obs[unit]
        return out

    n_samples, n_features = X.shape
    n_units = ratemap.shape[0]
    n_bins = np.atleast_1d(ratemap.shape[1:])
    n_dims = len(n_bins)

    assert n_features == n_units, "X has {} units, whereas ratemap has {}".format(
        n_features, n_units
    )

    lfx = np.log(ratemap)
    eterm = -ratemap.sum(axis=0) * dt

    posterior = np.empty((n_samples, *n_bins))
    posterior[:] = np.nan

    # decode each sample / bin separately
    for tt in range(n_samples):
        obs = X[tt]
        if obs.sum() > 0:
            posterior[tt] = (tile_obs(obs, *n_bins) * lfx).sum(axis=0) + eterm

    # normalize posterior:
    posterior = np.exp(
        posterior
        - logsumexp(posterior, axis=tuple(np.arange(1, n_dims + 1)), keepdims=True)
    )

    if n_dims > 1:
        expected = []
        for dd in range(1, n_dims + 1):
            axes = tuple(set(np.arange(1, n_dims + 1)) - set([dd]))
            expected.append(
                (bin_centers[dd - 1] * posterior.sum(axis=axes)).sum(axis=1)
            )
        expected_pth = np.vstack(expected).T
    else:
        expected_pth = (bin_centers * posterior).sum(axis=1)

    return posterior, expected_pth


class NDRateMap(BaseEstimator):
    """
    NDRateMap with persistent unit_ids and firing rates in Hz for N-dimensional data.

    Parameters
    ----------
    connectivity : {'continuous', 'discrete', 'circular'}, optional
        Defines how smoothing is applied. Default is 'continuous'.
        - 'continuous': Continuous smoothing.
        - 'discrete': No smoothing is applied.
        - 'circular': Circular smoothing (for angular variables).

    Attributes
    ----------
    connectivity : str
        Smoothing mode.
    ratemap_ : np.ndarray
        The estimated firing rate map.
    _unit_ids : np.ndarray
        Persistent unit IDs.
    _bins : np.ndarray
        Bin edges for each dimension.
    _bin_centers : np.ndarray
        Bin centers for each dimension.
    _mask : np.ndarray
        Mask for valid regions.
    """

    def __init__(self, connectivity="continuous"):
        self.connectivity = connectivity

        self._slicer = UnitSlicer(self)
        self.loc = ItemGetter_loc(self)
        self.iloc = ItemGetter_iloc(self)

    def __repr__(self):
        r = super().__repr__()
        if self._is_fitted():
            dimstr = ""
            for dd in range(self.n_dims):
                dimstr += ", n_bins_d{}={}".format(dd + 1, self.shape[dd + 1])
            r += " with shape (n_units={}{})".format(self.n_units, dimstr)
        return r

    def fit(self, X, y, dt=1, unit_ids=None):
        """
        Fit firing rates to the provided data.

        Parameters
        ----------
        X : array-like, with shape (n_dims, ), each element of which has
            shape (n_bins_dn, ) for n=1, ..., N; N=n_dims.
            Bin locations (centers) where ratemap is defined.
        y : array-like, shape (n_units, n_bins_d1, ..., n_bins_dN)
            Expected number of spikes in a temporal bin of width dt, for each of
            the predictor bins specified in X.
        dt : float, optional
            Temporal bin size with which firing rate y is defined. Default is 1.
        unit_ids : array-like, shape (n_units,), optional
            Persistent unit IDs that are used to associate units after
            permutation. If None, uses np.arange(n_units).

        Returns
        -------
        self : NDRateMap
            The fitted NDRateMap instance.
        """
        n_units, n_bins, n_dims = self._check_X_y(X, y)

        self.ratemap_ = y / dt
        self._bin_centers = X
        self._bins = np.array(n_dims * [None])

        if n_dims > 1:
            for dd in range(n_dims):
                bin_centers = np.squeeze(X[dd])
                dx = np.median(np.diff(bin_centers))
                bins = np.insert(
                    bin_centers[-1] + np.diff(bin_centers) / 2,
                    0,
                    bin_centers[0] - dx / 2,
                )
                bins = np.append(bins, bins[-1] + dx)
                self._bins[dd] = bins
        else:
            bin_centers = np.squeeze(X)
            dx = np.median(np.diff(bin_centers))
            bins = np.insert(
                bin_centers[-1] + np.diff(bin_centers) / 2, 0, bin_centers[0] - dx / 2
            )
            bins = np.append(bins, bins[-1] + dx)
            self._bins = bins

        if unit_ids is not None:
            if len(unit_ids) != n_units:
                raise ValueError(
                    "'unit_ids' must have same number of elements as 'n_units'. {} != {}".format(
                        len(unit_ids), n_units
                    )
                )
            self._unit_ids = unit_ids
        else:
            self._unit_ids = np.arange(n_units)

    def predict(self, X):
        """
        Predict firing rates for the given bin locations.

        Parameters
        ----------
        X : array-like
            Bin locations to predict firing rates for.

        Returns
        -------
        rates : array-like
            Predicted firing rates.
        """
        check_is_fitted(self, "ratemap_")
        raise NotImplementedError

    def synthesize(self, X):
        """
        Generate synthetic spike data based on the ratemap.

        Parameters
        ----------
        X : array-like
            Bin locations to synthesize spikes for.

        Returns
        -------
        spikes : array-like
            Synthetic spike data.
        """
        check_is_fitted(self, "ratemap_")
        raise NotImplementedError

    def __len__(self):
        return self.n_units

    def __iter__(self):
        """TuningCurve1D iterator initialization"""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """TuningCurve1D iterator advancer."""
        index = self._index
        if index > self.n_units - 1:
            raise StopIteration
        out = copy.copy(self)
        out.ratemap_ = self.ratemap_[tuple([index])]
        out._unit_ids = self._unit_ids[index]
        self._index += 1
        return out

    def __getitem__(self, *idx):
        """
        Access RateMap units by index.

        Parameters
        ----------
        *idx : int, slice, or list
            Indices of units to access.

        Returns
        -------
        out : NDRateMap
            Subset NDRateMap with selected units.
        """
        idx = [ii for ii in idx]
        if len(idx) == 1 and not isinstance(idx[0], int):
            idx = idx[0]
        if isinstance(idx, tuple):
            idx = [ii for ii in idx]

        try:
            out = copy.copy(self)
            out.ratemap_ = self.ratemap_[tuple([idx])]
            out._unit_ids = list(np.array(out._unit_ids)[tuple([idx])])
            out._slicer = UnitSlicer(out)
            out.loc = ItemGetter_loc(out)
            out.iloc = ItemGetter_iloc(out)
            return out
        except Exception:
            raise TypeError("unsupported subsctipting type {}".format(type(idx)))

    def get_peak_firing_order_ids(self):
        """Get the unit_ids in order of peak firing location for 1D RateMaps.

        Returns
        -------
        unit_ids : array-like
            The permutaiton of unit_ids such that after reordering, the peak
            firing locations are ordered along the RateMap.
        """
        check_is_fitted(self, "ratemap_")
        if self.is_2d:
            raise NotImplementedError(
                "get_peak_firing_order_ids() only implemented for 1D RateMaps."
            )
        peakorder = np.argmax(self.ratemap_, axis=1).argsort()
        return np.array(self.unit_ids)[peakorder]

    def reorder_units_by_ids(self, unit_ids, inplace=False):
        """Permute the unit ordering.

        #TODO
        If no order is specified, and an ordering exists from fit(), then the
        data in X will automatically be permuted to match that registered during
        fit().

        Parameters
        ----------
        unit_ids : array-like, shape (n_units,)

        Returns
        -------
        out : reordered RateMap
        """

        def swap_units(arr, frm, to):
            """swap 'units' of a 3D np.array"""
            arr[(frm, to), :] = arr[(to, frm), :]

        self._validate_unit_ids(unit_ids)
        if len(unit_ids) != len(self._unit_ids):
            raise ValueError(
                "unit_ids must be a permutation of self.unit_ids, not a subset thereof."
            )

        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        neworder = [list(self.unit_ids).index(x) for x in unit_ids]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            swap_units(out.ratemap_, frm, to)
            out._unit_ids[frm], out._unit_ids[to] = (
                out._unit_ids[to],
                out._unit_ids[frm],
            )
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]
        return out

    def _check_X_y(self, X, y):
        y = np.atleast_2d(y)

        n_units = y.shape[0]
        n_bins = y.shape[1:]
        n_dims = len(n_bins)

        if n_dims > 1:
            n_x_bins = tuple([len(x) for x in X])
        else:
            n_x_bins = tuple([len(X)])

        assert n_units > 0, "n_units must be a positive integer!"
        assert n_x_bins == n_bins, "X and y must have the same number of bins!"

        return n_units, n_bins, n_dims

    def _validate_unit_ids(self, unit_ids):
        self._check_unit_ids_in_ratemap(unit_ids)

        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("Duplicate unit_ids are not allowed.")

    def _check_unit_ids_in_ratemap(self, unit_ids):
        for unit_id in unit_ids:
            # NOTE: the check below allows for predict() to pass on only
            # a subset of the units that were used during fit! So we
            # could fit on 100 units, and then predict on only 10 of
            # them, if we wanted.
            if unit_id not in self.unit_ids:
                raise ValueError(
                    "unit_id {} was not present during fit(); aborting...".format(
                        unit_id
                    )
                )

    def _is_fitted(self):
        try:
            check_is_fitted(self, "ratemap_")
        except Exception:  # should really be except NotFitterError
            return False
        return True

    @property
    def connectivity(self):
        return self._connectivity

    @connectivity.setter
    def connectivity(self, val):
        self._connectivity = self._validate_connectivity(val)

    @staticmethod
    def _validate_connectivity(connectivity):
        connectivity = str(connectivity).strip().lower()
        options = ["continuous", "discrete", "circular"]
        if connectivity in options:
            return connectivity
        raise NotImplementedError(
            "connectivity '{}' is not supported yet!".format(str(connectivity))
        )

    @property
    def shape(self):
        """
        RateMap.shape = (n_units, n_features_x, n_features_y)
            OR
        RateMap.shape = (n_units, n_features)
        """
        check_is_fitted(self, "ratemap_")
        return self.ratemap_.shape

    @property
    def n_dims(self):
        check_is_fitted(self, "ratemap_")
        n_dims = len(self.shape) - 1
        return n_dims

    @property
    def is_1d(self):
        check_is_fitted(self, "ratemap_")
        if len(self.ratemap_.shape) == 2:
            return True
        return False

    @property
    def is_2d(self):
        check_is_fitted(self, "ratemap_")
        if len(self.ratemap_.shape) == 3:
            return True
        return False

    @property
    def n_units(self):
        check_is_fitted(self, "ratemap_")
        return self.ratemap_.shape[0]

    @property
    def unit_ids(self):
        check_is_fitted(self, "ratemap_")
        return self._unit_ids

    @property
    def n_bins(self):
        """(int) Number of external correlates (bins) along each dimension."""
        check_is_fitted(self, "ratemap_")
        if self.n_dims > 1:
            n_bins = tuple([len(x) for x in self.bin_centers])
        else:
            n_bins = len(self.bin_centers)
        return n_bins

    def max(self, axis=None, out=None):
        """
        maximum firing rate for each unit:
            RateMap.max()
        maximum firing rate across units:
            RateMap.max(axis=0)
        """
        raise NotImplementedError("the code was still for the 1D and 2D only version")
        check_is_fitted(self, "ratemap_")
        if axis is None:
            if self.is_2d:
                return self.ratemap_.max(axis=1, out=out).max(axis=1, out=out)
            else:
                return self.ratemap_.max(axis=1, out=out)
        return self.ratemap_.max(axis=axis, out=out)

    def min(self, axis=None, out=None):
        raise NotImplementedError("the code was still for the 1D and 2D only version")
        check_is_fitted(self, "ratemap_")
        if axis is None:
            if self.is_2d:
                return self.ratemap_.min(axis=1, out=out).min(axis=1, out=out)
            else:
                return self.ratemap_.min(axis=1, out=out)
        return self.ratemap_.min(axis=axis, out=out)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        raise NotImplementedError("the code was still for the 1D and 2D only version")
        check_is_fitted(self, "ratemap_")
        kwargs = {"dtype": dtype, "out": out, "keepdims": keepdims}
        if axis is None:
            if self.is_2d:
                return self.ratemap_.mean(axis=1, **kwargs).mean(axis=1, **kwargs)
            else:
                return self.ratemap_.mean(axis=1, **kwargs)
        return self.ratemap_.mean(axis=axis, **kwargs)

    @property
    def bins(self):
        return self._bins

    @property
    def bin_centers(self):
        return self._bin_centers

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        # TODO: mask validation
        raise NotImplementedError
        self._mask = val

    def plot(self, **kwargs):
        check_is_fitted(self, "ratemap_")
        if self.is_2d:
            raise NotImplementedError("plot() not yet implemented for 2D RateMaps.")
        pad = kwargs.pop("pad", None)
        _plot_ratemap(self, pad=pad, **kwargs)

    @keyword_deprecation(replace_x_with_y={"bw": "truncate"})
    def smooth(self, *, sigma=None, truncate=None, inplace=False, mode=None, cval=None):
        """Smooths the tuning curve with a Gaussian kernel.

        mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
            The mode parameter determines how the array borders are handled,
            where cval is the value when mode is equal to ‘constant’. Default is
            ‘reflect’
        truncate : float
            Truncate the filter at this many standard deviations. Default is 4.0.
        truncate : float, deprecated
            Truncate the filter at this many standard deviations. Default is 4.0.
        cval : scalar, optional
            Value to fill past edges of input if mode is ‘constant’. Default is 0.0
        """

        if sigma is None:
            sigma = 0.1  # in units of extern
        if truncate is None:
            truncate = 4
        if mode is None:
            mode = "reflect"
        if cval is None:
            cval = 0.0

        raise NotImplementedError
