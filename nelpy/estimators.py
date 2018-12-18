import numpy as np
import logging
import copy

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted, NotFittedError

from .preprocessing import DataWindow
from . import core

"""
FiringRateEstimator(BaseEstimator) DRAFT SPECIFICATION
    X : BST / spike counts
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

class RateMap(BaseEstimator):
    """
    RateMap with persistent unit_ids and firing rates in Hz.

    mode = ['continuous', 'discrete', 'circular']

    fit(X, y) estimates ratemap [discrete, continuous, circular]
    predict(X) predicts firing rate
    synthesize(X) generates spikes based on input (inhomogenous Poisson?)

    Parameters
    ----------
    X : array-like, shape (n_bins,), or (n_bins_x, n_bins_y)
        Bin locations where ratemap is defined.
    y : array-like, shape (n_units, n_bins) or (n_units, n_bins_x, n_bins_y)
    connectivity : string ['continuous', 'discrete', 'circular'], optional
        Defines how smoothing is applied. If 'discrete', then no smoothing is
        applied. Default is 'continuous'.
    unit_ids : array-like, shape (n_units,), optional (default=None)
        Persistent unit IDs that are used to associate units after
        permutation. Unit IDs are inherited from nelpy.core.BinnedEventArray
        objects, or initialized to np.arange(n_units).
    """

    def __init__(self, connectivity='continuous'):
        self.connectivity = connectivity

    def __repr__(self):
        r = super().__repr__()
        if self._is_fitted():
            if self.is_1d:
                r += ' with shape (n_units={}, n_bins_x={})'.format(*self.shape)
            else:
                r += ' with shape (n_units={}, n_bins_x={}, n_bins_y={})'.format(*self.shape)
        return r

    def fit(self, X, y, unit_ids=None):
        n_units, n_bins_x, n_bins_y = self._check_X_y(X, y)
        if n_bins_y > 0:
            self.ratemap_ = np.zeros((n_units, n_bins_x, n_bins_y)) #FIXME
        else:
            self.ratemap_ = np.zeros((n_units, n_bins_x)) #FIXME

        self._unit_ids = np.arange(n_units) #FIXME

    def predict(self, X):
        check_is_fitted(self, 'ratemap_')
        raise NotImplementedError

    def synthesize(self, X):
        check_is_fitted(self, 'ratemap_')
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
        out.ratemap_ = self.ratemap_[[index]]
        out._unit_ids = self._unit_ids[index]
        self._index += 1
        return out

    def __getitem__(self, *idx):
        """TuningCurve1D index access.

        Accepts integers, slices, and lists"""
        idx = [ii for ii in idx]
        if len(idx) == 1 and not isinstance(idx[0], int):
            idx = idx[0]
        if isinstance(idx, tuple):
            idx = [ii for ii in idx]

        try:
            out = copy.copy(self)
            out.ratemap_ = self.ratemap_[[idx]]
            out._unit_ids = out._unit_ids[idx]
            return out
        except Exception:
            raise TypeError(
                'unsupported subsctipting type {}'.format(type(idx)))

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
            arr[[frm, to],:] = arr[[to, frm],:]

        self._validate_unit_ids(unit_ids)
        if len(unit_ids) != len(self._unit_ids):
            raise ValueError('unit_ids must be a permutation of self.unit_ids, not a subset thereof.')

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
            out._unit_ids[frm], out._unit_ids[to] = out._unit_ids[to], out._unit_ids[frm]
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
                raise ValueError('unit_id {} was not present during fit(); aborting...'.format(unit_id))

    def _is_fitted(self):
        try:
            check_is_fitted(self, 'ratemap_')
        except NotFittedError:
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
        options = ['continuous', 'discrete', 'circular']
        if connectivity in options:
            return connectivity
        raise NotImplementedError("connectivity '{}' is not supported yet!".format(str(connectivity)))

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
        out.ratemap_ = np.transpose(out.ratemap_, axes=(0,2,1))
        return out

    @property
    def shape(self):
        """
            RateMap.shape = (n_units, n_features_x, n_features_y)
                OR
            RateMap.shape = (n_units, n_features)
        """
        check_is_fitted(self, 'ratemap_')
        return self.ratemap_.shape

    @property
    def is_1d(self):
        check_is_fitted(self, 'ratemap_')
        if len(self.ratemap_.shape) == 2:
            return True
        return False

    @property
    def is_2d(self):
        check_is_fitted(self, 'ratemap_')
        if len(self.ratemap_.shape) == 3:
            return True
        return False

    @property
    def n_units(self):
        check_is_fitted(self, 'ratemap_')
        return self.ratemap_.shape[0]

    @property
    def unit_ids(self):
        check_is_fitted(self, 'ratemap_')
        return self._unit_ids

    @property
    def n_bins(self):
        """(int) Number of external correlates (bins)."""
        check_is_fitted(self, 'ratemap_')
        if self.is_2d:
            return self.n_bins_x*self.n_bins_y
        return self.n_bins_x

    @property
    def n_bins_x(self):
        """(int) Number of external correlates (bins)."""
        check_is_fitted(self, 'ratemap_')
        return self.ratemap_.shape[1]

    @property
    def n_bins_y(self):
        """(int) Number of external correlates (bins)."""
        check_is_fitted(self, 'ratemap_')
        if self.is_1d:
            raise ValueError('RateMap is 1D; no y bins are defined.')
        return self.ratemap_.shape[2]

    def max(self, axis=None, out=None):
        """
        maximum firing rate for each unit:
            RateMap.max()
        maximum firing rate across units:
            RateMap.max(axis=0)
        """
        check_is_fitted(self, 'ratemap_')
        if axis == None:
            if self.is_2d:
                return self.ratemap_.max(axis=1, out=out).max(axis=1, out=out)
            else:
                return self.ratemap_.max(axis=1, out=out)
        return self.ratemap_.max(axis=axis, out=out)

    def min(self, axis=None, out=None):
        check_is_fitted(self, 'ratemap_')
        if axis == None:
            if self.is_2d:
                return self.ratemap_.min(axis=1, out=out).min(axis=1, out=out)
            else:
                return self.ratemap_.min(axis=1, out=out)
        return self.ratemap_.min(axis=axis, out=out)

    def mean(self, axis=None, dtype=None, out=None, keepdims=False):
        check_is_fitted(self, 'ratemap_')
        kwargs = {'dtype':dtype,
                  'out':out,
                  'keepdims':keepdims}
        if axis == None:
            if self.is_2d:
                return self.ratemap_.mean(axis=1, **kwargs).mean(axis=1, **kwargs)
            else:
                return self.ratemap_.mean(axis=1, **kwargs)
        return self.ratemap_.mean(axis=axis, **kwargs)

    @property
    def bins(self):
        raise NotImplementedError

    @property
    def bins_x(self):
        raise NotImplementedError

    @property
    def bins_y(self):
        raise NotImplementedError

    @property
    def bins_centers(self):
        raise NotImplementedError

    @property
    def bins_centers_x(self):
        raise NotImplementedError

    @property
    def bins_centers_y(self):
        raise NotImplementedError

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, val):
        #TODO: mask validation
        raise NotImplementedError
        self._mask = val

    def plot(self, **kwargs):
        raise NotImplementedError

    def smooth(self, *, sigma=None, bw=None, inplace=False, mode=None, cval=None):
        """Smooths the tuning curve with a Gaussian kernel.

        mode : {‘reflect’, ‘constant’, ‘nearest’, ‘mirror’, ‘wrap’}, optional
            The mode parameter determines how the array borders are handled,
            where cval is the value when mode is equal to ‘constant’. Default is
            ‘reflect’
        cval : scalar, optional
            Value to fill past edges of input if mode is ‘constant’. Default is 0.0
        """
        if sigma is None:
            sigma = 0.1 # in units of extern
        if bw is None:
            bw = 4
        if mode is None:
            mode = 'reflect'
        if cval is None:
            cval = 0.0

        raise NotImplementedError


class BayesianDecoderTemp(BaseEstimator):
    """
    Bayesian decoder wrapper class.

    mode = ['hist', 'glm-poisson', 'glm-binomial', 'glm', 'gvm', 'bars', 'gp']

    (gvm = generalized von mises; see http://kordinglab.com/spykes/getting-started.html)

    QQQ. Do we always bin first? does GLM and BARS use spike times, or binned
         spike counts? I think GLM uses binned spike counts with Poisson
         regression; not sure about BARS.

    QQQ. What other methods should be supported? BAKS? What is state of the art?

    QQQ. What if we want to know the fring rate over time? What does the input y
         look like then? How about trial averaged? How about a tuning curve?

    AAA. At the end of the day, this class should estimate a ratemap, and we
         need some way to set the domain of that ratemap, if desired, but it
         should not have to assume anything else. Values in y might be repeated,
         but if not, then we estimate the (single-trial) firing rate over time
         or whatever the associated y represents.

    See https://arxiv.org/pdf/1602.07389.pdf for more GLM intuition? and http://www.stat.columbia.edu/~liam/teaching/neurostat-fall18/glm-notes.pdf

    [2] https://www.biorxiv.org/content/biorxiv/early/2017/02/24/111450.full.pdf?%3Fcollection=
    http://kordinglab.com/spykes/getting-started.html
    https://xcorr.net/2011/10/03/using-the-binomial-glm-instead-of-the-poisson-for-spike-data/

    [1] http://www.stat.cmu.edu/~kass/papers/bars.pdf
    https://gist.github.com/AustinRochford/d640a240af12f6869a7b9b592485ca15
    https://discourse.pymc.io/t/bayesian-adaptive-regression-splines-and-mcmc-some-questions/756/5

    """

    def __init__(self, mode='hist', w=None):
        self._mode = self._validate_mode(mode)
        self._w = self._validate_window(w)

    @property
    def mode(self):
        return self._mode

    @property
    def w(self):
        return self._w

    @staticmethod
    def _validate_mode(mode):
        mode = str(mode).strip().lower()
        valid_modes = ['hist']
        if mode in valid_modes:
            return mode
        raise NotImplementedError("mode '{}' is not supported yet!".format(str(mode)))

    @staticmethod
    def _validate_window(w):
        if w is None:
            w = DataWindow(sum=True, bin_width=1)
        elif not isinstance(w, DataWindow):
            raise TypeError('w must be a nelpy DataWindow() type!')
        else:
            w = copy.copy(w)
        if w._sum is False:
            logging.warning('BayesianDecoder requires DataWindow (w) to have sum=True; changing to True')
            w._sum = True
        if w.bin_width is None:
            w.bin_width = 1
        return w

    def _check_X_y(self, X, y, *, method='fit', unit_ids=None):

        unit_ids = self._check_unit_ids_from_X(X, method=method, unit_ids=unit_ids)
        if isinstance(X, core.BinnedEventArray):
            if method == 'fit':
                self._w.bin_width = X.ds
                logging.info('Updating DataWindow.bin_width from training data.')
            else:
                if self._w.bin_width != X.ds:
                    raise ValueError('BayesianDecoder was fit with a bin_width of {}, but is being used to predict data with a bin_width of {}'.format(self.w.bin_width, X.ds))

            X, T = self.w.transform(X)

            if isinstance(y, core.RegularlySampledAnalogSignalArray):
                y = y(T).T

        if isinstance(y, core.RegularlySampledAnalogSignalArray):
            raise TypeError('y can only be a RegularlySampledAnalogSignalArray if X is a BinnedEventArray.')

        assert len(X) == len(y), "X and y must have the same number of samples!"

        return X, y

    def _ratemap_permute_unit_order(self, unit_ids):
        """Permute the unit ordering.

        If no order is specified, and an ordering exists from fit(), then the
        data in X will automatically be permuted to match that registered during
        fit().

        Parameters
        ----------
        unit_ids : array-like, shape (n_units,)
        """
        unit_ids = self._check_unit_ids(unit_ids)
        if len(unit_ids) != len(self._unit_ids):
            raise ValueError("To re-order (permute) units, 'unit_ids' must have the same length as self._unit_ids.")
        raise NotImplementedError("Ratemap re-ordering has not yet been implemented.")

    def _check_unit_ids(self, unit_ids):
        for unit_id in unit_ids:
            # NOTE: the check below allows for predict() to pass on only
            # a subset of the units that were used during fit! So we
            # could fit on 100 units, and then predict on only 10 of
            # them, if we wanted.
            if unit_id not in self._unit_ids:
                raise ValueError('unit_id {} was not present during fit(); aborting...'.format(unit_id))

    def _check_unit_ids_from_X(self, X, *, method='fit', unit_ids=None):
        if isinstance(X, core.BinnedEventArray):
            if unit_ids is not None:
                logging.warning("X is a nelpy BinnedEventArray; kwarg 'unit_ids' will be ignored")
            unit_ids = X.series_ids
            n_units = X.n_series
        else:
            n_units = X.shape[-1]
            if unit_ids is None:
                unit_ids = np.arange(n_units)

        if len(unit_ids) != n_units:
            raise ValueError("'X' has {} units, but 'unit_ids' has shape ({},).".format(n_units, len(unit_ids)))
        if method == 'fit':
            self._unit_ids = unit_ids
        else:
            self._check_unit_ids(unit_ids)
        return unit_ids

    def fit(self, X, y, *, lengths=None, unit_ids=None, sample_weight=None):
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

        # self._check_unit_ids(X, unit_ids, method='fit')
        X, y = self._check_X_y(X, y, method='fit')

        self.ratemap_ = None
        # return self._partial_fit(X, y, np.unique(y), _refit=True,
        #                          sample_weight=sample_weight)

    def predict(self, X, *, output=None, lengths=None, unit_ids=None):
        # if output is 'asa', then return an ASA
        check_is_fitted(self, 'ratemap_')
        unit_ids = self._check_unit_ids_from_X(X, unit_ids=unit_ids)
        raise NotImplementedError
        ratemap = self._get_temp_ratemap(unit_ids)
        return self._predict_from_ratemap(X, ratemap)

    def predict_proba(self, X, *, lengths=None, unit_ids=None):
        check_is_fitted(self, 'ratemap_')
        unit_ids = self._check_unit_ids_from_X(X, unit_ids=unit_ids)
        raise NotImplementedError
        ratemap = self._get_temp_ratemap(unit_ids)
        return self._predict_proba_from_ratemap(X, ratemap)

    def score(self, X, y, *, lengths=None, unit_ids=None):
        check_is_fitted(self, 'ratemap_')
        # X = self._permute_unit_order(X)
        X, y = self._check_X_y(X, y, method='score', unit_ids=unit_ids)
        unit_ids = self._check_unit_ids_from_X(X, unit_ids=unit_ids)
        raise NotImplementedError
        ratemap = self._get_temp_ratemap(unit_ids)
        return self._score_from_ratemap(X, ratemap)

    def score_samples(self, X, y, *, lengths=None, unit_ids=None):
        # X = self._permute_unit_order(X)
        check_is_fitted(self, 'ratemap_')
        raise NotImplementedError


class FiringRateEstimator(BaseEstimator):
    """
    FiringRateEstimator
    Estimate the firing rate of a spike train.

    mode = ['hist', 'glm-poisson', 'glm-binomial', 'glm', 'gvm', 'bars', 'gp']

    (gvm = generalized von mises; see http://kordinglab.com/spykes/getting-started.html)

    QQQ. Do we always bin first? does GLM and BARS use spike times, or binned
         spike counts? I think GLM uses binned spike counts with Poisson
         regression; not sure about BARS.

    QQQ. What other methods should be supported? BAKS? What is state of the art?

    QQQ. What if we want to know the fring rate over time? What does the input y
         look like then? How about trial averaged? How about a tuning curve?

    AAA. At the end of the day, this class should estimate a ratemap, and we
         need some way to set the domain of that ratemap, if desired, but it
         should not have to assume anything else. Values in y might be repeated,
         but if not, then we estimate the (single-trial) firing rate over time
         or whatever the associated y represents.

    See https://arxiv.org/pdf/1602.07389.pdf for more GLM intuition? and http://www.stat.columbia.edu/~liam/teaching/neurostat-fall18/glm-notes.pdf

    [2] https://www.biorxiv.org/content/biorxiv/early/2017/02/24/111450.full.pdf?%3Fcollection=
    http://kordinglab.com/spykes/getting-started.html
    https://xcorr.net/2011/10/03/using-the-binomial-glm-instead-of-the-poisson-for-spike-data/

    [1] http://www.stat.cmu.edu/~kass/papers/bars.pdf
    https://gist.github.com/AustinRochford/d640a240af12f6869a7b9b592485ca15
    https://discourse.pymc.io/t/bayesian-adaptive-regression-splines-and-mcmc-some-questions/756/5

    """

    def __init__(self, mode='hist', *args, **kwargs):
        self._mode = mode
        #TODO: check that mode is valid:

        # raise Exception if mode not supported or implemented yet
        pass

    def fit(self, X, y, lengths=None, sample_weight=None):
        """Fit Gaussian Naive Bayes according to X, y
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (n_samples,)
            Target values.
        sample_weight : array-like, shape (n_samples,), optional (default=None)
            Weights applied to individual samples (1. for unweighted).
            .. versionadded:: 0.17
               Gaussian Naive Bayes supports fitting with *sample_weight*.
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        return self._partial_fit(X, y, np.unique(y), _refit=True,
                                 sample_weight=sample_weight)

    def predict(self, X, lengths=None):
        raise NotImplementedError

    def predict_proba(self, X, lengths=None):
        raise NotImplementedError

    def score(self, X, y, lengths=None):
        raise NotImplementedError

    def score_samples(self, X, y, lengths=None):
        raise NotImplementedError


