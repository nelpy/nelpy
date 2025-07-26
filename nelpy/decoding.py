"""Bayesian encoding and decoding"""

__all__ = [
    "decode1D",
    "decode2D",
    "k_fold_cross_validation",
    "cumulative_dist_decoding_error_using_xval",
    "cumulative_dist_decoding_error",
    "get_mode_pth_from_array",
    "get_mean_pth_from_array",
]

import copy
import numbers

import numpy as np
from scipy import interpolate
from scipy.special import logsumexp

from . import auxiliary


class ItemGetter_loc(object):
    """
    .loc is primarily label based (that is, series_id based).

    Allows label-based selection of intervals and series in event arrays.
    Raises KeyError when the items are not found.

    Allowed inputs are:
        - A single label, e.g. 5 or 'a' (interpreted as a label, not a position)
        - A list or array of labels ['a', 'b', 'c']
        - A slice object with labels 'a':'f' (both start and stop are included)

    Parameters
    ----------
    obj : object
        The parent object to slice.
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, idx):
        """intervals, series"""
        intervalslice, seriesslice = self.obj._slicer[idx]

        # first convert series slice into list
        if isinstance(seriesslice, slice):
            start = seriesslice.start
            stop = seriesslice.stop
            istep = seriesslice.step
            try:
                if start is None:
                    istart = 0
                else:
                    istart = self.obj._series_ids.index(start)
            except ValueError:
                raise KeyError(
                    "series_id {} could not be found in BaseEventArray!".format(start)
                )
            try:
                if stop is None:
                    istop = self.obj.n_series
                else:
                    istop = self.obj._series_ids.index(stop) + 1
            except ValueError:
                raise KeyError(
                    "series_id {} could not be found in BaseEventArray!".format(stop)
                )
            if istep is None:
                istep = 1
            if istep < 0:
                istop -= 1
                istart -= 1
                istart, istop = istop, istart
            series_idx_list = list(range(istart, istop, istep))
        else:
            series_idx_list = []
            seriesslice = np.atleast_1d(seriesslice)
            for series in seriesslice:
                try:
                    uidx = self.obj.series_ids.index(series)
                except ValueError:
                    raise KeyError(
                        "series_id {} could not be found in BaseEventArray!".format(
                            series
                        )
                    )
                else:
                    series_idx_list.append(uidx)

        if not isinstance(series_idx_list, list):
            series_idx_list = list(series_idx_list)
        out = copy.copy(self.obj)
        try:
            out._data = out._data[series_idx_list]
            singleseries = len(out._data) == 1
        except AttributeError:
            out._data = out._data[series_idx_list]
            singleseries = len(out._data) == 1

        if singleseries:
            out._data = np.array(out._data[0], ndmin=2)
        out._series_ids = list(
            np.atleast_1d(np.atleast_1d(out._series_ids)[series_idx_list])
        )
        out._series_labels = list(
            np.atleast_1d(np.atleast_1d(out._series_labels)[series_idx_list])
        )
        # TODO: update tags
        if isinstance(intervalslice, slice):
            if (
                intervalslice.start is None
                and intervalslice.stop is None
                and intervalslice.step is None
            ):
                out.loc = ItemGetter_loc(out)
                out.iloc = ItemGetter_iloc(out)
                return out
        out = out._intervalslicer(intervalslice)
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out


class ItemGetter_iloc(object):
    """
    .iloc is primarily integer position based (from 0 to length-1 of the axis).

    Allows integer-based selection of intervals and series in event arrays.
    Raises IndexError if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing (conforms with python/numpy slice semantics).

    Allowed inputs are:
        - An integer e.g. 5
        - A list or array of integers [4, 3, 0]
        - A slice object with ints 1:7

    Parameters
    ----------
    obj : object
        The parent object to slice.
    """

    def __init__(self, obj):
        self.obj = obj

    def __getitem__(self, idx):
        """intervals, series"""
        intervalslice, seriesslice = self.obj._slicer[idx]
        out = copy.copy(self.obj)
        if isinstance(seriesslice, int):
            seriesslice = [seriesslice]
        out._data = out._data[seriesslice]
        singleseries = len(out._data) == 1
        if singleseries:
            out._data = np.array(out._data[0], ndmin=2)
        out._series_ids = list(
            np.atleast_1d(np.atleast_1d(out._series_ids)[seriesslice])
        )
        out._series_labels = list(
            np.atleast_1d(np.atleast_1d(out._series_labels)[seriesslice])
        )
        # TODO: update tags
        if isinstance(intervalslice, slice):
            if (
                intervalslice.start is None
                and intervalslice.stop is None
                and intervalslice.step is None
            ):
                out.loc = ItemGetter_loc(out)
                out.iloc = ItemGetter_iloc(out)
                return out
        out = out._intervalslicer(intervalslice)
        out.loc = ItemGetter_loc(out)
        out.iloc = ItemGetter_iloc(out)
        return out


def get_mode_pth_from_array(posterior, tuningcurve=None):
    """
    Compute the mode path (most likely position) from a posterior probability matrix.

    If a tuning curve is provided, the mode is mapped back to external coordinates/units.
    Otherwise, the mode is in bin space.

    Parameters
    ----------
    posterior : np.ndarray
        Posterior probability matrix (position x time).
    tuningcurve : TuningCurve1D, optional
        Tuning curve for mapping bins to external coordinates.

    Returns
    -------
    mode_pth : np.ndarray
        Most likely position at each time bin.
    """
    n_xbins = posterior.shape[0]

    if tuningcurve is None:
        xmin = 0
        xmax = n_xbins
    else:
        # TODO: this only works for TuningCurve1D currently
        if isinstance(tuningcurve, auxiliary.TuningCurve1D):
            xmin = tuningcurve.bins[0]
            xmax = tuningcurve.bins[-1]
        else:
            raise TypeError("tuningcurve type not yet supported!")

    _, bins = np.histogram([], bins=n_xbins, range=(xmin, xmax))
    # xbins = (bins + xmax / n_xbins)[:-1]

    mode_pth = np.argmax(posterior, axis=0) * xmax / n_xbins
    mode_pth = np.where(np.isnan(posterior.sum(axis=0)), np.nan, mode_pth)

    return mode_pth


def get_mean_pth_from_array(posterior, tuningcurve=None):
    """
    Compute the mean path (expected position) from a posterior probability matrix.

    If a tuning curve is provided, the mean is mapped back to external coordinates/units.
    Otherwise, the mean is in bin space.

    Parameters
    ----------
    posterior : np.ndarray
        Posterior probability matrix (position x time).
    tuningcurve : TuningCurve1D, optional
        Tuning curve for mapping bins to external coordinates.

    Returns
    -------
    mean_pth : np.ndarray
        Expected position at each time bin.
    """
    n_xbins = posterior.shape[0]

    if tuningcurve is None:
        xmin = 0
        xmax = 1
    else:
        # TODO: this only works for TuningCurve1D currently
        if isinstance(tuningcurve, auxiliary.TuningCurve1D):
            xmin = tuningcurve.bins[0]
            xmax = tuningcurve.bins[-1]
        else:
            raise TypeError("tuningcurve type not yet supported!")

    _, bins = np.histogram([], bins=n_xbins, range=(xmin, xmax))
    xbins = (bins + xmax / n_xbins)[:-1]

    mean_pth = (xbins * posterior.T).sum(axis=1)

    return mean_pth


def decode1D(
    bst, ratemap, xmin=0, xmax=100, w=1, nospk_prior=None, _skip_empty_bins=True
):
    """
    Decode binned spike trains using a 1D ratemap (Bayesian decoding).

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        Binned spike train array to decode.
    ratemap : array_like or TuningCurve1D
        Firing rate map with shape (n_units, n_ext), where n_ext is the number of external correlates (e.g., position bins). The rate map is in spks/second.
    xmin : float, optional
        Minimum value of external variable (default is 0).
    xmax : float, optional
        Maximum value of external variable (default is 100).
    w : int, optional
        Window size for decoding (default is 1).
    nospk_prior : array_like or float, optional
        Prior distribution over external correlates with shape (n_ext,). Used if no spikes are observed in a decoding window. If scalar, a uniform prior is assumed. Default is np.nan.
    _skip_empty_bins : bool, optional
        If True, skip bins with no spikes. If False, fill with prior.

    Returns
    -------
    posterior : np.ndarray
        Posterior distribution with shape (n_ext, n_posterior_bins).
    cum_posterior_lengths : np.ndarray
        Cumulative posterior lengths for each epoch.
    mode_pth : np.ndarray
        Most likely position at each time bin.
    mean_pth : np.ndarray
        Expected position at each time bin.

    Examples
    --------
    >>> posterior, cum_posterior_lengths, mode_pth, mean_pth = decode1D(bst, ratemap)
    """

    if w is None:
        w = 1
    assert float(w).is_integer(), "w must be a positive integer!"
    assert w > 0, "w must be a positive integer!"

    n_units, t_bins = bst.data.shape
    _, n_xbins = ratemap.shape

    # if we pass a TuningCurve1D object, extract the ratemap and re-order
    # units if necessary
    if isinstance(ratemap, auxiliary.TuningCurve1D) | isinstance(
        ratemap, auxiliary._tuningcurve.TuningCurve1D
    ):
        # xmin = ratemap.bins[0]
        xmax = ratemap.bins[-1]
        bin_centers = ratemap.bin_centers
        # re-order units if necessary
        ratemap = ratemap.reorder_units_by_ids(bst.unit_ids)
        ratemap = ratemap.ratemap
    else:
        # xmin = 0
        xmax = n_xbins
        bin_centers = np.arange(n_xbins)

    if nospk_prior is None:
        nospk_prior = np.full(n_xbins, np.nan)
    elif isinstance(nospk_prior, numbers.Number):
        nospk_prior = np.full(n_xbins, 1.0)

    assert nospk_prior.shape[0] == n_xbins, "prior must have length {}".format(n_xbins)
    assert nospk_prior.size == n_xbins, (
        "prior must be a 1D array with length {}".format(n_xbins)
    )

    lfx = np.log(ratemap)

    eterm = -ratemap.sum(axis=0) * bst.ds * w

    # if we decode using multiple bins at a time (w>1) then we have to decode each epoch separately:

    # first, we determine the number of bins we will decode. This requires us to scan over the epochs
    n_bins = 0
    cumlengths = np.cumsum(bst.lengths)
    posterior_lengths = np.zeros(bst.n_epochs, dtype=int)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        datalen = to_idx - prev_idx
        prev_idx = to_idx
        posterior_lengths[ii] = np.max((1, datalen - w + 1))

    n_bins = posterior_lengths.sum()
    posterior = np.zeros((n_xbins, n_bins))

    # next, we decode each epoch separately, one bin at a time
    cum_posterior_lengths = np.insert(np.cumsum(posterior_lengths), 0, 0)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        data = bst.data[:, prev_idx:to_idx]
        prev_idx = to_idx
        datacum = np.cumsum(
            data, axis=1
        )  # ii'th data segment, with column of zeros prepended
        datacum = np.hstack((np.zeros((n_units, 1)), datacum))
        re = w  # right edge ptr
        # TODO: check if datalen < w and act appropriately
        if posterior_lengths[ii] > 1:  # more than one full window fits into data length
            for tt in range(posterior_lengths[ii]):
                obs = datacum[:, re] - datacum[:, re - w]  # spikes in window of size w
                re += 1
                post_idx = cum_posterior_lengths[ii] + tt
                if obs.sum() == 0 and _skip_empty_bins:
                    # no spikes to decode in window!
                    posterior[:, post_idx] = nospk_prior
                else:
                    posterior[:, post_idx] = (
                        np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx
                    ).sum(axis=0) + eterm
        else:  # only one window can fit in, and perhaps only partially. We just take all the data we can get,
            # and ignore the scaling problem where the window size is now possibly less than bst.ds*w
            post_idx = cum_posterior_lengths[ii]
            obs = datacum[:, -1]  # spikes in window of size at most w
            if obs.sum() == 0 and _skip_empty_bins:
                # no spikes to decode in window!
                posterior[:, post_idx] = nospk_prior
            else:
                posterior[:, post_idx] = (
                    np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx
                ).sum(axis=0) + eterm

    # normalize posterior:
    posterior = np.exp(posterior - logsumexp(posterior, axis=0))

    # TODO: what was my rationale behid the following? Why not use bin centers?
    # _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    # xbins = (bins + xmax/n_xbins)[:-1]

    mode_pth = np.argmax(posterior, axis=0) * xmax / n_xbins
    mode_pth = np.where(np.isnan(posterior.sum(axis=0)), np.nan, mode_pth)
    mean_pth = (bin_centers * posterior.T).sum(axis=1)
    return posterior, cum_posterior_lengths, mode_pth, mean_pth


def decode2D(
    bst,
    ratemap,
    xmin=0,
    xmax=100,
    ymin=0,
    ymax=100,
    w=1,
    nospk_prior=None,
    _skip_empty_bins=True,
):
    """
    Decode binned spike trains using a 2D ratemap (Bayesian decoding).

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        Binned spike train array to decode.
    ratemap : array_like or TuningCurve2D
        Firing rate map with shape (n_units, ext_nx, ext_ny), where ext_nx and ext_ny are the number of external correlates (e.g., position bins). The rate map is in spks/second.
    xmin : float, optional
        Minimum x value of external variable (default is 0).
    xmax : float, optional
        Maximum x value of external variable (default is 100).
    ymin : float, optional
        Minimum y value of external variable (default is 0).
    ymax : float, optional
        Maximum y value of external variable (default is 100).
    w : int, optional
        Window size for decoding (default is 1).
    nospk_prior : array_like or float, optional
        Prior distribution over external correlates with shape (ext_nx, ext_ny). Used if no spikes are observed in a decoding window. If scalar, a uniform prior is assumed. Default is np.nan.
    _skip_empty_bins : bool, optional
        If True, skip bins with no spikes. If False, fill with prior.

    Returns
    -------
    posterior : np.ndarray
        Posterior distribution with shape (ext_nx, ext_ny, n_posterior_bins).
    cum_posterior_lengths : np.ndarray
        Cumulative posterior lengths for each epoch.
    mode_pth : np.ndarray
        Most likely (x, y) position at each time bin.
    mean_pth : np.ndarray
        Expected (x, y) position at each time bin.

    Examples
    --------
    >>> posterior, cum_posterior_lengths, mode_pth, mean_pth = decode2D(bst, ratemap)
    """

    def tile_obs(obs, nx, ny):
        n_units = len(obs)
        out = np.zeros((n_units, nx, ny))
        for unit in range(n_units):
            out[unit, :, :] = obs[unit]
        return out

    if w is None:
        w = 1
    assert float(w).is_integer(), "w must be a positive integer!"
    assert w > 0, "w must be a positive integer!"

    n_units, t_bins = bst.data.shape

    xbins = None
    ybins = None

    # if we pass a TuningCurve2D object, extract the ratemap and re-order
    # units if necessary
    if isinstance(ratemap, auxiliary.TuningCurve2D):
        xbins = ratemap.xbins
        ybins = ratemap.ybins
        xbin_centers = ratemap.xbin_centers
        ybin_centers = ratemap.ybin_centers
        # re-order units if necessary
        ratemap = ratemap.reorder_units_by_ids(bst.unit_ids)
        ratemap = ratemap.ratemap

    _, n_xbins, n_ybins = ratemap.shape

    if nospk_prior is None:
        nospk_prior = np.full((n_xbins, n_ybins), np.nan)
    elif isinstance(nospk_prior, numbers.Number):
        nospk_prior = np.full((n_xbins, n_ybins), 1.0)

    assert nospk_prior.shape == (
        n_xbins,
        n_ybins,
    ), "prior must have shape ({}, {})".format(n_xbins, n_ybins)

    lfx = np.log(ratemap)

    eterm = -ratemap.sum(axis=0) * bst.ds * w

    # if we decode using multiple bins at a time (w>1) then we have to decode each epoch separately:

    # first, we determine the number of bins we will decode. This requires us to scan over the epochs
    n_tbins = 0
    cumlengths = np.cumsum(bst.lengths)
    posterior_lengths = np.zeros(bst.n_epochs, dtype=int)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        datalen = to_idx - prev_idx
        prev_idx = to_idx
        posterior_lengths[ii] = np.max((1, datalen - w + 1))

    n_tbins = posterior_lengths.sum()

    ########################################################################
    posterior = np.zeros((n_xbins, n_ybins, n_tbins))

    # next, we decode each epoch separately, one bin at a time
    cum_posterior_lengths = np.insert(np.cumsum(posterior_lengths), 0, 0)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        data = bst.data[:, prev_idx:to_idx]
        prev_idx = to_idx
        datacum = np.cumsum(
            data, axis=1
        )  # ii'th data segment, with column of zeros prepended
        datacum = np.hstack((np.zeros((n_units, 1)), datacum))
        re = w  # right edge ptr
        # TODO: check if datalen < w and act appropriately
        if posterior_lengths[ii] > 1:  # more than one full window fits into data length
            for tt in range(posterior_lengths[ii]):
                obs = datacum[:, re] - datacum[:, re - w]  # spikes in window of size w
                re += 1
                post_idx = cum_posterior_lengths[ii] + tt
                if obs.sum() == 0 and not _skip_empty_bins:
                    # no spikes to decode in window!
                    posterior[:, :, post_idx] = nospk_prior
                else:
                    posterior[:, :, post_idx] = (
                        tile_obs(obs, n_xbins, n_ybins) * lfx
                    ).sum(axis=0) + eterm
        else:  # only one window can fit in, and perhaps only partially. We just take all the data we can get,
            # and ignore the scaling problem where the window size is now possibly less than bst.ds*w
            post_idx = cum_posterior_lengths[ii]
            obs = datacum[:, -1]  # spikes in window of size at most w
            if obs.sum() == 0 and not _skip_empty_bins:
                # no spikes to decode in window!
                posterior[:, :, post_idx] = nospk_prior
            else:
                posterior[:, :, post_idx] = (tile_obs(obs, n_xbins, n_ybins) * lfx).sum(
                    axis=0
                ) + eterm

    # normalize posterior:
    # see http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

    for tt in range(n_tbins):
        posterior[:, :, tt] = posterior[:, :, tt] - posterior[:, :, tt].max()
        posterior[:, :, tt] = np.exp(posterior[:, :, tt])
        posterior[:, :, tt] = posterior[:, :, tt] / posterior[:, :, tt].sum()

    # if xbins is None:
    #     _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    #     xbins = (bins + xmax/n_xbins)[:-1]
    # if ybins is None:
    #     _, bins = np.histogram([], bins=n_ybins, range=(ymin,ymax))
    #     ybins = (bins + ymax/n_ybins)[:-1]

    mode_pth = np.zeros((2, n_tbins))
    for tt in range(n_tbins):
        if np.any(np.isnan(posterior[:, :, tt])):
            mode_pth[0, tt] = np.nan
            mode_pth[0, tt] = np.nan
        else:
            x_, y_ = np.unravel_index(
                np.argmax(posterior[:, :, tt]), (n_xbins, n_ybins)
            )
            mode_pth[0, tt] = xbins[x_]
            mode_pth[1, tt] = ybins[y_]

    expected_x = (xbin_centers * posterior.sum(axis=1).T).sum(axis=1)
    expected_y = (ybin_centers * posterior.sum(axis=0).T).sum(axis=1)
    mean_pth = np.vstack((expected_x, expected_y))

    posterior = np.transpose(posterior, axes=[1, 0, 2])

    return posterior, cum_posterior_lengths, mode_pth, mean_pth


def k_fold_cross_validation(X, k=None, randomize=False):
    """
    Generate K (training, validation) pairs from the items in X for cross-validation.

    Parameters
    ----------
    X : list or int
        List of items, list of indices, or integer number of indices.
    k : int or str, optional
        Number of folds for k-fold cross-validation. 'loo' or 'LOO' for leave-one-out. Default is 5.
    randomize : bool, optional
        If True, shuffle X before partitioning. Default is False.

    Yields
    ------
    training : list
        Training set indices.
    validation : list
        Validation set indices.

    Examples
    --------
    >>> X = [i for i in range(97)]
    >>> for training, validation in k_fold_cross_validation(X, k=5):
    ...     print(training, validation)
    """
    # deal with default values:
    if isinstance(X, int):
        X = range(X)
    n_samples = len(X)
    if k is None:
        k = 5
    elif k == "loo" or k == "LOO":
        k = n_samples

    if randomize:
        from random import shuffle

        X = list(X)
        shuffle(X)
    for _k_ in range(k):
        training = [x for i, x in enumerate(X) if i % k != _k_]
        validation = [x for i, x in enumerate(X) if i % k == _k_]
        try:
            yield training, validation
        except StopIteration:
            return


class Cumhist(np.ndarray):
    """
    Cumulative histogram with interpolation support.

    Parameters
    ----------
    cumhist : np.ndarray
        Cumulative histogram values.
    bincenters : np.ndarray
        Bin centers corresponding to the cumulative histogram.
    """

    def __new__(cls, cumhist, bincenters):
        obj = np.asarray(cumhist).view(cls)
        obj._bincenters = bincenters
        return obj

    def __call__(self, *val):
        f = interpolate.interp1d(
            x=self, y=self._bincenters, kind="linear", fill_value=np.nan
        )
        try:
            vals = f(*val).item()
        except AttributeError:
            vals = f(*val)

        return vals


def cumulative_dist_decoding_error_using_xval(
    bst,
    extern,
    *,
    decodefunc=decode1D,
    k=5,
    transfunc=None,
    n_extern=100,
    extmin=0,
    extmax=100,
    sigma=3,
    n_bins=None,
    randomize=False,
):
    """
    Compute the cumulative distribution of decoding errors using k-fold cross-validation.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        Binned spike train array containing all epochs to decode.
    extern : object
        Query-able object of external correlates (e.g., position AnalogSignalArray).
    decodefunc : callable, optional
        Decoding function to use (default is decode1D).
    k : int, optional
        Number of folds for cross-validation (default is 5).
    transfunc : callable, optional
        Function to transform external variable (default is None).
    n_extern : int, optional
        Number of external bins (default is 100).
    extmin : float, optional
        Minimum value of external variable (default is 0).
    extmax : float, optional
        Maximum value of external variable (default is 100).
    sigma : float, optional
        Smoothing parameter for tuning curve (default is 3).
    n_bins : int, optional
        Number of decoding error bins (default is 200).
    randomize : bool, optional
        If True, randomize the order of epochs (default is False).

    Returns
    -------
    cumhist : Cumhist
        Cumulative histogram of decoding errors.
    bincenters : np.ndarray
        Bin centers for the cumulative histogram.
    """

    def _trans_func(extern, at):
        """Default transform function to map extern into numerical bins"""

        _, ext = extern.asarray(at=at)

        return ext

    if transfunc is None:
        transfunc = _trans_func

    if n_bins is None:
        n_bins = 200

    max_error = extmax - extmin

    # indices of training and validation epochs / events

    hist = np.zeros(n_bins)
    for training, validation in k_fold_cross_validation(
        bst.n_epochs, k=k, randomize=randomize
    ):
        # estimate place fields using bst[training]
        tc = auxiliary.TuningCurve1D(
            bst=bst[training],
            extern=extern,
            n_extern=n_extern,
            extmin=extmin,
            extmax=extmax,
            sigma=sigma,
        )
        # decode position using bst[validation]
        posterior, _, mode_pth, mean_pth = decodefunc(bst[validation], tc)
        # calculate validation error (for current fold) by comapring
        # decoded pos v target pos
        target = transfunc(extern, at=bst[validation].bin_centers)

        histnew, bins = np.histogram(
            np.abs(target - mean_pth), bins=n_bins, range=(0, max_error)
        )
        hist = hist + histnew

    # build cumulative error distribution
    cumhist = np.cumsum(hist)
    cumhist = cumhist / cumhist[-1]
    bincenters = (bins + (bins[1] - bins[0]) / 2)[:-1]

    # modify to start at (0,0):
    cumhist = np.insert(cumhist, 0, 0)
    bincenters = np.insert(bincenters, 0, 0)

    # modify to end at (max_error,1):
    cumhist = np.append(cumhist, 1)
    bincenters = np.append(bincenters, max_error)

    cumhist = Cumhist(cumhist, bincenters)
    return cumhist, bincenters


def cumulative_dist_decoding_error(
    bst, *, tuningcurve, extern, decodefunc=decode1D, transfunc=None, n_bins=None
):
    """
    Compute the cumulative distribution of decoding errors using a fixed tuning curve.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        Binned spike train array containing all epochs to decode.
    tuningcurve : TuningCurve1D
        Tuning curve to use for decoding.
    extern : object
        Query-able object of external correlates (e.g., position AnalogSignalArray).
    decodefunc : callable, optional
        Decoding function to use (default is decode1D).
    transfunc : callable, optional
        Function to transform external variable (default is None).
    n_bins : int, optional
        Number of decoding error bins (default is 200).

    Returns
    -------
    cumhist : Cumhist
        Cumulative histogram of decoding errors.
    bincenters : np.ndarray
        Bin centers for the cumulative histogram.
    """

    def _trans_func(extern, at):
        """Default transform function to map extern into numerical bins"""

        _, ext = extern.asarray(at=at)

        return ext

    if transfunc is None:
        transfunc = _trans_func
    if n_bins is None:
        n_bins = 200

    # indices of training and validation epochs / events

    max_error = tuningcurve.bins[-1] - tuningcurve.bins[0]

    posterior, _, mode_pth, mean_pth = decodefunc(bst=bst, ratemap=tuningcurve)
    target = transfunc(extern, at=bst.bin_centers)
    hist, bins = np.histogram(
        np.abs(target - mean_pth), bins=n_bins, range=(0, max_error)
    )

    # build cumulative error distribution
    cumhist = np.cumsum(hist)
    cumhist = cumhist / cumhist[-1]
    bincenters = (bins + (bins[1] - bins[0]) / 2)[:-1]

    # modify to start at (0,0):
    cumhist = np.insert(cumhist, 0, 0)
    bincenters = np.insert(bincenters, 0, 0)

    # modify to end at (max_error,1):
    cumhist = np.append(cumhist, 1)
    bincenters = np.append(bincenters, max_error)

    cumhist = Cumhist(cumhist, bincenters)

    return cumhist, bincenters


def rmse(predictions, targets):
    """
    Calculate the root mean squared error (RMSE) between predictions and targets.

    Parameters
    ----------
    predictions : array_like
        Array of predicted values.
    targets : array_like
        Array of target values.

    Returns
    -------
    rmse : float
        Root mean squared error of the predictions with respect to the targets.
    """
    predictions = np.asanyarray(predictions)
    targets = np.asanyarray(targets)
    rmse = np.sqrt(np.nanmean((predictions - targets) ** 2))
    return rmse


class BayesianDecoder(object):
    """
    Bayesian decoder for neural population activity.

    This class provides a scikit-learn-like API for Bayesian decoding using tuning curves.
    Supports 1D and 2D decoding, and can be extended for more complex models.

    Parameters
    ----------
    tuningcurve : TuningCurve1D or TuningCurve2D, optional
        Tuning curve to use for decoding.
    """

    def __init__(self, tuningcurve=None):
        """
        Initialize the BayesianDecoder.

        Parameters
        ----------
        tuningcurve : TuningCurve1D or TuningCurve2D, optional
            Tuning curve to use for decoding.
        """
        self.tuningcurve = tuningcurve

    def fit(self, X, y=None):
        """
        Fit the decoder to data X. (Stores the tuning curve if provided.)

        Parameters
        ----------
        X : array-like or TuningCurve1D/2D
            Training data or tuning curve.
        y : Ignored
        """
        # If X is a tuning curve, store it
        self.tuningcurve = X
        return self

    def predict_proba(self, X, **kwargs):
        """
        Predict posterior probabilities for data X.

        Parameters
        ----------
        X : array-like or BinnedEventArray
            Data to decode.
        Returns
        -------
        posterior : np.ndarray
            Posterior probability matrix.
        """
        if self.tuningcurve is None:
            raise ValueError(
                "No tuning curve set. Call fit() or provide tuningcurve in constructor."
            )
        # Use decode1D or decode2D depending on tuning curve
        if hasattr(self.tuningcurve, "ratemap") and hasattr(self.tuningcurve, "bins"):
            # TuningCurve1D or TuningCurve2D object
            ratemap = self.tuningcurve.ratemap
        else:
            ratemap = self.tuningcurve
        # Try to infer 1D vs 2D
        if ratemap.ndim == 2:
            posterior, _, _, _ = decode1D(X, ratemap, **kwargs)
        elif ratemap.ndim == 3:
            posterior, _, _, _ = decode2D(X, ratemap, **kwargs)
        else:
            raise ValueError("Tuning curve must be 2D or 3D array.")
        return posterior

    def predict(self, X, **kwargs):
        """
        Predict external variable from data X (returns mode path).

        Parameters
        ----------
        X : array-like or BinnedEventArray
            Data to decode.
        Returns
        -------
        mode_pth : np.ndarray
            Most likely position at each time bin.
        """
        if self.tuningcurve is None:
            raise ValueError(
                "No tuning curve set. Call fit() or provide tuningcurve in constructor."
            )
        if hasattr(self.tuningcurve, "ratemap") and hasattr(self.tuningcurve, "bins"):
            ratemap = self.tuningcurve.ratemap
        else:
            ratemap = self.tuningcurve
        if ratemap.ndim == 2:
            _, _, mode_pth, _ = decode1D(X, ratemap, **kwargs)
        elif ratemap.ndim == 3:
            _, _, mode_pth, _ = decode2D(X, ratemap, **kwargs)
        else:
            raise ValueError("Tuning curve must be 2D or 3D array.")
        return mode_pth

    def predict_asa(self, X, **kwargs):
        """
        Predict analog signal array (mean path) from data X.

        Parameters
        ----------
        X : array-like or BinnedEventArray
            Data to decode.
        Returns
        -------
        asa : AnalogSignalArray or np.ndarray
            Mean path as AnalogSignalArray if possible, else array.
        """
        if self.tuningcurve is None:
            raise ValueError(
                "No tuning curve set. Call fit() or provide tuningcurve in constructor."
            )
        if hasattr(self.tuningcurve, "ratemap") and hasattr(self.tuningcurve, "bins"):
            ratemap = self.tuningcurve.ratemap
        else:
            ratemap = self.tuningcurve
        if ratemap.ndim == 2:
            _, _, _, mean_pth = decode1D(X, ratemap, **kwargs)
        elif ratemap.ndim == 3:
            _, _, _, mean_pth = decode2D(X, ratemap, **kwargs)
        else:
            raise ValueError("Tuning curve must be 2D or 3D array.")
        # Try to return as AnalogSignalArray if possible, with timestamps if available
        try:
            from .core import AnalogSignalArray

            abscissa_vals = None
            # Try to get bin centers from X if possible
            if hasattr(X, "bin_centers"):
                abscissa_vals = X.bin_centers
            elif hasattr(X, "abscissa_vals"):
                abscissa_vals = X.abscissa_vals
            return AnalogSignalArray(data=mean_pth, abscissa_vals=abscissa_vals)
        except Exception:
            return mean_pth

    def __repr__(self):
        s = "<BayesianDecoder: "
        if self.tuningcurve is not None:
            s += f"tuningcurve shape={getattr(self.tuningcurve, 'ratemap', self.tuningcurve).shape}"
        else:
            s += "no tuningcurve"
        s += ">"
        return s
