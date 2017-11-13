"""Bayesian encoding and decoding"""

__all__ = ['decode1D',
           'decode2D',
           'k_fold_cross_validation',
           'cumulative_dist_decoding_error_using_xval',
           'cumulative_dist_decoding_error',
           'get_mode_pth_from_array',
           'get_mean_pth_from_array']

import numpy as np
from . import auxiliary

def get_mode_pth_from_array(posterior, tuningcurve=None):
    """If tuningcurve is provided, then we map it back to the external coordinates / units.
    Otherwise, we stay in the bin space."""
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

    _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    xbins = (bins + xmax/n_xbins)[:-1]

    mode_pth = np.argmax(posterior, axis=0)*xmax/n_xbins
    mode_pth = np.where(np.isnan(posterior.sum(axis=0)), np.nan, mode_pth)

    return mode_pth

def get_mean_pth_from_array(posterior, tuningcurve=None):
    """If tuningcurve is provided, then we map it back to the external coordinates / units.
    Otherwise, we stay in the bin space."""
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

    _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    xbins = (bins + xmax/n_xbins)[:-1]

    mean_pth = (xbins * posterior.T).sum(axis=1)

    return mean_pth

def decode1D(bst, ratemap, xmin=0, xmax=100, w=1, nospk_prior=None, _skip_empty_bins=True):
    """Decodes binned spike trains using a ratemap with shape (n_units, n_ext)

    TODO: complete docstring
    TODO: what if we have higher dimensional external correlates? This
    function assumes a 1D correlate. Even if we linearize a 2D
    environment, for example, then mean_pth decoding no longer works as
    expected, so this function should probably be refactored.

    Parameters
    ----------
    bst :
    ratemap: array_like
        Firing rate map with shape (n_units, n_ext), where n_ext is the
        number of external correlates, e.g., position bins. The rate map
        is in spks/second.
    xmin : float
    xmax : float
    w : int
    nospk_prior : array_like
        Prior distribution over external correlates with shape (n_ext,)
        that will be used if no spikes are observed in a decoding window
        Default is np.nan.
        If nospk_prior is any scalar, then a uniform prior is assumed.

    _skip_empty_bins is only used to return the posterior regardless of
    whether any spikes were observed, so that we can understand the spatial
    distribution in the absence of spikes, or at low firing rates.

    Returns
    -------
    posteriors : array
        Posterior distribution with shape (n_ext, n_posterior_bins),
        where n_posterior bins <= bst.n_bins, but depends on w and the
        event lengths.
    cum_posterior_lengths : array

    mode_pth :

    mean_pth :

    Examples
    --------

    """

    if w is None:
        w=1
    assert float(w).is_integer(), "w must be a positive integer!"
    assert w > 0, "w must be a positive integer!"

    n_units, t_bins = bst.data.shape
    _, n_xbins = ratemap.shape

    # if we pass a TuningCurve1D object, extract the ratemap and re-order
    # units if necessary
    if isinstance(ratemap, auxiliary.TuningCurve1D):
        xmin = ratemap.bins[0]
        xmax = ratemap.bins[-1]
        bin_centers = ratemap.bin_centers
        # re-order units if necessary
        ratemap = ratemap.reorder_units_by_ids(bst.unit_ids)
        ratemap = ratemap.ratemap
    else:
        xmin = 0
        xmax = n_xbins
        bin_centers = np.arange(n_xbins)

    if nospk_prior is None:
        nospk_prior = np.full(n_xbins, np.nan)
    elif isinstance(nospk_priors, numbers.Number):
        nospk_prior = np.full(n_xbins, 1.0)

    assert nospk_prior.shape[0] == n_xbins, "prior must have length {}".format(n_xbins)
    assert nospk_prior.size == n_xbins, "prior must be a 1D array with length {}".format(n_xbins)

    lfx = np.log(ratemap)

    eterm = -ratemap.sum(axis=0)*bst.ds*w

    # if we decode using multiple bins at a time (w>1) then we have to decode each epoch separately:

    # first, we determine the number of bins we will decode. This requires us to scan over the epochs
    n_bins = 0
    cumlengths = np.cumsum(bst.lengths)
    posterior_lengths = np.zeros(bst.n_epochs, dtype=np.int)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        datalen = to_idx - prev_idx
        prev_idx = to_idx
        posterior_lengths[ii] = np.max((1,datalen - w + 1))

    n_bins = posterior_lengths.sum()
    posterior = np.zeros((n_xbins, n_bins))

    # next, we decode each epoch separately, one bin at a time
    cum_posterior_lengths = np.insert(np.cumsum(posterior_lengths),0,0)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        data = bst.data[:,prev_idx:to_idx]
        prev_idx = to_idx
        datacum = np.cumsum(data, axis=1) # ii'th data segment, with column of zeros prepended
        datacum = np.hstack((np.zeros((n_units,1)), datacum))
        re = w # right edge ptr
        # TODO: check if datalen < w and act appropriately
        if posterior_lengths[ii] > 1: # more than one full window fits into data length
            for tt in range(posterior_lengths[ii]):
                obs = datacum[:, re] - datacum[:, re-w] # spikes in window of size w
                re+=1
                post_idx = cum_posterior_lengths[ii] + tt
                if obs.sum() == 0 and _skip_empty_bins:
                    # no spikes to decode in window!
                    posterior[:,post_idx] = nospk_prior
                else:
                    posterior[:,post_idx] = (np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx).sum(axis=0) + eterm
        else: # only one window can fit in, and perhaps only partially. We just take all the data we can get,
              # and ignore the scaling problem where the window size is now possibly less than bst.ds*w
            post_idx = cum_posterior_lengths[ii]
            obs = datacum[:, -1] # spikes in window of size at most w
            if obs.sum() == 0 and _skip_empty_bins:
                # no spikes to decode in window!
                posterior[:,post_idx] = nospk_prior
            else:
                posterior[:,post_idx] = (np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx).sum(axis=0) + eterm

    # normalize posterior:
    posterior = np.exp(posterior) / np.tile(np.exp(posterior).sum(axis=0),(n_xbins,1))

    # TODO: what was my rationale behid the following? Why not use bin centers?
    # _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    # xbins = (bins + xmax/n_xbins)[:-1]

    mode_pth = np.argmax(posterior, axis=0)*xmax/n_xbins
    mode_pth = np.where(np.isnan(posterior.sum(axis=0)), np.nan, mode_pth)
    mean_pth = (bin_centers * posterior.T).sum(axis=1)
    return posterior, cum_posterior_lengths, mode_pth, mean_pth

def decode2D(bst, ratemap, xmin=0, xmax=100, ymin=0, ymax=100, w=1, nospk_prior=None, _skip_empty_bins=True):
    """Decodes binned spike trains using a ratemap with shape (n_units, ext_nx, ext_ny)

    TODO: complete docstring
    TODO: what if we have higher dimensional external correlates? This
    function assumes a 2D correlate. Even if we linearize a 2D
    environment, for example, then mean_pth decoding no longer works as
    expected, so this function should probably be refactored.

    Parameters
    ----------
    bst :
    ratemap: array_like
        Firing rate map with shape (n_units, ext_nx, ext_ny), where n_ext is the
        number of external correlates, e.g., position bins. The rate map
        is in spks/second.
    xmin : float
    xmax : float
    w : int
    nospk_prior : array_like
        Prior distribution over external correlates with shape (n_ext,)
        that will be used if no spikes are observed in a decoding window
        Default is np.nan.
        If nospk_prior is any scalar, then a uniform prior is assumed.

    _skip_empty_bins is only used to return the posterior regardless of
    whether any spikes were observed, so that we can understand the spatial
    distribution in the absence of spikes, or at low firing rates.

    Returns
    -------
    posteriors : array
        Posterior distribution with shape (ext_nx, ext_ny, n_posterior_bins),
        where n_posterior bins <= bst.n_tbins, but depends on w and the
        event lengths.
    cum_posterior_lengths : array

    mode_pth :

    mean_pth :

    Examples
    --------

    """

    def tile_obs(obs, nx, ny):
        n_units = len(obs)
        out = np.zeros((n_units, nx, ny))
        for unit in range(n_units):
            out[unit,:,:] = obs[unit]
        return out

    if w is None:
        w=1
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
    elif isinstance(nospk_priors, numbers.Number):
        nospk_prior = np.full((n_xbins, n_ybins), 1.0)

    assert nospk_prior.shape == (n_xbins, n_ybins), "prior must have shape ({}, {})".format(n_xbins, n_ybins)

    lfx = np.log(ratemap)

    eterm = -ratemap.sum(axis=0)*bst.ds*w

    # if we decode using multiple bins at a time (w>1) then we have to decode each epoch separately:

    # first, we determine the number of bins we will decode. This requires us to scan over the epochs
    n_tbins = 0
    cumlengths = np.cumsum(bst.lengths)
    posterior_lengths = np.zeros(bst.n_epochs, dtype=np.int)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        datalen = to_idx - prev_idx
        prev_idx = to_idx
        posterior_lengths[ii] = np.max((1,datalen - w + 1))

    n_tbins = posterior_lengths.sum()

    ########################################################################
    posterior = np.zeros((n_xbins, n_ybins, n_tbins))

    # next, we decode each epoch separately, one bin at a time
    cum_posterior_lengths = np.insert(np.cumsum(posterior_lengths),0,0)
    prev_idx = 0
    for ii, to_idx in enumerate(cumlengths):
        data = bst.data[:,prev_idx:to_idx]
        prev_idx = to_idx
        datacum = np.cumsum(data, axis=1) # ii'th data segment, with column of zeros prepended
        datacum = np.hstack((np.zeros((n_units,1)), datacum))
        re = w # right edge ptr
        # TODO: check if datalen < w and act appropriately
        if posterior_lengths[ii] > 1: # more than one full window fits into data length
            for tt in range(posterior_lengths[ii]):
                obs = datacum[:, re] - datacum[:, re-w] # spikes in window of size w
                re+=1
                post_idx = cum_posterior_lengths[ii] + tt
                if obs.sum() == 0 and _skip_empty_bins:
                    # no spikes to decode in window!
                    posterior[:,:,post_idx] = nospk_prior
                else:
                    posterior[:,:,post_idx] = (tile_obs(obs, n_xbins, n_ybins) * lfx).sum(axis=0) + eterm
        else: # only one window can fit in, and perhaps only partially. We just take all the data we can get,
              # and ignore the scaling problem where the window size is now possibly less than bst.ds*w
            post_idx = cum_posterior_lengths[ii]
            obs = datacum[:, -1] # spikes in window of size at most w
            if obs.sum() == 0 and _skip_empty_bins:
                # no spikes to decode in window!
                posterior[:,:,post_idx] = nospk_prior
            else:
                posterior[:,:,post_idx] = (tile_obs(obs, n_xbins, n_ybins) * lfx).sum(axis=0) + eterm

    # normalize posterior:
    # see http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    for tt in range(n_tbins):
        posterior[:,:,tt] = posterior[:,:,tt] - posterior[:,:,tt].max()
        posterior[:,:,tt] = np.exp(posterior[:,:,tt])
        posterior[:,:,tt] = posterior[:,:,tt] / posterior[:,:,tt].sum()

    # if xbins is None:
    #     _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    #     xbins = (bins + xmax/n_xbins)[:-1]
    # if ybins is None:
    #     _, bins = np.histogram([], bins=n_ybins, range=(ymin,ymax))
    #     ybins = (bins + ymax/n_ybins)[:-1]

    mode_pth = np.zeros((2, n_tbins))
    for tt in range(n_tbins):
        if np.any(np.isnan(posterior[:,:,tt])):
            mode_pth[0,tt] = np.nan
            mode_pth[0,tt] = np.nan
        else:
            x_, y_ = np.unravel_index(np.argmax(posterior[:,:,tt]), (n_xbins, n_ybins))
            mode_pth[0,tt] = xbins[x_]
            mode_pth[1,tt] = ybins[y_]

    expected_x = (xbin_centers * posterior.sum(axis=1).T).sum(axis=1)
    expected_y = (ybin_centers * posterior.sum(axis=0).T).sum(axis=1)
    mean_pth = np.vstack((expected_x, expected_y))

    posterior = np.transpose(posterior, axes=[1,0,2])

    return posterior, cum_posterior_lengths, mode_pth, mean_pth

def k_fold_cross_validation(X, k=None, randomize=False):
    """
    Generates K (training, validation) pairs from the items in X.

    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length
    (K-1)*len(X)/K.

    Parameters
    ----------
    X : list or int
        list of items, or list of indices, or integer number of indices
    k : int, or str, optional
        k > 1 number of folds for k-fold cross validation; k='loo' or
        'LOO' for leave-one-out cross-validation (equivalent to
        k==n_samples). Default is 5.
    randomize : bool
         If true, a copy of X is shuffled before partitioning, otherwise
         its order is preserved in training and validation.

    Returns
    -------
    (training, validation)

    Example
    -------
    >>> X = [i for i in range(97)]
    >>> for training, validation in k_fold_cross_validation(X, k=5):
    >>>     print(training, validation)
    >>>     for x in X: assert (x in training) ^ (x in validation), x

    """
    # deal with default values:
    if isinstance(X, int):
        X = range(X)
    n_samples = len(X)
    if k is None:
        k=5
    elif k=='loo' or k=='LOO':
        k=n_samples

    if randomize:
        from random import shuffle
        X=list(X)
        shuffle(X)
    for _k_ in range(k):
        training = [x for i, x in enumerate(X) if i % k != _k_]
        validation = [x for i, x in enumerate(X) if i % k == _k_]
        yield training, validation

def cumulative_dist_decoding_error_using_xval(bst, extern,*, decodefunc=decode1D, tuningcurve=None, k=5, transfunc=None, n_extern=100, extmin=0, extmax=100, sigma=3, n_bins=None):
    """Cumulative distribution of decoding errors during epochs in
    BinnedSpikeTrainArray, evaluated using a k-fold cross-validation
    procedure.

    Parameters
    ----------
    bst: BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the epochs to be decoded.
        Should typically have the same type of epochs as the ratemap
        (e.g., online epochs), but this is not a requirement.
    tuningcurve : TuningCurve1D
    extern : query-able object of external correlates (e.g. pos AnalogSignalArray)
    ratemap : array_like
        The ratemap (in Hz) with shape (n_units, n_ext) where n_ext are
        the external correlates, e.g., position bins.
    k : int, optional
        Number of fold for k-fold cross-validation. Default is k=5.
    n_bins : int
        Number of decoding error bins, ranging from tuningcurve.extmin
        to tuningcurve.extmax.

    Returns
    -------

    (error, cum_prob)
        (see Fig 3.(b) of "Analysis of Hippocampal Memory Replay Using
        Neural Population Decoding", Fabian Kloosterman, 2012)

    NOTE: should we allow for an optional tuning curve to be specified,
          or should we always recompute it ourselves?
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
    for training, validation in k_fold_cross_validation(bst.n_epochs, k=k):
        # estimate place fields using bst[training]
        tc = auxiliary.TuningCurve1D(bst=bst[training], extern=extern, n_extern=n_extern, extmin=extmin, extmax=extmax, sigma=sigma)
        # decode position using bst[validation]
        posterior, _, mode_pth, mean_pth = decodefunc(bst[validation], tc)
        # calculate validation error (for current fold) by comapring
        # decoded pos v target pos
        target = transfunc(extern, at=bst[validation].bin_centers)

        histnew, bins = np.histogram(np.abs(target - mean_pth), bins=n_bins, range=(0, max_error))
        hist = hist + histnew

    # build cumulative error distribution
    cumhist = np.cumsum(hist)
    cumhist = cumhist / cumhist[-1]
    bincenters = (bins + (bins[1] - bins[0])/2)[:-1]

    # modify to start at (0,0):
    cumhist = np.insert(cumhist, 0, 0)
    bincenters = np.insert(bincenters, 0, 0)

    # modify to end at (max_error,1):
    cumhist = np.append(cumhist, 1)
    bincenters = np.append(bincenters, max_error)

    return cumhist, bincenters

def cumulative_dist_decoding_error(bst, *, tuningcurve, extern,
                                   decodefunc=decode1D, transfunc=None,
                                   n_bins=None):
    """Cumulative distribution of decoding errors during epochs in
    BinnedSpikeTrainArray using a fixed TuningCurve.

    Parameters
    ----------
    bst: BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the epochs to be decoded.
        Should typically have the same type of epochs as the ratemap
        (e.g., online epochs), but this is not a requirement.
    tuningcurve : TuningCurve1D
    extern : query-able object of external correlates (e.g. pos AnalogSignalArray)
    n_bins : int
        Number of decoding error bins, ranging from tuningcurve.extmin
        to tuningcurve.extmax.

    Returns
    -------

    (cumhist, bincenters)
        (see Fig 3.(b) of "Analysis of Hippocampal Memory Replay Using
        Neural Population Decoding", Fabian Kloosterman, 2012)

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
        np.abs(target - mean_pth),
        bins=n_bins,
        range=(0, max_error))

    # build cumulative error distribution
    cumhist = np.cumsum(hist)
    cumhist = cumhist / cumhist[-1]
    bincenters = (bins + (bins[1] - bins[0])/2)[:-1]

    # modify to start at (0,0):
    cumhist = np.insert(cumhist, 0, 0)
    bincenters = np.insert(bincenters, 0, 0)

    # modify to end at (max_error,1):
    cumhist = np.append(cumhist, 1)
    bincenters = np.append(bincenters, max_error)

    return cumhist, bincenters

def rmse(predictions, targets):
    """Calculate the root mean squared error of an array of predictions.

    Parameters
    ----------
    predictions : array_like
        Array of predicted values.
    targets : array_like
        Array of target values.

    Returns
    -------
    rmse: float
        Root mean squared error of the predictions wrt the targets.
    """
    predictions = np.asanyarray(predictions)
    targets = np.asanyarray(targets)
    rmse = np.sqrt(np.nanmean((predictions - targets) ** 2))
    return rmse