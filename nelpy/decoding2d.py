import numpy as np
from . import auxiliary

def decode2d(bst, ratemap, xmin=0, xmax=100, w=1, nospk_prior=None, _skip_empty_bins=True):
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

    # if we pass a TuningCurve2D object, extract the ratemap and re-order
    # units if necessary
    if isinstance(ratemap, auxiliary.TuningCurve2D):
        xmin = ratemap.xbins[0]
        xmax = ratemap.xbins[-1]
        ymin = ratemap.ybins[0]
        ymax = ratemap.ybins[-1]
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

    ########################################################################

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
                    ########################################################################
                    ########################################################################
                    ########################################################################
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

    ########################################################################

    # normalize posterior:
    posterior = np.exp(posterior)
    for tt in range(n_tbins):
        posterior[:,:,tt] = posterior[:,:,tt] / posterior[:,:,tt].sum()

    # _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax))
    # xbins = (bins + xmax/n_xbins)[:-1]

    # mode_pth = np.argmax(posterior, axis=0)*xmax/n_xbins
    # mode_pth = np.where(np.isnan(posterior.sum(axis=0)), np.nan, mode_pth)
    # mean_pth = (xbins * posterior.T).sum(axis=1)
    # return posterior, cum_posterior_lengths, mode_pth, mean_pth

    return posterior