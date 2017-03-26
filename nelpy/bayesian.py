"""Bayesian encoding and decoding"""

def decode1D(bst, ratemap, xmin=0, xmax=70, w=1):
    """Decodes binned spike trains using a ratemap with shape (n_units, n_ext)

    TODO: complete docstring
    TODO: what if we have higher dimensional external correlates? This function
    assumes a 1D correlate. Even if we linearize a 2D environment, for example,
    then mean_pth decoding no longer works as expected, so this function should
    probably be refactored.

    Parameters
    ----------
    bst :
    ratemap: array_like
        Firing rate map with shape (n_units, n_ext), where n_ext is the number of
        external correlates, e.g., position bins. The rate map is in spks/second.
    xmin : float
    xmax : float
    w : int

    Returns
    -------
    posteriors : array
        Posterior distribution with shape (n_ext, n_posterior_bins), where
        n_posterior bins <= bst.n_bins, but depends on w and the event lengths.
    cum_posterior_lengths : array

    mode_pth :

    mean_pth :
    """

    if w is None:
        w=1
    assert float(w).is_integer(), "w must be a positive integer!"
    assert w > 0, "w bust be a positive integer!"

    n_units, t_bins = bst.data.shape
    _, n_xbins = ratemap.shape


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
                posterior[:,post_idx] = (np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx).sum(axis=0) + eterm
        else: # only one window can fit in, and perhaps only partially. We just take all the data we can get,
              # and ignore the scaling problem where the window size is now possibly less than bst.ds*w
            post_idx = cum_posterior_lengths[ii]
            obs = datacum[:, -1] # spikes in window of size at most w
            posterior[:,post_idx] = (np.tile(np.array(obs, ndmin=2).T, n_xbins) * lfx).sum(axis=0) + eterm

    # normalize posterior:
    posterior = np.exp(posterior) / np.tile(np.exp(posterior).sum(axis=0),(n_xbins,1))

    _, bins = np.histogram([], bins=n_xbins, range=(xmin,xmax));
    xbins = (bins + xmax/n_xbins)[:-1]

    mode_pth = posterior.argmax(axis=0)*xmax/n_xbins
    mean_pth = (xbins * posterior.T).sum(axis=1)
    return posterior, cum_posterior_lengths, mode_pth, mean_pth