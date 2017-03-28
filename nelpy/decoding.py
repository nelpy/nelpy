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

def cumulative_dist_decoding_error_using_xval(bst, ratemap, k=5):
    """Cumulative distribution of decoding errors during epochs in
    BinnedSpikeTrainArray, evaluated using a k-fold cross-validation
    procedure.

    Parameters
    ----------
    bst: BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the epochs to be decoded.
        Should typically have the same type of epochs as the ratemap
        (e.g., online epochs), but this is not a requirement.
    ratemap : array_like
        The ratemap (in Hz) with shape (n_units, n_ext) where n_ext are
        the external correlates, e.g., position bins.
    k : int, optional
        Number of fold for k-fold cross-validation. Default is k=5.


    Returns
    -------

    (error, cum_prob)
        (see Fig 3.(b) of "Analysis of Hippocampal Memory Replay Using
        Neural Population Decoding", Fabian Kloosterman, 2012)

    """

    # NOTE! Indexing into BinnedSpikeTrainArrays is not yet supported,
    # but will be supported very soon, so the code below can be assumed
    # to work...

    # indices of training and validation epochs / events

    for training, validation in k_fold_cross_validation(bst.n_epochs, k=5):
        # indexing directly into BinnedSpikeTrainArray:
        print(bst[training], bst[validation])
        # indexing BinnedSpikeTrainArray using EpochArrays:
        print(bst[bst.support[training]], bst[bst.support[validation]])
        # estimate place fields using bst[training]
        # decode position using bst[validation]
        # calculate validation error (for current fold) by comapring
        # decoded pos v pos[bst[validation].support] or pos[bst.support][validation]

def plot_cum_dist_decoding_error(error, cumprob, *, ax=None, lw=None, **kwargs):
    """Plots the cumulative distribution of decoding errors.

    See Fig 3.(b) of "Analysis of Hippocampal Memory Replay Using Neural
        Population Decoding", Fabian Kloosterman, 2012.

    Parameters
    ----------

    Returns
    -------

    """

    if ax is None:
        ax = plt.gca()
    if lw is None:
        lw=1.5

    ax.plt(error, cumprob, lw=lw, **kwargs)
    ax.set_ylim(0,1)
    ax.set_ylabel('cumulative probability')
    ax.set_xlabel('error (cm)')

    # TODO: optionally plot the inset, with 0.5 and 0.7 marked

    return ax
