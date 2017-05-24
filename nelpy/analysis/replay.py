__all__ = ['linregress_ting',
           'linregress_array',
           'linregress_bst',
           'time_swap_array',
           'column_cycle_array',
           'trajectory_score_array',
           'trajectory_score_bst',
           'get_significant_events',
           'three_consecutive_bins_above_q',
           'score_hmm_time_resolved',
           'score_hmm_logprob_cumulative']

import warnings
import copy
import numpy as np

from scipy import stats
from .. import auxiliary
from ..decoding import decode1D as decode
from ..decoding import get_mode_pth_from_array, get_mean_pth_from_array

def linregress_ting(bst, tuningcurve, n_shuffles=250):
    """perform linear regression on all the events in bst, and return the R^2 values"""

    if float(n_shuffles).is_integer:
        n_shuffles = int(n_shuffles)
    else:
        raise ValueError("n_shuffles must be an integer!")

    posterior, bdries, mode_pth, mean_pth = decode(bst=bst, ratemap=tuningcurve)

#     bdries = np.insert(np.cumsum(bst.lengths), 0, 0)
    r2values = np.zeros(bst.n_epochs)
    r2values_shuffled = np.zeros((n_shuffles, bst.n_epochs))
    for idx in range(bst.n_epochs):
        y = mode_pth[bdries[idx]:bdries[idx+1]]
        x = np.arange(bdries[idx],bdries[idx+1], step=1)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        if len(y) > 0:
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
            r2values[idx] = rvalue**2
        else:
            r2values[idx] = np.nan #
        for ss in range(n_shuffles):
            if len(y) > 0:
                slope, intercept, rvalue, pvalue, stderr = stats.linregress(np.random.permutation(x), y)
                r2values_shuffled[ss, idx] = rvalue**2
            else:
                r2values_shuffled[ss, idx] = np.nan # event contained NO decoded activity... unlikely or even impossible with current code

#     sig_idx = np.argwhere(r2values[0,:] > np.percentile(r2values, q=q, axis=0))
#     np.argwhere(((R2[1:,:] >= R2[0,:]).sum(axis=0))/(R2.shape[0]-1)<0.05) # equivalent to above
    if n_shuffles > 0:
        return r2values, r2values_shuffled
    return r2values

def linregress_array(posterior):
    """perform linear regression on the posterior matrix, and return the slope, intercept, and R^2 value"""

    mode_pth = get_mode_pth_from_array(posterior)

    y = mode_pth
    x = np.arange(len(y))
    x = x[~np.isnan(y)]
    y = y[~np.isnan(y)]

    if len(y) > 0:
        slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
        return slope, intercept, rvalue**2
    else:
        return np.nan, np.nan, np.nan


def linregress_bst(bst, tuningcurve):
    """perform linear regression on all the events in bst, and return the slopes, intercepts, and R^2 values"""

    posterior, bdries, mode_pth, mean_pth = decode(bst=bst, ratemap=tuningcurve)

    slopes = np.zeros(bst.n_epochs)
    intercepts = np.zeros(bst.n_epochs)
    r2values = np.zeros(bst.n_epochs)
    for idx in range(bst.n_epochs):
        y = mode_pth[bdries[idx]:bdries[idx+1]]
        x = np.arange(bdries[idx],bdries[idx+1], step=1)
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]

        if len(y) > 0:
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(x, y)
            slopes[idx] = slope
            intercepts[idx] = intercept
            r2values[idx] = rvalue**2
        else:
            slopes[idx] = np.nan
            intercepts[idx] = np.nan
            r2values[idx] = np.nan #
#     if bst.n_epochs == 1:
#         return np.asscalar(slopes), np.asscalar(intercepts), np.asscalar(r2values)
    return slopes, intercepts, r2values

def time_swap_array(posterior):
    """Time swap.
    Note: it is often possible to simply shuffle the time bins, and not the actual data, for computational
    efficiency. Still, this function works as expected."""
    out = copy.copy(posterior)
    rows, cols = posterior.shape

    colidx = np.arange(cols)
    shuffle_cols = np.random.permutation(colidx)
    out = out[:,shuffle_cols]

    return out

def time_swap_bst(bst):
    """Time swap on BinnedSpikeTrainArray, swapping only within each epoch."""
    out = copy.copy(bst) # should this be deep?
    shuffled = np.arange(bst.n_bins)
    edges = np.insert(np.cumsum(bst.lengths),0,0)
    for ii in range(bst.n_epochs):
        segment = shuffled[edges[ii]:edges[ii+1]]
        shuffled[edges[ii]:edges[ii+1]] = np.random.permutation(segment)

    out._data = out._data[:,shuffled]

    return out

def column_cycle_array(posterior, amt=None):
    """Also called 'position cycle' by Kloosterman et al.
    If amt is an array of the same length as posterior, then
    cycle each column by the corresponding amount in amt.
    Otherwise, cycle each column by a random amount."""
    out = copy.copy(posterior)
    rows, cols = posterior.shape

    if amt is None:
        for col in range(cols):
            if np.isnan(np.sum(posterior[:,col])):
                continue
            else:
                out[:,col] = np.roll(posterior[:,col], np.random.randint(1, rows))
    else:
        if len(amt) == cols:
            for col in range(cols):
                if np.isnan(np.sum(posterior[:,col])):
                    continue
                else:
                    out[:,col] = np.roll(posterior[:,col], int(amt[col]))
        else:
            raise TypeError("amt does not seem to be the correct shape!")
    return out

def trajectory_score_array(posterior, slope=None, intercept=None, w=None, weights=None, normalize=False):
    """Docstring goes here

    This is the score that Davidson et al. maximizes, in order to get a linear trajectory,
    but here we kind of assume that that we have the trajectory already, and then just score it.

    w is the number of bin rows to include in score, in each direction. That is, w=0 is only the modes,
    and w=1 is a band of width=3, namely the modes, and 1 bin above, and 1 bin below the mode.

    The score is NOT averaged!"""

    rows, cols = posterior.shape

    if w is None:
        w = 0
    if not float(w).is_integer:
        raise ValueError("w has to be an integer!")
    if slope is None or intercept is None:
        slope, intercept, _ = linregress_array(posterior=posterior)

    x = np.arange(cols)
    line_y = np.round((slope*x + intercept)) # in position bin #s

    # idea: cycle each column so that the top w rows are the band surrounding the regression line

    if np.isnan(slope): # this will happen if we have 0 or only 1 decoded bins
        return np.nan
    else:
        temp = column_cycle_array(posterior, -line_y+w)

    if normalize:
        num_non_nan_bins = round(np.nansum(posterior))
    else:
        num_non_nan_bins = 1

    return np.nansum(temp[:2*w+1,:])/num_non_nan_bins


def trajectory_score_bst(bst, tuningcurve, w=None, n_shuffles=250,
                         weights=None, normalize=False):
    """Compute the trajectory scores from Davidson et al. for each event
    in the BinnedSpikeTrainArray.

    This function returns the trajectory scores by decoding all the
    events in the BinnedSpikeTrainArray, and then calling an external
    function to determine the slope and intercept for each event, and
    then finally computing the scores for those events.

    If n_shuffles > 0, then in addition to the trajectory scores,
    shuffled scores will be returned for both column cycle shuffling, as
    well as posterior time bin shuffling (time swap).

    NOTE1: this function does NOT attempt to find the line that
    maximizes the trajectory score. Instead, it delegates the
    determination of the line to an external function (which currently
    is called from trajectory_score_array), and at the time of writing
    this documentation, is simply the best line fit to the modes of the
    decoded posterior distribution.

    NOTE2: the score is then the sum of the probabilities in a band of
    w bins around the line, ignoring bins that are NaNs. Even when w=0
    (only the sum of peak probabilities) this is different from the r^2
    coefficient of determination, in that here more concentrated
    posterior probabilities contribute heavily than weaker ones.

    NOTE3: the returned scores are NOT normalized, but if desired, they
    can be normalized by dividing by the number of non-NaN bins in each
    event.

    Reference(s)
    ------------
    Davidson TJ, Kloosterman F, Wilson MA (2009)
        Hippocampal replay of extended experience. Neuron 63:497–507

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the candidate events to
        score.
    tuningcurve : TuningCurve1D
        Tuning curve to decode events in bst.
    w : int, optional (default is 0)
        Half band width for calculating the trajectory score. If w=0,
        then only the probabilities falling directly under the line are
        used. If w=1, then a total band of 2*w+1 = 3 will be used.
    n_shuffles : int, optional (default is 250)
        Number of times to perform both time_swap and column_cycle
        shuffles.
    weights : not yet used, but idea is to assign weights to the bands
        surrounding the line
    normalize : bool, optional (default is False)
        If True, the scores will be normalized by the number of non-NaN
        bins in each event.

    Returns
    -------
    scores, [scores_time_swap, scores_col_cycle]
        scores is of size (bst.n_epochs, )
        scores_time_swap and scores_col_cycle are each of size
            (n_shuffles, bst.n_epochs)
    """

    if w is None:
        w = 0
    if not float(w).is_integer:
        raise ValueError("w has to be an integer!")

    if float(n_shuffles).is_integer:
        n_shuffles = int(n_shuffles)
    else:
        raise ValueError("n_shuffles must be an integer!")

    posterior, bdries, mode_pth, mean_pth = decode(bst=bst,
                                                   ratemap=tuningcurve)

    # idea: cycle each column so that the top w rows are the band
    # surrounding the regression line

    scores = np.zeros(bst.n_epochs)
    if n_shuffles > 0:
        scores_time_swap = np.zeros((n_shuffles, bst.n_epochs))
        scores_col_cycle = np.zeros((n_shuffles, bst.n_epochs))

    for idx in range(bst.n_epochs):
        posterior_array = posterior[:, bdries[idx]:bdries[idx+1]]
        scores[idx] = trajectory_score_array(posterior=posterior_array,
                                             w=w,
                                             normalize=normalize)
        for shflidx in range(n_shuffles):
            # time swap:

            posterior_ts = time_swap_array(posterior_array)
            posterior_cs = column_cycle_array(posterior_array)
            scores_time_swap[shflidx, idx] = trajectory_score_array(
                posterior=posterior_ts,
                w=w,
                normalize=normalize)
            scores_col_cycle[shflidx, idx] = trajectory_score_array(
                posterior=posterior_cs,
                w=w,
                normalize=normalize)

    if n_shuffles > 0:
        return scores, scores_time_swap, scores_col_cycle
    return scores

def shuffle_transmat(transmat):
    """Shuffle transition probability matrix within each row, leaving self transitions in tact.

    It is assumed that the transmat is stochastic-row-wise, meaning that A_{ij} = Pr(S_{t+1}=j|S_t=i).

    Parameters
    ----------
    transmat : array of size (n_states, n_states)
        Transition probability matrix, where A_{ij} = Pr(S_{t+1}=j|S_t=i).

    Returns
    -------
    shuffled : array of size (n_states, n_states)
        Shuffled transition probability matrix.
    """
    shuffled = transmat.copy()

    nrows, ncols = transmat.shape
    for rowidx in range(nrows):
        all_but_diagonal = np.append(np.arange(rowidx), np.arange(rowidx+1, ncols))
        shuffle_idx = np.random.permutation(all_but_diagonal)
        shuffle_idx = np.insert(shuffle_idx, rowidx, rowidx)
        shuffled[rowidx,:] = shuffled[rowidx, shuffle_idx]

    return shuffled

def score_hmm_logprob(bst, hmm, normalize=False):
    """Score events in a BinnedSpikeTrainArray by computing the log
    probability under the model.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
    hmm : PoissonHMM
    normalize : bool, optional. Default is False.
        If True, log probabilities will be normalized by their sequence
        lengths.
    Returns
    -------
    logprob : array of size (n_events,)
        Log probabilities, one for each event in bst.
    """

    logprob = np.atleast_1d(hmm.score(bst))
    if normalize:
        logprob = np.atleast_1d(logprob) / bst.lengths

    return logprob

def score_hmm_transmat_shuffle(bst, hmm, n_shuffles=250, normalize=False):
    """Score sequences using a hidden Markov model, and a model where
    the transition probability matrix has been shuffled.BaseException

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the candidate events to
        score.
    hmm : PoissonHMM
        Trained hidden markov model to score sequences.
    n_shuffles : int, optional (default is 250)
        Number of times to perform both time_swap and column_cycle
        shuffles.
    normalize : bool, optional (default is False)
        If True, the scores will be normalized by event lengths.

    Returns
    -------
    scores : array of size (n_events,)
    shuffled : array of size (n_shuffles, n_events)
    """

    if float(n_shuffles).is_integer:
        n_shuffles = int(n_shuffles)
    else:
        raise ValueError("n_shuffles must be an integer!")

    hmm_shuffled = copy.deepcopy(hmm)
    scores = score_hmm_logprob(bst=bst,
                               hmm=hmm,
                               normalize=normalize)
    n_events = bst.n_epochs
    shuffled = np.zeros((n_shuffles, n_events))
    for ii in range(n_shuffles):
        hmm_shuffled.transmat_ = shuffle_transmat(hmm_shuffled.transmat_)
        shuffled[ii,:] = score_hmm_logprob(bst=bst,
                                           hmm=hmm_shuffled,
                                           normalize=normalize)

    return scores, shuffled

def score_hmm_timeswap_shuffle(bst, hmm, n_shuffles=250, normalize=False):
    """Score sequences using a hidden Markov model, and a model where
    the transition probability matrix has been shuffled.BaseException

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the candidate events to
        score.
    hmm : PoissonHMM
        Trained hidden markov model to score sequences.
    n_shuffles : int, optional (default is 250)
        Number of times to perform both time_swap and column_cycle
        shuffles.
    normalize : bool, optional (default is False)
        If True, the scores will be normalized by event lengths.

    Returns
    -------
    scores : array of size (n_events,)
    shuffled : array of size (n_shuffles, n_events)
    """

    scores = score_hmm_logprob(bst=bst,
                               hmm=hmm,
                               normalize=normalize)
    n_events = bst.n_epochs
    shuffled = np.zeros((n_shuffles, n_events))
    for ii in range(n_shuffles):
        bst_shuffled = time_swap_bst(bst=bst)
        shuffled[ii,:] = score_hmm_logprob(bst=bst_shuffled,
                                           hmm=hmm,
                                           normalize=normalize)

    return scores, shuffled

def get_significant_events(scores, shuffled_scores, q=95):
    """Return the significant events based on percentiles.

    NOTE: The score is compared to the distribution of scores obtained
    using the randomized data and a Monte Carlo p-value can be computed
    according to: p = (r+1)/(n+1), where r is the number of
    randomizations resulting in a score higher than (ETIENNE EDIT: OR EQUAL TO?)
    the real score and n is the total number of randomizations performed.

    Parameters
    ----------
    scores : array of shape (n_events,)
    shuffled_scores : array of shape (n_shuffles, n_events)
    q : float in range of [0,100]
        Percentile to compute, which must be between 0 and 100 inclusive.

    Returns
    -------
    sig_event_idx : array of shape (n_sig_events,)
        Indices (from 0 to n_events-1) of significant events.
    pvalues :
    """

    n, _ = shuffled_scores.shape
    r = np.sum(shuffled_scores >= scores, axis=0)
    pvalues = (r+1)/(n+1)

    sig_event_idx = np.argwhere(scores > np.percentile(
        shuffled_scores,
        axis=0,
        q=q)).squeeze()

    return np.atleast_1d(sig_event_idx), np.atleast_1d(pvalues)

def score_hmm_logprob_cumulative(bst, hmm, normalize=False):
    """Score events in a BinnedSpikeTrainArray by computing the log
    probability under the model.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
    hmm : PoissonHMM
    normalize : bool, optional. Default is False.
        If True, log probabilities will be normalized by their sequence
        lengths.
    Returns
    -------
    logprob : array of size (n_events,)
        Log probabilities, one for each event in bst.
    """

    logprob = np.atleast_1d(hmm._cum_score_per_bin(bst))
    if normalize:
        cumlengths = []
        for evt in bst.lengths:
            cumlengths.extend(np.arange(1, evt+1).tolist())
        cumlengths = np.array(cumlengths)
        logprob = np.atleast_1d(logprob) / cumlengths

    return logprob

def score_hmm_time_resolved(bst, hmm, n_shuffles=250, normalize=False):
    """Score sequences using a hidden Markov model, and a model where
    the transition probability matrix has been shuffled.BaseException

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        BinnedSpikeTrainArray containing all the candidate events to
        score.
    hmm : PoissonHMM
        Trained hidden markov model to score sequences.
    n_shuffles : int, optional (default is 250)
        Number of times to perform both time_swap and column_cycle
        shuffles.
    normalize : bool, optional (default is False)
        If True, the scores will be normalized by event lengths.

    Returns
    -------
    scores : array of size (n_events,)
    shuffled : array of size (n_shuffles, n_events)
    """

    if float(n_shuffles).is_integer:
        n_shuffles = int(n_shuffles)
    else:
        raise ValueError("n_shuffles must be an integer!")

    hmm_shuffled = copy.deepcopy(hmm)
    Lbraw = score_hmm_logprob_cumulative(bst=bst,
                               hmm=hmm,
                               normalize=normalize)

    # per event, compute L(:b|raw) - L(:b-1|raw)
    Lb = copy.copy(Lbraw)

    cumLengths = np.cumsum(bst.lengths)
    cumLengths = np.insert(cumLengths, 0, 0)

    for ii in range(bst.n_epochs):
        LE = cumLengths[ii]
        RE = cumLengths[ii+1]
        Lb[LE+1:RE] -= Lbraw[LE:RE-1]

    n_bins = bst.n_bins
    shuffled = np.zeros((n_shuffles, n_bins))
    for ii in range(n_shuffles):
        hmm_shuffled.transmat_ = shuffle_transmat(hmm_shuffled.transmat_)
        Lbtmat = score_hmm_logprob_cumulative(bst=bst,
                               hmm=hmm_shuffled,
                               normalize=normalize)

        # per event, compute L(:b|tmat) - L(:b-1|raw)
        NL = copy.copy(Lbtmat)
        for jj in range(bst.n_epochs):
            LE = cumLengths[jj]
            RE = cumLengths[jj+1]
            NL[LE+1:RE] -= Lbraw[LE:RE-1]

        shuffled[ii,:] = NL

    scores = Lb

    return scores, shuffled

def three_consecutive_bins_above_q(pvals, lengths, q=0.75, n_consecutive=3):
    cumLengths = np.cumsum(lengths)
    cumLengths = np.insert(cumLengths, 0, 0)

    above_thresh = 100*(1 - pvals) > q
    idx = []
    for ii in range(len(lengths)):
        LE = cumLengths[ii]
        RE = cumLengths[ii+1]
        temp = 0
        for b in above_thresh[LE:RE]:
            if b:
                temp +=1
            else:
                temp = 0 # reset
        if temp >= n_consecutive:
            idx.append(ii)

    return np.array(idx)

def _scoreOrderD_time_swap(hmm, state_sequences, lengths, n_shuffles=250, normalize=False):
    """Compute order score of state sequences

    A score of 0 means there's only one state.
    """

    scoresD = [] # scores with no adjacent duplicates
    n_sequences = len(state_sequences)
    shuffled = np.zeros((n_shuffles, n_sequences))

    for seqid in range(n_sequences):
        logP = np.log(hmm.transmat_)
        pth = state_sequences[seqid]
        plen = len(pth)
        logPseq = 0
        for ii in range(plen-1):
            logPseq += logP[pth[ii],pth[ii+1]]
        score = logPseq - np.log(plen)
        scoresD.append(score)
        for nn in range(n_shuffles):
            logPseq = 0
            pth = np.random.permutation(pth)
            for ii in range(plen-1):
                logPseq += logP[pth[ii],pth[ii+1]]
            score = logPseq - np.log(plen)
            shuffled[nn, seqid] = score

    scoresD = np.array(scoresD)

    if normalize:
        scoresD = scoresD/lengths
        shuffled = shuffled/lengths

    return scoresD, shuffled

def score_hmm_order_time_swap(bst, hmm, n_shuffles=250, normalize=False):
    lp, paths, centers = hmm.decode(X=bst)
    scores, shuffled = _scoreOrderD_time_swap(hmm, paths, lengths=bst.lengths, n_shuffles=n_shuffles, normalize=normalize)
    if normalize:
        scores = scores/bst.lengths
        shuffled = shuffled/bst.lengths

    return scores, shuffled