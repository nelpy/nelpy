__all__ = ['linregress_ting',
           'linregress_array',
           'linregress_bst',
           'time_swap_array',
           'column_cycle_array',
           'trajectory_score_array',
           'trajectory_score_bst',
           'get_significant_events']

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

    return sig_event_idx, pvalues