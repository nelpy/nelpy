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

from scipy.ndimage import convolve
from scipy import stats

from ..hmmutils import PoissonHMM
from ..core import SpikeTrainArray
from .. import auxiliary
from ..decoding import decode1D as decode
from ..decoding import k_fold_cross_validation
from ..decoding import get_mode_pth_from_array, get_mean_pth_from_array

def get_line_of_best_Davidson_score(bst, tuningcurve, w=3, n_samples=50000):
    tc = tuningcurve

    if bst.n_epochs > 1:
        raise TypeError('You can only pass one PBE at a time!')

    def sub2ind(array_shape, rows, cols):
        return rows*array_shape[1] + cols

    def calc_ri(NT, NP, phi, rho, ci, ci_mid, ri_mid):
        """
        Note: Not matrix dimensions! Think of dim0 as
        x coordinate, dim1 as y coordinate, etc.
        """
        ri = (rho - (ci - ci_mid) * np.cos(phi)) / np.sin(phi) + ri_mid
        ri = np.around(ri).astype(int) # Find nearest position bin

        return ri

    def _score_line_ri_ci(posterior, precond_posterior, NT, NP, ri, ci):
        scores_outside_track = np.nanmedian(posterior[:,(ri > NP - 1) | (ri < 0)], axis=0)

        coords = sub2ind(posterior.shape, ri, ci)
        coords = coords[(ri < NP) & (ri >= 0)]
        scores_within_track = np.take(precond_posterior, coords)

        num_empty_bins = np.isnan(scores_outside_track).sum() + np.isnan(scores_within_track).sum()

        score_within_track = np.nansum(scores_within_track)
        if (score_within_track) > 0 & (num_empty_bins > 0):
            temp = np.nanmedian(scores_within_track)*num_empty_bins
        else:
            temp = 0
        score_outside_track = np.nansum(scores_outside_track) + temp

        score = score_within_track + score_outside_track

        # we divide by NT later on to be more efficient
        # final_score = score/NT

        return score

    def find_best_line(posterior, precond_posterior, phis, rhos):
        best_score = 0
        best_ri = []

        NP, NT = posterior.shape

        ci_mid = (NT + 1)/2 # CONST
        ri_mid = (NP + 1)/2 # CONST
        ci = np.arange(NT)  # CONST

        for phi, rho in zip(phis, rhos):

            # parameterize line
            ri = calc_ri(NT, NP, phi, rho, ci, ci_mid, ri_mid)

            score = _score_line_ri_ci(posterior, precond_posterior, NT, NP, ri, ci)

            if score > best_score:
                best_score = score
                best_ri = ri

        score = _score_line_ri_ci(posterior, precond_posterior, NT, NP, best_ri, ci)/NT

        return score, best_ri

    # decode neural activity
    posterior_array, bdries, mode_pth, mean_pth = decode(bst=bst, ratemap=tc, xmax=310)

    # precondition matrix kernel for banded summation
    k = np.zeros((2*w+1, 3))
    k[:,1] = 1

    NP, NT = posterior_array.shape

    D = np.sqrt((NT-1)**2 + (NP-1)**2)
    phi_range = (-0.5*np.pi, 0.5*np.pi)
    rho_range = (-0.5*D, 0.5*D)

    phis = phi_range[0] + np.random.rand(n_samples)*(phi_range[1] - phi_range[0])
    phis[(phis < 0.0001) & (phis > -0.0001)] = 0.0001
    rhos = rho_range[0] + np.random.rand(n_samples)*(rho_range[1] - rho_range[0])

    precond_posterior = convolve(posterior_array, k, mode='constant', cval=0.0)

    score, ri = find_best_line(posterior=posterior_array,
                                    precond_posterior=precond_posterior,
                                    phis=phis,
                                    rhos=rhos)

    ci = np.arange(NT)

    return score, ri, ci

def score_hmm_events(bst, k_folds=None, num_states=30, n_shuffles=5000, shuffle='row-wise', verbose=False):
    """scores all sequences in the entire bst"""
    if k_folds is None:
        k_folds = 5

    if shuffle == 'row-wise':
        rowwise = True
    elif shuffle == 'col-wise':
        rowwise = False
    else:
        raise ValueError("tmat must be either 'row-wise' or 'col-wise'")

    X = [ii for ii in range(bst.n_epochs)]

    scores_hmm = np.zeros(bst.n_epochs)
    scores_hmm_shuffled = np.zeros((bst.n_epochs, n_shuffles))

    for kk, (training, validation) in enumerate(k_fold_cross_validation(X, k=k_folds)):
        if verbose:
            print('  fold {}/{}'.format(kk+1, k_folds))

        PBEs_train = bst[training]
        PBEs_test = bst[validation]

        # train HMM on all training PBEs
        hmm = PoissonHMM(n_components=num_states, random_state=0, verbose=False)
        hmm.fit(PBEs_train)

        # reorder states according to transmat ordering
        transmat_order = hmm.get_state_order('transmat')
        hmm.reorder_states(transmat_order)

        # compute scores_hmm (log likelihoods) of validation set:
        scores_hmm[validation] = hmm.score(PBEs_test)

        hmm_shuffled = copy.deepcopy(hmm)
        for nn in range(n_shuffles):
            # shuffle transition matrix:
            if rowwise:
                hmm_shuffled.transmat_ = shuffle_transmat(hmm_shuffled.transmat)
            else:
                hmm_shuffled.transmat_ = shuffle_transmat_Kourosh_breaks_stochasticity(hmm_shuffled.transmat)
                hmm_shuffled.transmat_ = hmm_shuffled.transmat / np.tile(hmm_shuffled.transmat.sum(axis=1), (hmm_shuffled.n_components, 1)).T

            # score validation set with shuffled HMM
            scores_hmm_shuffled[validation, nn] = hmm_shuffled.score(PBEs_test)

    n_scores = len(scores_hmm)
    scores_hmm_percentile = np.array([stats.percentileofscore(scores_hmm_shuffled[idx], scores_hmm[idx], kind='mean') for idx in range(n_scores)])

    return scores_hmm, scores_hmm_shuffled, scores_hmm_percentile

def score_Davidson_final_bst_fast(bst, tuningcurve, w=None, n_shuffles=2000, n_samples=35000, verbose=False):
    """Compute the trajectory scores from Davidson et al. 2009 for each event
    in the BinnedSpikeTrainArray. DO IT EVEN FASTER!!!
    """

    def sub2ind(n_cols, rows, cols):
        return rows*n_cols + cols

    def calc_ri(NT, NP, phi, rho, ci, ci_mid, ri_mid):
        """
        Note: Not matrix dimensions! Think of dim0 as
        x coordinate, dim1 as y coordinate, etc.
        """
        ri = (rho - (ci - ci_mid) * np.cos(phi)) / np.sin(phi) + ri_mid
        ri = np.around(ri).astype(int) # Find nearest position bin

        return ri

    def _score_line_ri_ci(posterior, precond_posterior, NT, NP, ri, ci, ncols, median_post, nanbins, n_nanbins):

        scores_outside_track = median_post[((ri > NP - 1) & ~nanbins) | ((ri < 0)& ~nanbins)]

        coords = sub2ind(NT, ri, ci)
        coords = coords[(ri < NP) & (ri >= 0) & (~nanbins)]
        scores_within_track = np.take(precond_posterior, coords)

        nanscore = 0
        if n_nanbins > 0:
            nanscore = n_nanbins * np.median(scores_within_track)

        score_within_track = np.sum(scores_within_track)
        score_outside_track = np.sum(scores_outside_track)

        score = score_within_track + score_outside_track

        # we divide by NT later on to be more efficient
        # final_score = score/NT

        return score

    def find_best_line(posterior, precond_posterior, phis, rhos, NP, NT, median_post, nanbins, n_nanbins):
        best_score = 0
        best_ri = []

        # n_rows, n_cols = posterior.shape
#         NP, NT = posterior.shape

        ci_mid = (NT + 1)/2 # CONST
        ri_mid = (NP + 1)/2 # CONST
        ci = np.arange(NT)  # CONST

        for phi, rho in zip(phis, rhos):

            # parameterize line
            ri = calc_ri(NT, NP, phi, rho, ci, ci_mid, ri_mid)

            score = _score_line_ri_ci(posterior, precond_posterior, NT, NP, ri, ci, NT, median_post, nanbins, n_nanbins)
            if score > best_score:
                best_score = score
                best_ri = ri

        score = _score_line_ri_ci(posterior, precond_posterior, NT, NP, best_ri, ci, NT, median_post, nanbins, n_nanbins)/NT
        return score, best_ri

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

    # precondition matrix kernel for banded summation
    k = np.zeros((2*w+1, 3))
    k[:,1] = 1

    scores_bayes = np.zeros(bst.n_epochs)

    if n_shuffles > 0:
        scores_bayes_shuffled = np.zeros((n_shuffles, bst.n_epochs))

    for idx in range(bst.n_epochs):
        if verbose:
            print("scoring event ", idx+1, "/", bst.n_epochs)

        posterior_array = posterior[:, bdries[idx]:bdries[idx+1]]

        # now we zero out all the nan bins (we compensate for them later...)
        nanbins = np.isnan(np.max(posterior_array, axis=0))
        n_nanbins = np.count_nonzero(nanbins)
        posterior_array[:, nanbins] = 0

        # now pre-compute median of entire array
        posterior_median = np.median(posterior_array, axis=0)

        NP, NT = posterior_array.shape

        D = np.sqrt((NT-1)**2 + (NP-1)**2)
        phi_range = (-0.5*np.pi, 0.5*np.pi)
        rho_range = (-0.5*D, 0.5*D)

        phis = phi_range[0] + np.random.rand(n_samples)*(phi_range[1] - phi_range[0])
        phis[(phis < 0.0001) & (phis > -0.0001)] = 0.0001
        rhos = rho_range[0] + np.random.rand(n_samples)*(rho_range[1] - rho_range[0])

        precond_posterior = convolve(posterior_array, k, mode='constant', cval=0.0)

        scores_bayes[idx], _ = find_best_line(posterior=posterior_array,
                                        precond_posterior=precond_posterior,
                                        phis=phis,
                                        rhos=rhos,
                                        NP=NP,
                                        NT=NT,
                                        median_post=posterior_median,
                                        nanbins=nanbins,
                                        n_nanbins=n_nanbins)
        if n_shuffles > 0:
            posterior_cs = copy.deepcopy(posterior_array)
            precond_posterior_cs = copy.deepcopy(precond_posterior)

            for shflidx in range(n_shuffles):

                for col in range(NT):
                    random_offset = np.random.randint(1, NP)
                    posterior_cs[:,col] = np.roll(posterior_cs[:,col], random_offset)
                    precond_posterior_cs[:,col] = np.roll(precond_posterior_cs[:,col], random_offset)

                # ideally we should re-sample phi and rho here for every sequence, but to save time, we don't...
                scores_bayes_shuffled[shflidx, idx], _ = find_best_line(posterior=posterior_cs,
                                                                    precond_posterior=precond_posterior_cs,
                                                                    phis=phis,
                                                                    rhos=rhos,
                                                                    NP=NP,
                                                                    NT=NT,
                                                                    median_post=posterior_median,
                                                                    nanbins=nanbins,
                                                                    n_nanbins=n_nanbins)
    if n_shuffles > 0:
        scores_bayes_shuffled = scores_bayes_shuffled.T
        n_scores = len(scores_bayes)
        scores_bayes_percentile = np.array([stats.percentileofscore(scores_bayes_shuffled[idx], scores_bayes[idx], kind='mean') for idx in range(n_scores)])
        return scores_bayes, scores_bayes_shuffled, scores_bayes_percentile
    return scores_bayes

def score_Davidson_final_bst(bst, tuningcurve, w=None, n_shuffles=2000, n_samples=35000, verbose=False):
    """Compute the trajectory scores from Davidson et al. 2009 for each event
    in the BinnedSpikeTrainArray. DO IT FAST!!!
    """

    def sub2ind(array_shape, rows, cols):
        return rows*array_shape[1] + cols

    def calc_ri(NT, NP, phi, rho, ci, ci_mid, ri_mid):
        """
        Note: Not matrix dimensions! Think of dim0 as
        x coordinate, dim1 as y coordinate, etc.
        """
        ri = (rho - (ci - ci_mid) * np.cos(phi)) / np.sin(phi) + ri_mid
        ri = np.around(ri).astype(int) # Find nearest position bin

        return ri

    def _score_line_ri_ci(posterior, precond_posterior, NT, NP, ri, ci):

        scores_outside_track = np.nanmedian(posterior[:,(ri > NP - 1) | (ri < 0)], axis=0)

        coords = sub2ind(posterior.shape, ri, ci)
        coords = coords[(ri < NP) & (ri >= 0)]
        scores_within_track = np.take(precond_posterior, coords)

        num_empty_bins = np.isnan(scores_outside_track).sum() + np.isnan(scores_within_track).sum()

        score_within_track = np.nansum(scores_within_track)
        if (score_within_track) > 0 & (num_empty_bins > 0):
            temp = np.nanmedian(scores_within_track)*num_empty_bins
        else:
            temp = 0
        score_outside_track = np.nansum(scores_outside_track) + temp

        score = score_within_track + score_outside_track

        # we divide by NT later on to be more efficient
        # final_score = score/NT

        return score

    def find_best_line(posterior, precond_posterior, phis, rhos):
        best_score = 0
        best_ri = []

        NP, NT = posterior.shape

        ci_mid = (NT + 1)/2 # CONST
        ri_mid = (NP + 1)/2 # CONST
        ci = np.arange(NT)  # CONST

        for phi, rho in zip(phis, rhos):

            # parameterize line
            ri = calc_ri(NT, NP, phi, rho, ci, ci_mid, ri_mid)

            score = _score_line_ri_ci(posterior, precond_posterior, NT, NP, ri, ci)
            if score > best_score:
                best_score = score
                best_ri = ri

        score = _score_line_ri_ci(posterior, precond_posterior, NT, NP, best_ri, ci)/NT
        return score, best_ri

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

    # precondition matrix kernel for banded summation
    k = np.zeros((2*w+1, 3))
    k[:,1] = 1

    scores_bayes = np.zeros(bst.n_epochs)

    if n_shuffles > 0:
        scores_bayes_shuffled = np.zeros((n_shuffles, bst.n_epochs))

    for idx in range(bst.n_epochs):
        if verbose:
            print("scoring event ", idx+1, "/", bst.n_epochs)

        posterior_array = posterior[:, bdries[idx]:bdries[idx+1]]

        NP, NT = posterior_array.shape

        D = np.sqrt((NT-1)**2 + (NP-1)**2)
        phi_range = (-0.5*np.pi, 0.5*np.pi)
        rho_range = (-0.5*D, 0.5*D)

        phis = phi_range[0] + np.random.rand(n_samples)*(phi_range[1] - phi_range[0])
        phis[(phis < 0.0001) & (phis > -0.0001)] = 0.0001
        rhos = rho_range[0] + np.random.rand(n_samples)*(rho_range[1] - rho_range[0])

        precond_posterior = convolve(posterior_array, k, mode='constant', cval=0.0)

        scores_bayes[idx], _ = find_best_line(posterior=posterior_array,
                                        precond_posterior=precond_posterior,
                                        phis=phis,
                                        rhos=rhos)
        if n_shuffles > 0:
            posterior_cs = copy.deepcopy(posterior_array)
            precond_posterior_cs = copy.deepcopy(precond_posterior)

            for shflidx in range(n_shuffles):

                for col in range(NT):
                    random_offset = np.random.randint(1, NP)
                    posterior_cs[:,col] = np.roll(posterior_cs[:,col], random_offset)
                    precond_posterior_cs[:,col] = np.roll(precond_posterior_cs[:,col], random_offset)

                # ideally we should re-sample phi and rho here for every sequence, but to save time, we don't...
                scores_bayes_shuffled[shflidx, idx], _ = find_best_line(posterior=posterior_cs,
                                                                    precond_posterior=precond_posterior_cs,
                                                                    phis=phis,
                                                                    rhos=rhos)
    if n_shuffles > 0:
        scores_bayes_shuffled = scores_bayes_shuffled.T
        n_scores = len(scores_bayes)
        scores_bayes_percentile = np.array([stats.percentileofscore(scores_bayes_shuffled[idx], scores_bayes[idx], kind='mean') for idx in range(n_scores)])
        return scores_bayes, scores_bayes_shuffled, scores_bayes_percentile
    return scores_bayes

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
    out = copy.deepcopy(posterior)
    rows, cols = posterior.shape

    colidx = np.arange(cols)
    shuffle_cols = np.random.permutation(colidx)
    out = out[:,shuffle_cols]

    return out

def time_swap_bst(bst):
    """Time swap on BinnedSpikeTrainArray, swapping only within each epoch."""
    out = copy.deepcopy(bst) # should this be deep? YES! Oh my goodness, yes!
    shuffled = np.arange(bst.n_bins)
    edges = np.insert(np.cumsum(bst.lengths),0,0)
    for ii in range(bst.n_epochs):
        segment = shuffled[edges[ii]:edges[ii+1]]
        shuffled[edges[ii]:edges[ii+1]] = np.random.permutation(segment)

    out._data = out._data[:,shuffled]

    return out

def incoherent_shuffle_bst(bst):
    """Incoherent shuffle on BinnedSpikeTrainArray, swapping only within each epoch."""
    out = copy.deepcopy(bst) # should this be deep? YES! Oh my goodness, yes!
    data = out._data
    edges = np.insert(np.cumsum(bst.lengths),0,0)

    for uu in range(bst.n_units):
        for ii in range(bst.n_epochs):
            segment = np.squeeze(data[uu, edges[ii]:edges[ii+1]])
            segment = np.roll(segment, np.random.randint(len(segment)))
            data[uu, edges[ii]:edges[ii+1]] = segment

    return out

def poisson_surrogate_bst(bst):
    """Create a Poisson surrogate of BinnedSpikeTrainArray."""
    firing_rates = bst.n_spikes / bst.support.duration # firing rates in Hz

    spikes = []

    for rate in firing_rates:
        unit_spikes = []
        for start, stop in bst.support.time:
            evt_duration = stop - start
            n_evt_spikes = np.random.poisson(rate * evt_duration)
            spike_times = start + np.random.uniform(0, evt_duration, n_evt_spikes)
            unit_spikes.extend(spike_times)

        spikes.append(unit_spikes)

    support = bst.support.expand(bst.ds/2, direction='stop')
    poisson_st = SpikeTrainArray(timestamps=spikes, support=support)

    out = poisson_st.bin(ds=bst.ds)
    # out = out[bst.support]

    return out

def spike_id_shuffle_bst(bst, st_flat):
    """Create a spike ID shuffled surrogate of BinnedSpikeTrainArray."""
    all_spiketimes = st_flat.time.squeeze()
    spike_ids = np.zeros(len(all_spiketimes))

    n_spikes = np.ones(bst.n_units)* np.floor(st_flat.n_spikes[0] / bst.n_units)

    pointer = 0
    for uu, n_spikes in enumerate(n_spikes):
        spike_ids[pointer:pointer+int(n_spikes)] = uu
        pointer += int(n_spikes)

    # permute spike IDs
    spike_ids = np.random.permutation(spike_ids)

    # now re-assign all spike times according to sampling above
    spikes = []
    for unit in range(bst.n_units):
        spikes.append(all_spiketimes[spike_ids==unit])

    support = bst.support.expand(bst.ds/2, direction='stop')
    shuffled_st = SpikeTrainArray(timestamps=spikes, support=support)

    out = shuffled_st.bin(ds=bst.ds)
    # out = out[bst.support]

    return out

def unit_id_shuffle_bst(bst):
    """Create a unit ID shuffled surrogate of BinnedSpikeTrainArray."""
    out = copy.deepcopy(bst) # should this be deep?
    data = out._data
    edges = np.insert(np.cumsum(bst.lengths),0,0)

    unit_list = np.arange(bst.n_units)

    for ii in range(bst.n_epochs):
        segment = data[:, edges[ii]:edges[ii+1]]
        out._data[:, edges[ii]:edges[ii+1]] = segment[np.random.permutation(unit_list)]

    return out

def column_cycle_array(posterior, amt=None):
    """Also called 'position cycle' by Kloosterman et al.
    If amt is an array of the same length as posterior, then
    cycle each column by the corresponding amount in amt.
    Otherwise, cycle each column by a random amount."""
    out = copy.deepcopy(posterior)
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
    posterior probabilities contribute more heavily than weaker ones.

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

def shuffle_transmat_Kourosh_breaks_stochasticity(transmat):
    """Shuffle transition probability matrix within each column, leaving self transitions in tact.

    It is assumed that the transmat is stochastic-row-wise, meaning that A_{ij} = Pr(S_{t+1}=j|S_t=i).

    NOTE: this breaks stochasticity! To get back to a stochastic matrix, we should do:
    transmat = transmat / np.tile(transmat.sum(axis=1), (hmm.n_components, 1)).T

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
    for colidx in range(ncols):
        all_but_diagonal = np.append(np.arange(colidx), np.arange(colidx+1, nrows))
        shuffle_idx = np.random.permutation(all_but_diagonal)
        shuffle_idx = np.insert(shuffle_idx, colidx, colidx)
        shuffled[:, colidx] = shuffled[shuffle_idx, colidx]

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
    the transition probability matrix has been shuffled.

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

def score_hmm_incoherent_shuffle(bst, hmm, n_shuffles=250, normalize=False):
    """Docstring goes here.

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
        bst_shuffled = incoherent_shuffle_bst(bst=bst)
        shuffled[ii,:] = score_hmm_logprob(bst=bst_shuffled,
                                           hmm=hmm,
                                           normalize=normalize)

    return scores, shuffled

def score_hmm_poisson_shuffle(bst, hmm, n_shuffles=250, normalize=False):
    """Docstring goes here.

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
        bst_shuffled = poisson_surrogate_bst(bst=bst)
        shuffled[ii,:] = score_hmm_logprob(bst=bst_shuffled,
                                           hmm=hmm,
                                           normalize=normalize)

    return scores, shuffled

def score_hmm_spike_id_shuffle(bst, hmm, st_flat, n_shuffles=250, normalize=False):
    """Docstring goes here.

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
        bst_shuffled = spike_id_shuffle_bst(bst=bst, st_flat=st_flat)
        shuffled[ii,:] = score_hmm_logprob(bst=bst_shuffled,
                                           hmm=hmm,
                                           normalize=normalize)

    return scores, shuffled

def score_hmm_unit_id_shuffle(bst, hmm, n_shuffles=250, normalize=False):
    """Docstring goes here.

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
        bst_shuffled = unit_id_shuffle_bst(bst=bst)
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
    Lb = copy.deepcopy(Lbraw)

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
        NL = copy.deepcopy(Lbtmat)
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