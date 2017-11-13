#encoding : utf-8
"""nelpy.hmmutils contains helper functions and wrappers for working
with hmmlearn.
"""

# TODO: add helper code to
# * choose number of parameters
# * decode (with orderings)

# see https://github.com/ckemere/hmmlearn
from hmmlearn.hmm import PoissonHMM as PHMM
from warnings import warn
import numpy as np
from pandas import unique
from matplotlib.pyplot import subplots
import copy

from .core import BinnedSpikeTrainArray # may have to be from . import core, and then core.BinnedSpikeTrainArray
from .utils import swap_cols, swap_rows
from . import plotting
from . decoding import decode1D

__all__ = ['PoissonHMM']

class PoissonHMM(PHMM):
    """Nelpy extension of PoissonHMM: Hidden Markov Model with
    independent Poisson emissions.

    Parameters
    ----------
    n_components : int
        Number of states.

    startprob_prior : array, shape (n_components, )
        Initial state occupation prior distribution.

    transmat_prior : array, shape (n_components, n_components)
        Matrix of prior transition probabilities between states.

    algorithm : string, one of the :data:`base.DECODER_ALGORITHMS`
        Decoder algorithm.

    random_state: RandomState or an int seed (0 by default)
        A random number generator instance.

    n_iter : int, optional
        Maximum number of iterations to perform.

    tol : float, optional
        Convergence threshold. EM will stop if the gain in log-likelihood
        is below this value.

    verbose : bool, optional
        When ``True`` per-iteration convergence reports are printed
        to :data:`sys.stderr`. You can diagnose convergence via the
        :attr:`monitor_` attribute.

    params : string, optional
        Controls which parameters are updated in the training
        process.  Can contain any combination of 's' for startprob,
        't' for transmat, 'm' for means and 'c' for covars. Defaults
        to all parameters.

    init_params : string, optional
        Controls which parameters are initialized prior to
        training.  Can contain any combination of 's' for
        startprob, 't' for transmat, 'm' for means and 'c' for covars.
        Defaults to all parameters.

    Attributes
    ----------
    n_features : int
        Dimensionality of the (independent) Poisson emissions.

    monitor_ : ConvergenceMonitor
        Monitor object used to check the convergence of EM.

    transmat_ : array, shape (n_components, n_components)
        Matrix of transition probabilities between states.

    startprob_ : array, shape (n_components, )
        Initial state occupation distribution.

    means_ : array, shape (n_components, n_features)
        Mean parameters for each state.

    extern_ : array, shape (n_components, n_extern)
        Augmented mapping from state space to external variables.

    Examples
    --------
    >>> from nelpy.hmmutils import PoissonHMM
    >>> PoissonHMM(n_components=2)...

    """

    __attributes__ = ['_fs',
                      '_ds',
                      '_unit_ids',
                      '_unit_labels',
                      '_unit_tags']

    def __init__(self, *, n_components, n_iter=None, init_params=None,
                 params=None, random_state=None, verbose=False):

        # assign default parameter values
        if n_iter is None:
            n_iter = 50
        if init_params is None:
            init_params = 'stm'
        if params is None:
            params = 'stm'

        # TODO: I don't understand why super().__init__ does not work?
        PHMM.__init__(self,
                      n_components=n_components,
                      n_iter=n_iter,
                      init_params=init_params,
                      params=params,
                      random_state=random_state,
                      verbose=verbose)

        # initialize BinnedSpikeTrain attributes
        for attrib in self.__attributes__:
            exec("self." + attrib + " = None")

        self._extern_ = None
        # self._extern_map = None

        # create shortcuts to super() methods that are overridden in
        # this class
        self._fit = PHMM.fit
        self._score = PHMM.score
        self._score_samples = PHMM.score_samples
        self._predict = PHMM.predict
        self._predict_proba = PHMM.predict_proba
        self._decode = PHMM.decode

        self._sample = PHMM.sample

    def __repr__(self):
        try:
            rep = super().__repr__()
        except:
            warning.warn(
                "couldn't access super().__repr__;"
                " upgrade dependencies to resolve this issue."
                )
            rep = "PoissonHMM"
        if self._extern_ is not None:
            fit_ext = "True"
        else:
            fit_ext = "False"
        try:
            fit = "False"
            if self.means_ is not None:
                fit = "True"
        except AttributeError:
            fit = "False"
        fitstr = "; fit=" + fit + ", fit_ext=" + fit_ext
        return "nelpy." + rep + fitstr

    @property
    def extern_(self):
        """Mapping from states to external variables (e.g., position)"""
        if self._extern_ is not None:
            return self._extern_
        else:
            warn("no state <--> external mapping has been learnt yet!")
            return None

    def _get_order_from_transmat(self, start_state=None):
        """Determine a state ordering based on the transition matrix.

        This is a greedy approach, starting at the a priori most probable
        state, and moving to the next most probable state according to
        the transition matrix, and so on.

        Parameters
        ----------
        start_state : int, optional
            Initial state to begin from. Defaults to the most probable
            a priori state.

        Returns
        -------
        new_order : list
            List of states in transmat order.
        """

        # unless specified, start in the a priori most probable state
        if start_state is None:
            start_state = np.argmax(self.startprob_)

        new_order = [start_state]
        num_states = self.transmat_.shape[0]
        rem_states = np.arange(0,start_state).tolist()
        rem_states.extend(np.arange(start_state+1,num_states).tolist())
        cs = start_state # current state

        for ii in np.arange(0, num_states-1):
            # find largest transition to set of remaining states
            nstilde = np.argmax(self.transmat_[cs,rem_states])
            ns = rem_states[nstilde]
            # remove selected state from list of remaining states
            rem_states.remove(ns)
            cs = ns
            new_order.append(cs)

        return new_order

    @property
    def unit_ids(self):
        return self._unit_ids

    @property
    def unit_labels(self):
        return self._unit_labels

    @property
    def means(self):
        """Observation matrix, (n_components, n_units)."""
        return self.means_

    @property
    def transmat(self):
        """Transition probability matrix, (n_components, n_components).
        NOTE: Aij = Pr(S_{t+1}=j|S_t=i).
        """
        return self.transmat_

    @property
    def startprob(self):
        """Prior distribution over states, (n_components,)."""
        return self.startprob_

    def get_state_order(self, method=None, start_state=None):
        """return a state ordering, optionally using augmented data.

        method \in ['transmat' (default), 'mode', 'mean']

        If 'mode' or 'mean' is selected, self._extern_ must exist

        NOTE: both 'mode' and 'mean' assume that _extern_ is in sorted
        order; this is not verified explicitly.
        """
        if method is None:
            method = 'transmat'

        neworder = []

        if method == 'transmat':
            return self._get_order_from_transmat(start_state=start_state)
        elif method == 'mode':
            if self._extern_ is not None:
                neworder = self._extern_.argmax(axis=1).argsort()
            else:
                raise Exception("External mapping does not exist yet."
                                "First use PoissonHMM.fit_ext()")
        elif method == 'mean':
            if self._extern_ is not None:
                (np.tile(np.arange(self._extern_.shape[1]),(self.n_components,1))*self._extern_).sum(axis=1).argsort()
                neworder = self._extern_.argmax(axis=1).argsort()
            else:
                raise Exception("External mapping does not exist yet."
                                "First use PoissonHMM.fit_ext()")
        else:
            raise NotImplementedError("ordering method '" + str(method) + "' not supported!")
        return neworder

    def _reorder_units_by_ids(self, neworder):
        """Reorder unit_ids to match that of a BinnedSpikeTrain.

        WARNING! modifies self.means_ in-place

        neworder must be list-like, of size (n_units,) and in terms of
        unit_ids

        Return
        ------
        self : reordered PoissonHMM
        """

        neworder = [self.unit_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            swap_cols(self.means_, frm, to)
            self._unit_ids[frm], self._unit_ids[to] = self._unit_ids[to], self._unit_ids[frm]
            self._unit_labels[frm], self._unit_labels[to] = self._unit_labels[to], self._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        return self

    def reorder_states(self, neworder):
        """Reorder internal HMM states according to a specified order.

        neworder must be list-like, of size (n_components,)
        """
        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            swap_cols(self.transmat_, frm, to)
            swap_rows(self.transmat_, frm, to)
            swap_rows(self.means_, frm, to)
            if self._extern_ is not None:
                swap_rows(self._extern_, frm, to)
            self.startprob_[frm], self.startprob_[to] = self.startprob_[to], self.startprob_[frm]
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

    def assume_attributes(self, binnedSpikeTrainArray):
        """Assume subset of attributes from a BinnedSpikeTrainArray.

        This is used primarily to enable the sampling of sequences after
        a model has been fit.
        """

        if self._ds is not None:
            warn("PoissonHMM(BinnedSpikeTrain) attributes already exist.")
        for attrib in self.__attributes__:
            exec("self." + attrib + " = binnedSpikeTrainArray." + attrib)
        self._unit_ids = copy.copy(binnedSpikeTrainArray.unit_ids)
        self._unit_labels = copy.copy(binnedSpikeTrainArray.unit_labels)
        self._unit_tags = copy.copy(binnedSpikeTrainArray.unit_tags)

    def _has_same_unit_id_order(self, unit_ids):
        """Returns True if the unit_ids are in the specified order."""
        if self._unit_ids is None:
            return True
        if len(unit_ids) != len(self.unit_ids):
            raise TypeError("Incorrect number of unit_ids encountered!")
        for ii, unit_id in enumerate(unit_ids):
            if unit_id != self.unit_ids[ii]:
                return False
        return True

    def _sliding_window_array(self, bst, w=1):
        """Returns an unwrapped data array by sliding w bins one bin at a time.

        If w==1, then bins are non-overlapping.

        Parameters
        ----------
        bst : BinnedSpikeTrainArray, with data array of shape (n_units, n_bins)

        Returns
        -------
        unwrapped : new data array of shape (n_sliding_bins, n_units)
        lengths : array of shape (n_sliding_bins,)
        """

        if w is None:
            w=1
        assert float(w).is_integer(), "w must be a positive integer!"
        assert w > 0, "w must be a positive integer!"

        if not isinstance(bst, BinnedSpikeTrainArray):
            raise NotImplementedError ("support for other datatypes not yet implemented!")

        # potentially re-organize internal observation matrix to be
        # compatible with BinnedSpikeTrainArray
        if not self._has_same_unit_id_order(bst.unit_ids):
            self._reorder_units_by_ids(bst.unit_ids)

        if w == 1:
            return bst.data.T, bst.lengths

        n_units, t_bins = bst.data.shape

        # if we decode using multiple bins at a time (w>1) then we have to decode each epoch separately:

        # first, we determine the number of bins we will decode. This requires us to scan over the epochs
        n_bins = 0
        cumlengths = np.cumsum(bst.lengths)
        lengths = np.zeros(bst.n_epochs, dtype=np.int)
        prev_idx = 0
        for ii, to_idx in enumerate(cumlengths):
            datalen = to_idx - prev_idx
            prev_idx = to_idx
            lengths[ii] = np.max((1,datalen - w + 1))

        n_bins = lengths.sum()

        unwrapped = np.zeros((n_units, n_bins))

        # next, we decode each epoch separately, one bin at a time
        cum_lengths = np.insert(np.cumsum(lengths),0,0)

        prev_idx = 0
        for ii, to_idx in enumerate(cumlengths):
            data = bst.data[:,prev_idx:to_idx]
            prev_idx = to_idx
            datacum = np.cumsum(data, axis=1) # ii'th data segment, with column of zeros prepended
            datacum = np.hstack((np.zeros((n_units,1)), datacum))
            re = w # right edge ptr
            # TODO: check if datalen < w and act appropriately
            if lengths[ii] > 1: # more than one full window fits into data length
                for tt in range(lengths[ii]):
                    obs = datacum[:, re] - datacum[:, re-w] # spikes in window of size w
                    re+=1
                    post_idx = lengths[ii] + tt
                    unwrapped[:,post_idx] = obs
            else: # only one window can fit in, and perhaps only partially. We just take all the data we can get,
                # and ignore the scaling problem where the window size is now possibly less than bst.ds*w
                post_idx = cum_lengths[ii]
                obs = datacum[:, -1] # spikes in window of size at most w
                unwrapped[:,post_idx] = obs

        return unwrapped.T, lengths

    def decode(self, X, lengths=None, w=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
            WARNING! Each decoding window is assumed to be similar in
            size to those used during training. If not, the tuning curves
            have to be scaled appropriately!
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.
        algorithm : string, one of the ``DECODER_ALGORITHMS``
            decoder algorithm to be used.

        Returns
        -------
        logprob : float
            Log probability of the PRODUCED STATE SEQUENCE.
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.
        centers : array, shape (n_samples, )
            time-centers of all bins contained in ``X``

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            return self._decode(self, X=X, lengths=lengths, algorithm=algorithm), None
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            state_sequences = []
            centers = []
            for seq in X:
                windowed_arr, lengths = self._sliding_window_array(bst=seq, w=w)
                logprob, state_sequence = self._decode(self, windowed_arr, lengths=lengths, algorithm=algorithm)
                logprobs.append(logprob)
                state_sequences.append(state_sequence)
                centers.append(seq.centers)
            return logprobs, state_sequences, centers

    def _decode_from_lambda_only(self, X, lengths=None):
        """Decode using the observation (lambda) matrix only. That is, pure
           memoryless decoding.

        >>> posteriors, state_sequences = hmm._decode_from_lambda_only(bst)

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
            WARNING! Each decoding window is assumed to be similar in
            size to those used during training. If not, the tuning curves
            have to be scaled appropriately!
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        posteriors : array, shape (n_components, n_samples)
            State-membership probabilities for each sample in ``X``;
            one array for each sequence in X.
        state_sequences : array, shape (n_samples, )
            Labels for each sample from ``X``; one array for each sequence in X.
        """
        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            raise NotImplementedError ("Not yet implemented!")
        else:
            # we have a BinnedSpikeTrainArray
            ratemap = copy.deepcopy(self.means_.T)
            # make sure X and ratemap have same unit_id ordering!
            neworder = [self.unit_ids.index(x) for x in X.unit_ids]
            oldorder = list(range(len(neworder)))
            for oi, ni in enumerate(neworder):
                frm = oldorder.index(ni)
                to = oi
                swap_rows(ratemap, frm, to)
                oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

            posteriors = []
            state_sequences = []
            for seq in X:
                posteriors_, cumlengths, mode_pth, mean_pth = decode1D(bst=seq, ratemap=ratemap)
                # nanlocs = np.argwhere(np.isnan(mode_pth))
                # state_sequences_ = mode_pth.astype(int)
                state_sequences_ = mode_pth
                posteriors.append(posteriors_)
                state_sequences.append(state_sequences_)

            return posteriors, state_sequences

    def predict_proba(self, X, lengths=None, w=None, returnLengths=False):
        """Compute the posterior probability for each state in the model.

        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        posteriors : array, shape (n_components, n_samples)
            State-membership probabilities for each sample from ``X``.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            print("we have a " + str(type(X)))
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            if returnLengths:
                return np.transpose(self._predict_proba(self, X, lengths=lengths)), lengths
            return np.transpose(self._predict_proba(self, X, lengths=lengths))
        else:
            # we have a BinnedSpikeTrainArray
            windowed_arr, lengths = self._sliding_window_array(bst=X, w=w)
            if returnLengths:
                return np.transpose(self._predict_proba(self, windowed_arr, lengths=lengths)), lengths
            return np.transpose(self._predict_proba(self, windowed_arr, lengths=lengths))

    def predict(self, X, lengths=None, w=None):
        """Find most likely state sequence corresponding to ``X``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X``.
        """

        _, state_sequences, centers = self.decode(X=X, lengths=lengths, w=w)
        return state_sequences

    def sample(self, n_samples=1, random_state=None):
        # TODO: here we should really use X.unit_ids, tags, etc. to
        # return a BST object. Probably have to copy the attributes
        # during init, but we will only have these if BST is used,
        # instead of a feature matrix. So, if we only used a feature
        # matrix, then we return a feature matrix? Or just a new,
        # compatible BST?
        """Generate random samples from the model.

        DESCRIPTION GOES HERE... TODO TODO TODO TODO

        Parameters
        ----------
        n_samples : int
            Number of samples to generate.

        random_state: RandomState or an int seed (0 by default)
            A random number generator instance. If ``None``, the object's
            random_state is used.

        Returns
        -------
        X : array, shape (n_samples, n_features)
            Feature matrix.
        state_sequence : array, shape (n_samples, )
            State sequence produced by the model.
        """
        return self._sample(self, n_samples=n_samples, random_state=random_state)
        # raise NotImplementedError(
        #     "PoissonHMM.sample() has not been implemented yet.")

    def score_samples(self, X, lengths=None, w=None):
        """Compute the log probability under the model and compute posteriors.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        logprob : float
            Log likelihood of ``X``; one scalar for each sequence in X.

        posteriors : array, shape (n_components, n_samples)
            State-membership probabilities for each sample in ``X``;
            one array for each sequence in X.

        See Also
        --------
        score : Compute the log probability under the model.
        decode : Find most likely state sequence corresponding to ``X``.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            logprobs, posteriors = self._score_samples(self, X, lengths=lengths)
            return logprobs, posteriors.T
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            posteriors = []
            for seq in X:
                windowed_arr, lengths = self._sliding_window_array(bst=seq, w=w)
                logprob, posterior = self._score_samples(self, X=windowed_arr, lengths=lengths)
                logprobs.append(logprob)
                posteriors.append(posterior.T)
            return logprobs, posteriors

    def score(self, X, lengths=None, w=None):
        """Compute the log probability under the model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        logprob : float, or list of floats
            Log likelihood of ``X``; one scalar for each sequence in X.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            return self._score(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            for seq in X:
                windowed_arr, lengths = self._sliding_window_array(bst=seq, w=w)
                logprob = self._score(self, X=windowed_arr, lengths=lengths)
                logprobs.append(logprob)
        return logprobs

    def _cum_score_per_bin(self, X, lengths=None, w=None):
        """Compute the log probability under the model, cumulatively for each bin per event."""

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            return self._score(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            for seq in X:
                windowed_arr, lengths = self._sliding_window_array(bst=seq, w=w)
                n_bins, _ = windowed_arr.shape
                for ii in range(1, n_bins+1):
                    logprob = self._score(self, X=windowed_arr[:ii,:])
                    logprobs.append(logprob)
        return logprobs

    def fit(self, X, lengths=None, w=None):
        """Estimate model parameters using nelpy objects.

        An initialization step is performed before entering the
        EM-algorithm. If you want to avoid this step for a subset of
        the parameters, pass proper ``init_params`` keyword argument
        to estimator's constructor.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_units)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        self : object
            Returns self.
        """
        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            self._fit(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            windowed_arr, lengths = self._sliding_window_array(bst=X, w=w)
            self._fit(self, windowed_arr, lengths=lengths)
            # adopt unit_ids, unit_labels, etc. from BinnedSpikeTrain
            self.assume_attributes(X)
        return self

    def fit_ext(self, X, ext, n_extern=None, lengths=None, save=True, w=None):
        """Learn a mapping from the internal state space, to an external
        augmented space (e.g. position).

        Returns a row-normalized version of (n_states, n_ext), that
        is, a distribution over external bins for each state.

        X : BinnedSpikeTrainArray

        ext : array-like
            array of external correlates (n_bins, )
        n_extern : int
            number of extern variables, with range 0,.. n_extern-1

        save : bool
            stores extern in PoissonHMM if true, discards it if not

        self.extern_ of size (n_components, n_extern)
        """

        ext_map = np.arange(n_extern)
        if n_extern is None:
            n_extern = len(unique(ext))
            for ii, ele in enumerate(unique(ext)):
                ext_map[ele] = ii

        # idea: here, ext can be anything, and n_extern should be range
        # we can e.g., define extern correlates {leftrun, rightrun} and
        # fit the mapping. This is not expexted to be good at all for
        # most states, but it could allow us to identify a state or two
        # for which there *might* be a strong predictive relationship.
        # In this way, the binning, etc. should be done external to this
        # function, but it might still make sense to encapsulate it as
        # a helper function inside PoissonHMM?

        # xpos, ypos = get_position(exp_data['session1']['posdf'], bst.centers)
        # x0=0; xl=100; n_extern=50
        # xx_left = np.linspace(x0,xl,n_extern+1)
        # xx_mid = np.linspace(x0,xl,n_extern+1)[:-1]; xx_mid += (xx_mid[1]-xx_mid[0])/2
        # ext = np.digitize(xpos, xx_left) - 1 # spatial bin numbers

        extern = np.zeros((self.n_components, n_extern))

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            posteriors = self.predict_proba(X=X, lengths=lengths, w=w)
        else:
            # we have a BinnedSpikeTrainArray
            posteriors = self.predict_proba(X=X, lengths=lengths, w=w)

        posteriors = np.vstack(posteriors.T)  # 1D array of states, of length n_bins

        if len(posteriors) != len(ext):
            raise ValueError("ext must have same length as decoded state sequence!")

        for ii, posterior in enumerate(posteriors):
            if not np.isnan(ext[ii]):
                extern[:,ext_map[int(ext[ii])]] += np.transpose(posterior)

        # normalize extern tuning curves:
        rowsum = np.tile(extern.sum(axis=1),(n_extern,1)).T
        rowsum = np.where(np.isclose(rowsum, 0), 1, rowsum)
        extern = extern/rowsum

        if save:
            self._extern_ = extern
            # self._extern_map = ext_map

        return extern

    def fit_ext2(self, X, ext, n_extern=None, lengths=None, w=None):
        """Learn a mapping from the internal state space, to an external
        augmented space (e.g. position).

        Returns a column-normalized version of (n_states, n_ext), that
        is, a distribution over states for each extern bin.

        X : BinnedSpikeTrainArray

        ext : array-like
            array of external correlates (n_bins, )
        n_extern : int
            number of extern variables, with range 0,.. n_extern-1

        save : bool
            stores extern in PoissonHMM if true, discards it if not

        self.extern_ of size (n_components, n_extern)
        """

        ext_map = np.arange(n_extern)
        if n_extern is None:
            n_extern = len(unique(ext))
            for ii, ele in enumerate(unique(ext)):
                ext_map[ele] = ii

        # idea: here, ext can be anything, and n_extern should be range
        # we can e.g., define extern correlates {leftrun, rightrun} and
        # fit the mapping. This is not expexted to be good at all for
        # most states, but it could allow us to identify a state or two
        # for which there *might* be a strong predictive relationship.
        # In this way, the binning, etc. should be done external to this
        # function, but it might still make sense to encapsulate it as
        # a helper function inside PoissonHMM?

        # xpos, ypos = get_position(exp_data['session1']['posdf'], bst.centers)
        # x0=0; xl=100; n_extern=50
        # xx_left = np.linspace(x0,xl,n_extern+1)
        # xx_mid = np.linspace(x0,xl,n_extern+1)[:-1]; xx_mid += (xx_mid[1]-xx_mid[0])/2
        # ext = np.digitize(xpos, xx_left) - 1 # spatial bin numbers

        extern = np.zeros((self.n_components, n_extern))

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
            posteriors = self.predict_proba(X=X, lengths=lengths, w=w)
        else:
            # we have a BinnedSpikeTrainArray
            posteriors = self.predict_proba(X=X, lengths=lengths, w=w)
        posteriors = np.vstack(posteriors.T)  # 1D array of states, of length n_bins

        if len(posteriors) != len(ext):
            raise ValueError("ext must have same length as decoded state sequence!")

        for ii, posterior in enumerate(posteriors):
            if not np.isnan(ext[ii]):
                extern[:,ext_map[int(ext[ii])]] += np.transpose(posterior)

        # normalize extern tuning curves:
        colsum = np.tile(extern.sum(axis=0), (self.n_components, 1))
        colsum = np.where(np.isclose(colsum, 0), 1, colsum)
        extern = extern/colsum

        return extern

    def decode_ext(self, X, lengths=None, w=None, ext_shape=None):
        """Find memoryless most likely state sequence corresponding to ``X``,
        (that is, the symbol-by-symbol MAP sequence) and then map those
        states to an associated external representation (e.g. position).

        example 1d
        ----------
        posterior_pos, bdries, mode_pth, mean_pth = hmm.decode_ext(bst_no_ripple, ext_shape=(vtc.n_bins,))
        mean_pth = vtc.bins[0] + mean_pth*(vtc.bins[-1] - vtc.bins[0])

        example 2d
        ----------
        posterior_, bdries_, mode_pth_, mean_pth_ = hmm.decode_ext(bst, ext_shape=(ext_nx, ext_ny))
        mean_pth_[0,:] = vtc2d.xbins[0] + mean_pth_[0,:]*(vtc2d.xbins[-1] - vtc2d.xbins[0])
        mean_pth_[1,:] = vtc2d.ybins[0] + mean_pth_[1,:]*(vtc2d.ybins[-1] - vtc2d.ybins[0])

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix of individual samples.
            OR
            nelpy.BinnedSpikeTrainArray
        lengths : array-like of integers, shape (n_sequences, ), optional
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``. This is not used when X is
            a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
            automatically inferred.

        Returns
        -------
        ext_posteriors, bdries, mode_pth, mean_pth

        ext_posteriors : array, shape (n_extern, n_samples)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """

        _, n_extern = self._extern_.shape

        if ext_shape is None:
            ext_shape = (n_extern)

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            raise NotImplementedError("not implemented yet.")
            if w is not None:
                raise NotImplementedError ("sliding window decoding for feature matrices not yet implemented!")
        else:
            # we have a BinnedSpikeTrainArray
            pass
        if len(ext_shape) == 1:
            # do old style decoding
            # TODO: this can be improved to be like the 2D case!
            state_posteriors, lengths = self.predict_proba(X=X, lengths=lengths, w=w, returnLengths=True)
            # fixy = np.mean(self._extern_ * np.arange(n_extern), axis=1)
            # mean_pth = np.sum(state_posteriors.T*fixy, axis=1) # range 0 to 1
            ext_posteriors = np.dot((self._extern_ * np.arange(n_extern)).T, state_posteriors)
            # normalize ext_posterior distributions:
            ext_posteriors = ext_posteriors / ext_posteriors.sum(axis=0)
            mean_pth = (ext_posteriors.T*np.atleast_2d(np.linspace(0,1, n_extern))).sum(axis=1)
            mode_pth = np.argmax(ext_posteriors, axis=0)/n_extern # range 0 to n_extern

        elif len(ext_shape) == 2:
            ext_posteriors = np.zeros((ext_shape[0], ext_shape[1], X.n_bins))
            # get posterior distribution over states, of size (num_States, n_extern)
            state_posteriors, lengths = self.predict_proba(X=X, lengths=lengths, w=w, returnLengths=True)
            # for each bin, compute the distribution in the external domain
            for bb in range(X.n_bins):
                ext_posteriors[:,:,bb] = np.reshape((self._extern_*state_posteriors[:,[bb]]).sum(axis=0), ext_shape)
            # now compute mean and mode paths
            expected_x = np.sum((ext_posteriors.sum(axis=1)*np.atleast_2d(np.linspace(0,1, ext_shape[0])).T), axis=0)
            expected_y = np.sum((ext_posteriors.sum(axis=0)*np.atleast_2d(np.linspace(0,1, ext_shape[1])).T), axis=0)
            mean_pth = np.vstack((expected_x, expected_y))

            mode_pth = np.zeros((2, X.n_bins))
            for tt in range(X.n_bins):
                if np.any(np.isnan(ext_posteriors[:,:,tt])):
                    mode_pth[0,tt] = np.nan
                    mode_pth[0,tt] = np.nan
                else:
                    x_, y_ = np.unravel_index(np.argmax(ext_posteriors[:,:,tt]), (ext_shape[0], ext_shape[1]))
                    mode_pth[0,tt] = x_/ext_shape[0]
                    mode_pth[1,tt] = y_/ext_shape[1]

            ext_posteriors = np.transpose(ext_posteriors, axes=[1,0,2])
        else:
            raise TypeError("shape not currently supported!")

        bdries = np.cumsum(lengths)

        return ext_posteriors, bdries, mode_pth, mean_pth

    def _plot_external(self, *, figsize=(3,5), sharey=True,
                       labelstates=None, ec=None, fillcolor=None,
                       lw=None):
        """plot the externally associated state<-->extern mapping

        WARNING! This function is not complete, and hence 'private',
        and may be moved somewhere else later on.
        """

        if labelstates is None:
            labelstates = [1, self.n_components]
        if ec is None:
            ec = 'k'
        if fillcolor is None:
            fillcolor = 'gray'
        if lw is None:
            lw = 1.5

        fig, axes = subplots(self.n_components, 1, figsize=figsize, sharey=sharey)

        xvals = np.arange(len(self._extern_.T[:,0]))

        for state, ax in enumerate(axes):
            ax.fill_between(xvals, 0, self._extern_.T[:,state], color=fillcolor)
            ax.plot(xvals, self._extern_.T[:,state], color=ec, lw=lw)
            if state + 1 in labelstates:
                ax.set_ylabel(str(state+1), rotation=0, y=-0.1)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plotting.utils.no_yticks(ax)
            plotting.utils.no_xticks(ax)
        # fig.suptitle('normalized place fields sorted by peak location (left) and mean location (right)', y=0.92, fontsize=14)
        # ax.set_xticklabels(['0','20', '40', '60', '80', '100'])
        ax.set_xlabel('external variable')
        fig.text(0.02, 0.5, 'normalized state distribution', va='center', rotation='vertical')

        return fig, ax

# def score_samples_ext(self, X, lengths=None):
#         """Compute the log probability under the model and compute posteriors.

#         Parameters
#         ----------
#         X : array-like, shape (n_samples, n_features)
#             Feature matrix of individual samples.
#             OR
#             nelpy.BinnedSpikeTrainArray
#         lengths : array-like of integers, shape (n_sequences, ), optional
#             Lengths of the individual sequences in ``X``. The sum of
#             these should be ``n_samples``. This is not used when X is
#             a nelpy.BinnedSpikeTrainArray, in which case the lenghts are
#             automatically inferred.

#         Returns
#         -------
#         logprob : float
#             Log likelihood of ``X``.

#         posteriors : array, shape (n_samples, n_components)
#             State-membership probabilities for each sample in ``X``.

#         See Also
#         --------
#         score : Compute the log probability under the model.
#         decode : Find most likely state sequence corresponding to ``X``.
#         """

#         if not isinstance(X, BinnedSpikeTrainArray):
#             # assume we have a feature matrix
#             return self._score_samples(self, X, lengths=lengths)
#         else:
#             # we have a BinnedSpikeTrainArray
#             logprobs = []
#             posteriors = []
#             for seq in X:
#                 logprob, posterior = self._score_samples(self, seq.data.T)
#                 logprobs.append(logprob)
#                 posteriors.append(posterior)
#             return logprobs, posteriors