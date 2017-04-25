#encoding : utf-8
"""nelpy.hmmutils contains helper functions and wrappers for working
with hmmlearn.
"""

# TODO: add helper code to
# * choose number of parameters
# * decode (with orderings)

# see https://github.com/ckemere/hmmlearn
from hmmlearn.hmm import PoissonHMM as PHMM
from .core import BinnedSpikeTrainArray # may have to be from . import core, and then core.BinnedSpikeTrainArray
from .utils import swap_cols, swap_rows
from warnings import warn
import numpy as np
from pandas import unique
from . import plotting
from matplotlib.pyplot import subplots

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

        if start_state is None:
            start_state = np.argmax(self.startprob_)

        new_order = [start_state]
        num_states = self.transmat_.shape[0]
        rem_states = np.arange(0,start_state).tolist()
        rem_states.extend(np.arange(start_state+1,num_states).tolist())
        cs = start_state

        for ii in np.arange(0,num_states-1):
            nstilde = np.argmax(self.transmat_[cs,rem_states])
            ns = rem_states[nstilde]
            rem_states.remove(ns)
            cs = ns
            new_order.append(cs)

        return new_order

    def get_state_order(self, method=None):
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
            return self._get_order_from_transmat()
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

    def decode(self, X, lengths=None, algorithm=None):
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
            return self._decode(self, X, lengths=lengths), None
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            state_sequences = []
            centers = []
            for seq in X:
                logprob, state_sequence = self._decode(self, seq.data.T, lengths=seq.lengths, algorithm=algorithm)
                logprobs.append(logprob)
                state_sequences.append(state_sequence)
                centers.append(seq.centers)
            return logprobs, state_sequences, centers

    def predict_proba(self, X, lengths=None):
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
            return np.transpose(self._predict_proba(self, X, lengths=lengths))
        else:
            # we have a BinnedSpikeTrainArray
            return np.transpose(self._predict_proba(self, X.data.T, lengths=X.lengths))

    def predict(self, X, lengths=None):
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

        _, state_sequences, centers = self.decode(X, lengths)
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
        raise NotImplementedError(
            "PoissonHMM.sample() has not been implemented yet.")

    def score_samples(self, X, lengths=None):
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
            return self._score_samples(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            posteriors = []
            for seq in X:
                logprob, posterior = self._score_samples(self, seq.data.T)
                logprobs.append(logprob)
                posteriors.append(posterior.T)
            return logprobs, posteriors

    def score(self, X, lengths=None):
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
        logprob : float
            Log likelihood of ``X``; one scalar for each sequence in X.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.
        decode : Find most likely state sequence corresponding to ``X``.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            return self._score(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            for seq in X:
                logprob = self._score(self, seq.data.T)
                logprobs.append(logprob)
        return logprobs

    def fit(self, X, lengths=None):
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
            self._fit(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            self._fit(self, X.data.T, lengths=X.lengths)
            self.assume_attributes(X)
        return self

    def fit_ext(self, X, ext, n_extern=None, lengths=None, save=True):
        """Learn a mapping from the internal state space, to an external
        augmented space (e.g. position).

        X : BinnedSpikeTrainArray

        ext : array-lke
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

        posteriors = self.predict_proba(X)
        posteriors = np.vstack(posteriors.T)  # 1D array of states, of length n_bins

        if len(posteriors) != len(ext):
            raise ValueError("ext must have same length as decoded state sequence!")

        for ii, posterior in enumerate(posteriors):
            extern[:,ext_map[ext[ii]]] += np.transpose(posterior)

        # normalize extern tuning curves:
        colsum = np.tile(extern.sum(axis=1),(n_extern,1)).T
        extern = extern/colsum

        if save:
            self._extern_ = extern
            # self._extern_map = ext_map

        return extern

    def decode_ext(self, X, lengths=None, algorithm=None):
        """Find most likely state sequence corresponding to ``X``, and
        then map those states to an associated external representation
        (e.g. position).

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
        algorithm : string, one of the ``DECODER_ALGORITHMS`` decoder
            algorithm to be used.

        Returns
        -------
        logprob : float
            Log probability of the produced state sequence.

        ext_sequences : array, shape (n_samples, )
            External labels for each sample from ``X`` obtained via a
            given decoder ``algorithm``.

        ext_posteriors : array, shape (n_extern, n_samples)
            State-membership probabilities for each sample in ``X``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            raise NotImplementedError("not implemented yet.")
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            state_sequences = []
            external_sequences = []
            posteriors = []
            for seq in X:
                logprob, state_sequence = self._decode(self, seq.data.T, lengths=seq.lengths, algorithm=algorithm)
                logprobs.append(logprob)
                external_sequence = state_sequence
                external_sequences.append(external_sequence)
                posterior = self.predict_proba(seq)
                posterior = np.dot(self._extern_.T, posterior)
                posteriors.append(posterior)
            return logprobs, external_sequences, posteriors

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