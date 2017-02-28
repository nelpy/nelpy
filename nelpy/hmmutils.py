#encoding : utf-8
"""nelpy.hmmutils contains helper functions and wrappers for working
with hmmlearn.
"""

# TODO: add helper code to
# * choose number of parameters
# * fit model
# * score sequence paths
# * score observations
# * decode (with orderings)
# * learn mapping to abstract behavior (default: position)

# see https://github.com/ckemere/hmmlearn
from hmmlearn.hmm import PoissonHMM as PHMM
from .objects import BinnedSpikeTrainArray
from .utils import swap_cols, swap_rows
from warnings import warn

class PoissonHMM(PHMM):
    """Nelpy extension of PoissonHMM.

    Parameters
    ----------

    Attributes
    ----------
    """

    __attributes__ = ['_fs',
                      '_ds',
                      '_unit_ids',
                      '_unit_labels',
                      '_unit_tags']

    def __init__(self, *, n_components, n_iter=None, init_params=None,
                 params=None, verbose=False):

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
                      verbose=verbose)

        # initialize BinnedSpikeTrain attributes
        for attrib in self.__attributes__:
            exec("self." + attrib + " = None")

        # create shortcuts to super() methods that are overridden in
        # this class
        self._fit = PHMM.fit
        self._score = PHMM.score
        self._score_samples = PHMM.score_samples
        self._predict = PHMM.predict
        self._predict_proba = PHMM.predict_proba
        self._decode = PHMM.decode

        self._sample = PHMM.sample

    def get_new_state_order(self, *, method=None, Xtrain=None, Xextern=None):
        """return a state ordering, optionally using augmented data.

        method \in ['transmat' (default), 'mode', 'mean']

        If 'mode' or 'mean' is selected, Xtrain and Xextern are required
        """
        if method is None:
            method = 'transmat'
        neworder = []
        raise NotImplementedError("not implemented yet")

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
            Log probability of the produced state sequence.

        state_sequence : array, shape (n_samples, )
            Labels for each sample from ``X`` obtained via a given
            decoder ``algorithm``.

        See Also
        --------
        score_samples : Compute the log probability under the model and
            posteriors.

        score : Compute the log probability under the model.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            return self._decode(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            logprobs = []
            state_sequences = []
            centers = []
            for seq in X:
                logprob, state_sequence = self._decode(self, seq.data.T, lengths=seq.lengths)
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
        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample from ``X``.
        """

        if not isinstance(X, BinnedSpikeTrainArray):
            # assume we have a feature matrix
            return self._predict_proba(self, X, lengths=lengths)
        else:
            # we have a BinnedSpikeTrainArray
            return self._predict_proba(self, X.data.T, lengths=X.lengths)

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

        _, state_sequence = self.decode(X, lengths)
        return state_sequence

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
            Log likelihood of ``X``.

        posteriors : array, shape (n_samples, n_components)
            State-membership probabilities for each sample in ``X``.

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
            return self._score_samples(self, X.data.T, lengths=X.lengths)

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
            Log likelihood of ``X``.

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
            return self._score(self, X.data.T, lengths=X.lengths)

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

    def __repr__(self):
        try:
            rep = super().__repr__()
        except:
            warning.warn(
                "couldn't access super().__repr__;"
                " upgrade dependencies to resolve this issue."
                )
            rep = "PoissonHMM"
        return "nelpy." + rep

    def fit_ext(self):
        """Learn a mapping from the internal state space, to an external
        augmented space (e.g. position).
        """
        raise NotImplementedError(
            "nelpy.PoissonHMM.fit_ext() not yet implemented")

    def decode_ext(self):
        """Decode observations to the state space, and then map those
        states to an associated external representation (e.g. position).
        """
        raise NotImplementedError(
            "nelpy.PoissonHMM.decode_ext() not yet implemented")
