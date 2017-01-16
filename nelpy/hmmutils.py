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

class PoissonHMM(PHMM):
    """Nelpy PoissonHMM.

    Parameters
    ----------

    Attributes
    ----------
    """
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

        # create shortcuts to super() methods that are overridden in
        # this class

        # TODO: does this syntax actually work?
        self.fit_ = PHMM.fit
        self.decode_ = PHMM.fit
        self.score_ = PHMM.score
        self.score_samples_ = PHMM.score_samples

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
        lengths : array-like of integers, shape (n_sequences, )
            Lengths of the individual sequences in ``X``. The sum of
            these should be ``n_samples``.

        Returns
        -------
        self : object
            Returns self.
        """

        # bst.lengths and bst.data.T should get us very close
        pass

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
            "nelpy.PoissonHMM.decde_ext() not yet implemented")

    def decode_ext(self):
        """Decode observations to the state space, and then map those
        states to an associated external representation (e.g. position).
        """
        raise NotImplementedError(
            "nelpy.PoissonHMM.decde_ext() not yet implemented")
