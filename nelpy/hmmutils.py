#encoding : utf-8
"""nelpy.hmmutils contains helper functions and wrappers for working
with hmmlearn.
"""

from hmmlearn import hmm # see https://github.com/ckemere/hmmlearn

# TODO: consider whether to simply write a wrapper for hmmlearn, or
# to inherit from it. As of now, I think writing a simple wrapper is
# more in line with what I need.

# TODO: add helper code to 
# * choose number of parameters
# * fit model
# * score sequence paths
# * score observations
# * decode (with orderings)
# * learn mapping to abstract behavior (default: position)

class PoissonHMM:
    """Class description.

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

        # initialize a PoissonHMM object
        self._hmm = hmm.PoissonHMM(n_components=n_components,
                                   n_iter=n_iter,
                                   init_params=init_params,
                                   params=params,
                                   verbose=verbose)

    def __repr__(self):
        return "nelpy." + self.hmm.__repr__()

    @property
    def hmm(self):
        """Property description goes here."""
        return self._hmm

    @property
    def means_(self):
        """Property description goes here."""
        raise NotImplementedError("property not implemented yet")

    @property
    def transmat_(self):
        """Property description goes here."""
        raise NotImplementedError("property not implemented yet")

    @property
    def n_units(self):
        """Property description goes here."""
        raise NotImplementedError("property not implemented yet")
        # return self.means_.shape

    @property
    def n_components(self):
        """Number of components (states)."""
        return self.hmm.n_components

    @property
    def startprob_prior(self):
        """Property description goes here."""
        return self.hmm.startprob_prior

    @startprob_prior.setter
    def startprob_prior(self, val):
        self.hmm.startprob_prior = val

    @property
    def n_iter(self):
        """Property description goes here."""
        return self.hmm.n_iter

    @n_iter.setter
    def n_iter(self, val):
        """Property description goes here."""
        self.hmm.n_iter = val

    # def score():

    # def fit():

    # def scoreSAMPLES():

