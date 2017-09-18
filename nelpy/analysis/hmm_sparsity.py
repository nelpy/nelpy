import copy
import numpy as np

from collections import defaultdict
from sklearn.model_selection import train_test_split


from ..core import SpikeTrainArray
from ..hmmutils import PoissonHMM
from .ergodic import steady_state

class HMMSurrogate():

    def __init__(self, *, kind, st, num_states=None, ds=None, test_size=None,
                 random_state=None, verbose=False, description='', PBE_idx=None):
        """

        WARNING!!! All the shuffle methods currently operate directly on PBEs_train,
        and PBEs_test is left unmodified. If we want to evaluate test sets, this
        behavior needs to change!

        Parameters
        ==========
        kind: string
            One of ['actual', 'incoherent', 'coherent', 'poisson', 'unit_id', 'spike_id']
        st: SpikeTrainArray restricted to the PBEs
        num_states: int, optional
            Number of states in the hidden Markov model. Default is 50.
        ds : float, optional
            Bin size for PBEs. Default is 0.02 (=20 ms).
        test_size: float, optional
            Proportion of data to use as test data. Default is 0.2 (=20 %)
        random_state: int, optional
            Random seed for numpy, default is 1
        PBE_idx: tuple of lists , optional
            (PBE_trainidx, PBE_testidx)
        """

        if kind == 'actual':
            self.shuffle = self._do_nothing
        elif kind == 'incoherent':
            self.shuffle = self._within_event_incoherent_shuffle
        elif kind == 'coherent':
            self.shuffle = self._within_event_coherent_shuffle
        elif kind == 'poisson':
            self.shuffle = self._within_event_homogeneous_poisson
        elif kind == 'unit_id':
            self.shuffle = self._within_event_unit_id_shuffle
        elif kind == 'spike_id':
            self.shuffle = self._spike_id_shuffle
        else:
            raise ValueError("unknown data surrogate kind!")

        if num_states is None:
            num_states = 50
        if ds is None:
            ds = 0.02 # 20 ms bin size
        if test_size is None:
            test_size = 0.2
        if random_state is not None:
            np.random.seed(random_state)

        self._random_state = random_state
        self._st = st
        self._num_states = num_states
        self._ds = ds
        self._test_size = test_size
        self._random_state = random_state
        self._verbose = verbose
        self.label = kind
        self.description = description

        self._preprocess(PBE_idx=PBE_idx)
        self.hmm = PoissonHMM(n_components=self._num_states, random_state=self._random_state, verbose=self._verbose)

        self.results = defaultdict(list)

    @property
    def transmat(self):
        return self.hmm.transmat_

    @property
    def means(self):
        return self.hmm.means_

    def clear_results(self):
        self.results = defaultdict(list)

    def score_gini(self, kind='tmat_departure'):
        """Calculate and record the gini coefficients."""
        # kinds = ['tmat_arrival', 'tmat_departure', 'tmat', 'lambda', 'lambda_across_states', 'lambda_across_units']

        gini_distr = []

        # transmat departure sparsity
        if kind=='tmat_departure':
            for row in self.hmm.transmat_:
                gini_distr.append(self._gini(row))
            self.results['gini_tmat_departure'].append(gini_distr)

        # transmat arrival sparsity
        if kind=='tmat_arrival':
            for row in self.hmm.transmat_.T:
                gini_distr.append(self._gini(row))
            self.results['gini_tmat_arrival'].append(gini_distr)

        # transmat sparsity
        if kind=='tmat':
            gini_distr = self._gini(self.hmm.transmat_)
            self.results['gini_tmat'].append(gini_distr)

        # lambda sparsity
        if kind=='lambda':
            gini_distr = self._gini(self.hmm.means_)
            self.results['gini_lambda'].append(gini_distr)

        # lambda_across_states sparsity
        if kind=='lambda_across_states':
            for row in self.hmm.means_.T:
                gini_distr.append(self._gini(row))
            self.results['gini_lambda_across_states'].append(gini_distr)

        # lambda_across_units sparsity
        if kind=='lambda_across_units':
            for row in self.hmm.means_:
                gini_distr.append(self._gini(row))
            self.results['gini_lambda_across_units'].append(gini_distr)

    def score_bottleneck_ratio(self, n_samples=50000):
        def Qij(i, j, P, pi):
            return pi[i] * P[i,j]

        def QAB(A, B, P, pi):
            sumQ = 0
            for i in A:
                for j in B:
                    sumQ += Qij(i, j, P, pi)
            return sumQ

        def complement(S, Omega):
            return Omega - S

        def Pi(S, pi):
            sumS = 0
            for i in S:
                sumS += pi[i]
            return sumS

        def Phi(S, P, pi, Omega):
            Sc = complement(S, Omega)
            return QAB(S, Sc, P, pi) / Pi(S, pi)

        P = self.hmm.transmat_
        num_states = self._num_states
        Omega = set(range(num_states))
        pi_ = steady_state(P).real

        min_Phi = 1
        for nn in range(n_samples):
            n_samp_in_subset = np.random.randint(1, num_states-1)
            S = set(np.random.choice(num_states, n_samp_in_subset, replace=False))
            while Pi(S, pi_) > 0.5:
                n_samp_in_subset -=1
                if n_samp_in_subset < 1:
                    n_samp_in_subset = 1
                S = set(np.random.choice(num_states, n_samp_in_subset, replace=False))
            candidate_Phi = Phi(S, P, pi_, Omega)
            if candidate_Phi < min_Phi:
                min_Phi = candidate_Phi
                if self._verbose:
                    print("{}: {} (|S| = {})".format(nn, min_Phi, len(S)))

        self.results['bottleneck'].append(min_Phi)

    def score_mixing_time(self, eps=0.25):
        raise NotImplementedError

    def score_spectrum(self):
        pass

    def _preprocess_PBEs(self, PBE_idx=None):
        """used for most types of shuffles"""
        # compute PBEs
        self.PBEs = self._st.bin(ds=self._ds)

        if self.PBEs.n_epochs == 1:
            raise ValueError("spike train is continuous, and does not have more than one event!")

        if PBE_idx is not None:
            self._trainidx, self._testidx = PBE_idx # tuple unpacking
        else:
            # split into train and test data
            if self._random_state is not None:
                self._trainidx, self._testidx = train_test_split(np.arange(self.PBEs.n_epochs), test_size=self._test_size, random_state=self._random_state)
            else:
                self._trainidx, self._testidx = train_test_split(np.arange(self.PBEs.n_epochs), test_size=self._test_size, random_state=1)

        self._trainidx.sort()
        self._testidx.sort()

        self.PBEs_train = self.PBEs[self._trainidx]
        self.PBEs_test = self.PBEs[self._testidx]

    def _preprocess_STs(self):
        """used for spike ID shuffle, where spike times must be shuffled"""
        # split into train and test data
        self._preprocess_PBEs()
        self._st_flat = self._st.flatten().time.squeeze()

    def _preprocess(self, PBE_idx=None):
        self._preprocess_PBEs(PBE_idx=PBE_idx)
        if self.label == 'spike_id':
            self._preprocess_STs()

    def fit(self):
        self.hmm = PoissonHMM(n_components=self._num_states, verbose=self._verbose)

        # train HMM on all training PBEs
        self.hmm.fit(self.PBEs_train)

        # re-order states of hmm
        transmat_order = self.hmm.get_state_order('transmat')
        self.hmm.reorder_states(transmat_order)

    def score_loglikelihood(self):
        # record log-likelihood on both train and validation sets after fitting model:

        # train_LL = np.array(self.hmm.score(self.PBEs_train)).sum() # one scalar for each sequence in training set
        # test_LL = np.array(self.hmm.score(self.PBEs_test)).sum() # one scalar for each sequence in training set
        train_LL = self.hmm.score(self.PBEs_train) # one scalar for each sequence in training set
        test_LL = self.hmm.score(self.PBEs_test) # one scalar for each sequence in training set

        self.results['loglikelihood_train'].extend(train_LL)
        self.results['loglikelihood_test'].extend(test_LL)

    def _gini(self, array):
        """Calculate the Gini coefficient of a numpy array."""
        # https://github.com/oliviaguest/gini
        # based on bottom eq:
        # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
        # from:
        # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
        # All values are treated equally, arrays must be 1d:
        array = array.flatten()
        if np.amin(array) < 0:
            # Values cannot be negative:
            array -= np.amin(array)
        # Values cannot be 0:
        array += 0.0000001
        # Values must be sorted:
        array = np.sort(array)
        # Index per array element:
        index = np.arange(1,array.shape[0]+1)
        # Number of array elements:
        n = array.shape[0]
        # Gini coefficient:
        return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

    def _do_nothing(self, kind='train'):
        """Do nothing to the data."""
        pass


    def _within_event_coherent_shuffle(self, kind='train'):
        """Time swap on BinnedSpikeTrainArray, swapping only within each epoch."""
        if kind == 'train':
            bst = self.PBEs_train
        elif kind == 'test':
            bst = self.PBEs_test
        else:
            raise ValueError("kind '{}' not understood!".format(kind))

        out = copy.deepcopy(bst) # should this be deep?
        shuffled = np.arange(bst.n_bins)
        edges = np.insert(np.cumsum(bst.lengths),0,0)
        for ii in range(bst.n_epochs):
            segment = shuffled[edges[ii]:edges[ii+1]]
            shuffled[edges[ii]:edges[ii+1]] = np.random.permutation(segment)

        out._data = out._data[:,shuffled]

        if kind == 'train':
            self.PBEs_train = out
        else:
            self.PBEs_test = out

    def _within_event_incoherent_shuffle(self, kind='train'):
        """Time cycle on BinnedSpikeTrainArray, cycling only within each epoch.
        We cycle each unit independently, within each epoch.
        """
        if kind == 'train':
            bst = self.PBEs_train
        elif kind == 'test':
            bst = self.PBEs_test
        else:
            raise ValueError("kind '{}' not understood!".format(kind))

        out = copy.deepcopy(bst) # should this be deep?
        data = out._data
        edges = np.insert(np.cumsum(bst.lengths),0,0)

        for uu in range(bst.n_units):
            for ii in range(bst.n_epochs):
                segment = np.squeeze(data[uu, edges[ii]:edges[ii+1]])
                segment = np.roll(segment, np.random.randint(len(segment)))
                data[uu, edges[ii]:edges[ii+1]] = segment

        if kind == 'train':
            self.PBEs_train = out
        else:
            self.PBEs_test = out

    def _within_event_homogeneous_poisson(self, kind='train'):
        """Estimate mean firing rates across all events, and then generate
        homogeneous Poisson spikes within each event. That is, ISIs do not
        span multiple events, but are started fresh within each event."""

        if kind == 'train':
            bst = self.PBEs_train
        elif kind == 'test':
            bst = self.PBEs_test
        else:
            raise ValueError("kind '{}' not understood!".format(kind))

        firing_rates = bst.n_spikes / bst.support.duration # firing rate in Hz

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

        if kind == 'train':
            self.PBEs_train = poisson_st.bin(ds=bst.ds)
        else:
            self.PBEs_test = poisson_st.bin(ds=bst.ds)

    def _spike_id_shuffle(self, proportional=False, kind='train'):
        """Shuffle the cell/unit identity of each spike, independently.

        If 'proportional' is True, then sample spike IDs in proportion to
        the original data numbers of spikes. Otherwise, sample spike IDs
        uniformly.
        """

        all_spiketimes = self._st_flat
        spike_ids = np.zeros(len(all_spiketimes))

        if proportional:
            n_spikes = self._st.n_spikes
        else:
            n_spikes = np.ones(self._st.n_units)* np.floor(self._st.n_spikes.sum() / self._st.n_units)

        pointer = 0
        for uu, n_spikes in enumerate(n_spikes):
            spike_ids[pointer:pointer+int(n_spikes)] = uu
            pointer += int(n_spikes)

        # permute spike IDs
        spike_ids = np.random.permutation(spike_ids)

        # now re-assign all spike times according to sampling above
        spikes = []
        for unit in range(self._st.n_units):
            spikes.append(all_spiketimes[spike_ids==unit])

        shuffled_st = SpikeTrainArray(timestamps=spikes, support=self._st.support)

        if kind == 'train':
            self.PBEs_train = shuffled_st.bin(ds=self._ds)[self._trainidx]
        elif kind == 'test':
            self.PBEs_test = shuffled_st.bin(ds=self._ds)[self._testidx]
        else:
            raise ValueError("kind '{}' not understood!".format(kind))

    def _within_event_unit_id_shuffle(self, kind='train'):
        """Unit ID shuffle on BinnedSpikeTrainArray, shuffling independently within each epoch."""

        if kind == 'train':
            bst = self.PBEs_train
        elif kind == 'test':
            bst = self.PBEs_test
        else:
            raise ValueError("kind '{}' not understood!".format(kind))

        out = copy.deepcopy(bst) # should this be deep?
        data = out._data
        edges = np.insert(np.cumsum(bst.lengths),0,0)

        unit_list = np.arange(bst.n_units)

        for ii in range(bst.n_epochs):
            segment = data[:, edges[ii]:edges[ii+1]]
            out._data[:, edges[ii]:edges[ii+1]] = segment[np.random.permutation(unit_list)]

        if kind == 'train':
            self.PBEs_train = out
        else:
            self.PBEs_test = out