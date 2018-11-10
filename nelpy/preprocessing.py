"""Data preprocessing objects and functions."""

import numpy as np

from .utils import PrettyDuration
from . import core

__all__ = ['DataWindow',
           'StreamingDataWindow']


class DataWindow():
    """
    DataWindow
    Data window description to describe stride and/or data aggregation.

    Parameters
    ----------
    bins_before : int, optional (default=0)
        How many bins before the output to include in the window.
    bins_after : int, optional (default=0)
        How many bins after the output to include in the window.
    bins_current : int, optional (default=1)
        Whether (1) or not (0) to include the concurrent bin in the window.
    bins_stride : int, optional (default=1)
        Number of bins to advance the window during each time step.
    bin_width : float, optional (default=None)
        Width of single bin (default units are in seconds).

    Examples
    --------
    >>> w = DataWindow(1,1,1,1)
    DataWindow(bins_before=1, bins_after=1, bins_current=1, bins_stride=1, bin_width=None)

    # Implicit bin size of 1 second, centered window of duration 5 seconds, stride of 2 seconds:
    >>> w = DataWindow(2,2,1,2)
    DataWindow(bins_before=2, bins_after=2, bins_current=1, bins_stride=2)

    # Excplicit bin size of 1 second, centered window of duration 5 seconds, stride of 2 seconds:
    >>> w = DataWindow(2,2,1,2,1)
    DataWindow(bins_before=2, bins_after=2, bins_current=1, bins_stride=2, bin_width=1)
            Total bin width = 5 seconds
    """

    def __init__(self, bins_before=0, bins_after=0, bins_current=1, bins_stride=1, bin_width=None):
        self.bins_before = bins_before
        self.bins_after = bins_after
        self.bins_current = bins_current
        self.bins_stride = bins_stride
        self.bin_width = bin_width
        pass

    def __str__(self):
        if self.bin_width is not None:
            repr_string = "DataWindow(bins_before={}, bins_after={}, bins_current={}, bins_stride={}, bin_width={})" \
            .format(self._bins_before, self._bins_after, self._bins_current, self._bins_stride, self._bin_width)
        else:
            repr_string = "DataWindow(bins_before={}, bins_after={}, bins_current={}, bins_stride={})" \
            .format(self._bins_before, self._bins_after, self._bins_current, self._bins_stride)
        return repr_string

    def __repr__(self):
        if self.bin_width is not None:
            repr_string = "DataWindow(bins_before={}, bins_after={}, bins_current={}, bins_stride={}, bin_width={})" \
            .format(self.bins_before, self.bins_after, self.bins_current, self.bins_stride, self.bin_width)
            repr_string += "\n\tTotal bin width = {}" \
            .format(PrettyDuration((self.bins_before + self.bins_after + self.bins_current)*self.bin_width))
        else:
            repr_string = "DataWindow(bins_before={}, bins_after={}, bins_current={}, bins_stride={})" \
            .format(self.bins_before, self.bins_after, self.bins_current, self.bins_stride)
        return repr_string

    def apply(self, X, T=None, lengths=None, flatten=False):
        """
        Apply window specification to data in X.

        NOTE: this function is epoch-aware.

        WARNING: this function works in-core, and may use a lot of memory
                 to represent the unwrapped (windowed) data. If you have
                 a large dataset, using the streaming version may be better.

        Parameters
        ----------
        X : numpy 2d array of shape (n_samples, n_features)
                OR
            array-like of shape (n_epochs, ), each element of which is
            a numpy 2d array of shape (n_samples, n_features)
                OR
            nelpy.core.BinnedEventArray / BinnedSpikeTrainArray
                The number of spikes in each time bin for each neuron/unit.
        T : array-like of shape (n_samples,), optional (default=None)
                Timestamps / sample numbers corresponding to data in X.
        lengths : array-like, optional (default=None)
                Only used / allowed when X is a 2d numpy array, in which case
                sum(lengths) must equal n_samples.
                Array of lengths (in number of bins) for each contiguous segment
                in X.
        flatten : int, optional (default=False)
            Whether or not to flatten the output data.

        Returns
        -------
        Z : Windowed data of shape (n_samples, window_size, n_features).
            Note that n_samples in the output may not be the same as n_samples
            in the input, since window specifications can affect which and how
            many samples to return.
            When flatten is True, then Z has shape (n_samples, window_size*n_features).
        T : array-like of shape (n_samples,)
            Timestamps associated with data contained in Z.
        """

        X, T, lengths = self._tidy(X=X, T=T, lengths=lengths)
        z = []
        t = []
        for ii, jj in self._iter_from_X_lengths(X=X, lengths=lengths):
            x, tx = self._apply_contiguous(X[ii:jj], T[ii:jj], flatten=flatten)
            z.append(x)
            t.extend(tx)

        Z = np.vstack(z)
        T = np.array(t)

        return Z, T

    def _apply_contiguous(self, X, T=None, flatten=False):
        """
        Apply window specification to data in X.

        NOTE: this function works on a single epoch only (i.e. assumes data
              is contiguous).

        NOTE: instead of returning partial data (with NaNs filling the rest),
              we only return those bins (windows) whose specifications are wholly
              contained in the data, similar to how binning in nelpy only includes
              those bins that fit wholly in the data support.

        WARNING: this function works in-core, and may use a lot of memory
                 to represent the unwrapped (windowed) data. If you have
                 a large dataset, using the streaming version may be better.

        Parameters
        ----------
        X : numpy 2d array of shape (n_samples, n_features)
        T : array-like of shape (n_samples,), optional (default=None)
                Timestamps / sample numbers corresponding to data in X.
        flatten : int, optional (default=False)
            Whether or not to flatten the output data.

        Returns
        -------
        Z : Windowed data of shape (n_samples, window_size, n_features).
            Note that n_samples in the output may not be the same as n_samples
            in the input, since window specifications can affect which and how
            many samples to return.
            When flatten is True, then Z has shape (n_samples, window_size*n_features).
        T : array-like of shape (n_samples,)
            Timestamps associated with data contained in Z.
        """

        bins_before = self.bins_before
        bins_after = self.bins_after
        bins_current = self.bins_current
        stride = self.bins_stride

        n_samples, n_features = X.shape
        n_zamples = int(np.ceil((n_samples - bins_before - bins_after)/stride))

        Z = np.empty([n_zamples, bins_before + bins_after + bins_current, n_features])
        Z[:] = np.NaN

        frm_idx = 0
        curr_idx = bins_before

        for zz in range(n_zamples):
            if bins_current == 1:
                idx = np.arange(frm_idx, frm_idx+bins_before + bins_after + bins_current)
            else:
                idx = list(range(frm_idx, frm_idx+bins_before))
                idx.extend(list(range(frm_idx+bins_before+1, frm_idx+bins_before+1+bins_after)))

        #     print('{}  @ {}'.format(idx, curr_idx))

            Z[zz,:] = X[idx,:]
            curr_idx += stride
            frm_idx += stride

        if flatten:
            Z = Z.reshape(Z.shape[0],(Z.shape[1]*Z.shape[2]))

        if T is not None:
            t_idx = list(range(bins_before, n_samples-bins_after, stride))
            T = T[t_idx]

        return Z, T

    def stream(self, X, chunk_size=1, flatten=False):
        """Streaming window specification on data X.

        Q. Should this return a generator? Should it BE a generator? I think we
            should return an iterable?

        Examples
        --------
        >>> w = DataWindow()
        >>> ws = w.stream(X)
        >>> for x in ws:
                print(x)

        """
        X, T, lengths = self._tidy(X)
        return StreamingDataWindow(self, X=X, flatten=flatten)

    def _tidy(self, X, T=None, lengths=None):
        """Transform data into a tidy, standardized, minimalist form.

        NOTE: No windowing is present in tidy data; windowind is APPLIED
              to tidy data when using DataWindow.apply().

        Parameters
        ----------
        X : numpy 2d array of shape (n_samples, n_features)
                OR
            array-like of shape (n_epochs, ), each element of which is
            a numpy 2d array of shape (n_samples, n_features)
                OR
            nelpy.core.BinnedEventArray / BinnedSpikeTrainArray
                The number of spikes in each time bin for each neuron/unit.
        T : array-like of shape (n_samples,), optional (default=None)
                Timestamps / sample numbers corresponding to data in X.
        lengths : array-like, optional (default=None)
                Only used / allowed when X is a 2d numpy array, in which case
                sum(lengths) must equal n_samples.
                Array of lengths (in number of bins) for each contiguous segment
                in X.

        Returns
        -------
        tidyX : numpy 2d array of shape (n_samples, n_features)
            The number of spikes in each time bin for each neuron/unit.
        tidyT : array-like of shape (n_samples,)
            Timestamps / sample numbers corresponding to data in X.
        lengths : array-like
            Array of lengths (in number of bins) for each contiguous segment
            in tidyX.

        Examples
        --------

        X = np.zeros((20, 8))
        X = [np.zeros((20,50)), np.zeros((30, 50)), np.zeros((80, 50))]
        X = [np.zeros((20,50)), np.zeros((30, 50)), np.zeros((80, 30))]
        w = DataWindow(bin_width=0.02)

        X, T, lengths = w._tidy(X)
        X, T, lengths = w._tidy(X, T=np.arange(50))
        X, T, lengths = w._tidy(X, lengths=[20,5,10])
        """

        # here we should transform BSTs, numpy arrays, check for dimensions, etc
        if isinstance(X, core.BinnedEventArray):
            if self._bin_width is not None:
                if self._bin_width != X.ds:
                    raise ValueError('The DataWindow has ``bin_width``={}, whereas ``X.ds``={}.'
                                     .format(self._bin_width, X.ds))
            T = X.bin_centers
            lengths = X.lengths
            X = X.data.T

            return X, T, lengths

        try:
            x = X[0,0]
            if X.ndim != 2:
                raise ValueError('X is expected to be array-like with shape (n_samples, n_features).')
            n_samples, n_features = X.shape
            if lengths is not None:
                tot_length = np.sum(lengths)
                if tot_length != n_samples:
                    raise ValueError('The sum of ``lengths`` should equal ``n_samples``. [sum(lengths)={}; n_samples={}]'
                             .format(tot_length, n_samples))
        except (IndexError, TypeError):
            try:
                x = X[0]
                if x.ndim != 2:
                    raise ValueError('Each element of X is expected to be array-like with shape (n_samples, n_features).')
                if lengths is not None:
                    raise ValueError('``lengths`` should not be specified when the shape of X is (n_epochs,)')
                n_samples, n_features = x.shape
                lengths = []
                for x in X:
                    lengths.append(x.shape[0])
                    if x.ndim != 2:
                        raise ValueError('Each element of X is expected to be array-like with shape (n_samples, n_features).')
                    if x.shape[1] != n_features:
                        raise ValueError('Each element of X is expected to have the same number of features.')
                X = np.vstack(X)
            except (IndexError, TypeError):
                raise TypeError('Windowing of type {} not supported!'.format(str(type(X))))
        n_samples, n_features = X.shape
        if T is not None:
            assert len(T) == n_samples, "T must have the same number of elements as n_samples."
        else:
            if self._bin_width is not None:
                ds = self._bin_width
            else:
                ds = 1
            T = np.arange(n_samples)*ds + ds/2

        return X, T, lengths

    def _iter_from_X_lengths(self, X, lengths=None):
        """
        Helper function to iterate over contiguous segments of data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
                Feature matrix of individual samples.
                Typically the number of spikes in each time bin for each neuron.
        lengths : array-like of integers, shape (n_epochs, ), optional
                Lengths of the individual epochs in ``X``. The sum of
                these should be ``n_samples``.
                Array of lengths (in number of bins) for each contiguous segment
                in X.

        Returns
        -------
        start, end : indices of a contiguous segment in data, so that
                     segment = data[start:end]
        """

        if X.ndim != 2:
            raise ValueError('X is expected to be array-like with shape (n_samples, n_features).')

        n_samples = X.shape[0]

        if lengths is None:
            yield 0, n_samples
        else:
            end = np.cumsum(lengths).astype(np.int)

            if end[-1] != n_samples:
                raise ValueError('The sum of ``lengths`` should equal ``n_samples``. [sum(lengths)={}; n_samples={}]'
                                 .format(end[-1], n_samples))

            start = end - lengths

            for i in range(len(lengths)):
                yield start[i], end[i]

    @property
    def bins_before(self):
        return self._bins_before

    @bins_before.setter
    def bins_before(self, val):
        assert float(val).is_integer(), "``bins_before`` must be a non-negative integer!"
        assert val >= 0, "``bins_before`` must be a non-negative integer!"
        self._bins_before = int(val)

    @property
    def bins_after(self):
        return self._bins_after

    @bins_after.setter
    def bins_after(self, val):
        assert float(val).is_integer(), "``bins_after`` must be a non-negative integer!"
        assert val >= 0, "``bins_after`` must be a non-negative integer!"
        self._bins_after = int(val)

    @property
    def bins_current(self):
        return self._bins_current

    @bins_current.setter
    def bins_current(self, val):
        assert float(val).is_integer(), "``bins_current`` must be a either 1 or 0!"
        assert val in [0,1], "``bins_current`` must be a either 1 or 0!"
        self._bins_current = int(val)

    @property
    def bins_stride(self):
        return self._bins_stride

    @bins_stride.setter
    def bins_stride(self, val):
        assert float(val).is_integer(), "``bins_stride`` must be a non-negative integer!"
        assert val >= 0, "``bins_stride`` must be a non-negative integer!"
        self._bins_stride = int(val)

    @property
    def bin_width(self):
        return self._bin_width

    @bin_width.setter
    def bin_width(self, val):
        if val is not None:
            assert float(val) > 0, "``bin_width`` must be a non-negative number (float)!"
        self._bin_width = val

    @property
    def params(self):
        """Dictionary of DataWindow parameters.

        Can be used to instantiate a new DataWindow, for example:

        >>> w_new = DataWindow(**w.params)
        """
        params = {'bins_before'  : self.bins_before,
                  'bins_after'   : self.bins_after,
                  'bins_current' : self.bins_current,
                  'bins_stride'  : self.bins_stride,
                  'bin_width'    : self.bin_width
                 }
        return params


class StreamingDataWindow():
    """
    StreamingDataWindow

    StreamingDataWindow is an iterable with an associated data object.

    See https://hackmag.com/coding/lets-tame-data-streams-with-python/
    """

    def __init__(self, w, X, flatten=False):
        self._w = w
        self.X = X
        self._flatten = False

    def flatten(self, inplace=False):
        # what's the opposite of flatten?
        pass

    def __repr__(self):
        return 'StreamingDataWindow(\n\tw={},\n\tX={},\n\tflatten={})' \
                .format(str(self.w), str(self.X), str(self._flatten))# + str(self.w)

    def __iter__(self):
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        index = self._index
        # if index > self.n_intervals - 1:
        #     raise StopIteration

        self._index += 1

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, val):
        if not isinstance(val, DataWindow):
            raise TypeError('w must be a nelpy.preprocessing.DataWindow type!')
        else:
            self._w = val