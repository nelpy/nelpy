import numpy as np
import pytest

import nelpy as nel


class TestGaussianFilterCoverage:
    """Test various code paths in gaussian_filter function to increase coverage."""

    def test_gaussian_filter_analog_signal_array(self):
        """Test gaussian_filter with AnalogSignalArray."""
        # Create a simple ASA
        data = np.random.randn(2, 100)  # 2 signals
        fs = 100

        asa = nel.AnalogSignalArray(data, fs=fs)

        # Test basic gaussian filtering
        filtered = nel.utils.gaussian_filter(asa, sigma=0.01, truncate=4)

        assert isinstance(filtered, nel.AnalogSignalArray)
        assert filtered.data.shape == asa.data.shape
        assert filtered.fs == asa.fs

    def test_gaussian_filter_binned_spike_train(self):
        """Test gaussian_filter with BinnedSpikeTrainArray."""
        # Create mock spike data
        spike_times = [np.array([0.1, 0.2, 0.5]), np.array([0.15, 0.3, 0.6])]
        st = nel.SpikeTrainArray(spike_times, support=nel.EpochArray([[0, 1]]))

        # Bin the spike train
        bst = st.bin(ds=0.01)

        # Test gaussian filtering on BST - this should return a BST
        filtered = nel.utils.gaussian_filter(bst, sigma=0.02, truncate=4)

        assert isinstance(filtered, nel.BinnedSpikeTrainArray)
        assert filtered.data.shape == bst.data.shape

    def test_gaussian_filter_within_intervals(self):
        """Test gaussian_filter with within_intervals=True."""
        # Create ASA with multiple epochs
        data = np.random.randn(1, 200)
        fs = 100
        support = nel.EpochArray([[0, 1], [2, 3]])  # Two separate intervals

        asa = nel.AnalogSignalArray(data, fs=fs, support=support)

        # Test filtering within intervals
        filtered = nel.utils.gaussian_filter(asa, sigma=0.01, within_intervals=True)

        assert isinstance(filtered, nel.AnalogSignalArray)
        assert filtered.data.shape == asa.data.shape

    def test_gaussian_filter_error_cases(self):
        """Test error cases in gaussian_filter."""
        # Test unsupported object type
        with pytest.raises(
            NotImplementedError, match="gaussian_filter for .* is not yet supported"
        ):
            nel.utils.gaussian_filter([1, 2, 3], sigma=0.1)

    def test_get_mua_with_gaussian_smoothing(self):
        """Test get_mua function which uses gaussian_filter internally."""
        # Create spike train data
        spike_times = [
            np.array([0.1, 0.2, 0.3, 0.5, 0.7]),
            np.array([0.15, 0.25, 0.4, 0.6, 0.8]),
        ]
        st = nel.SpikeTrainArray(spike_times, support=nel.EpochArray([[0, 1]]))

        # Test MUA computation with smoothing
        mua = nel.utils.get_mua(st, ds=0.01, sigma=0.02, truncate=4)

        assert isinstance(mua, nel.AnalogSignalArray)
        assert mua.data.shape[0] == 1  # MUA is single signal

    def test_get_mua_no_smoothing(self):
        """Test get_mua with sigma=0 (no smoothing)."""
        spike_times = [np.array([0.1, 0.2, 0.5])]
        st = nel.SpikeTrainArray(spike_times, support=nel.EpochArray([[0, 1]]))

        # Test with no smoothing
        mua = nel.utils.get_mua(st, ds=0.01, sigma=0, truncate=4)

        assert isinstance(mua, nel.AnalogSignalArray)

    def test_signal_envelope_1d_smoothing(self):
        """Test signal_envelope_1d function which uses gaussian filtering."""
        # Create test signal
        fs = 1000
        t = np.linspace(0, 1, fs)
        # Create a modulated signal
        signal = np.sin(2 * np.pi * 10 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))

        # Test envelope detection with smoothing - should return 2D array
        envelope = nel.utils.signal_envelope_1d(signal, sigma=0.01, fs=fs)

        assert envelope.shape == (1, signal.shape[0])  # Returns 2D array
        assert np.all(envelope >= 0)  # Envelope should be positive

    def test_signal_envelope_1d_with_asa(self):
        """Test signal_envelope_1d with AnalogSignalArray input."""
        fs = 1000
        t = np.linspace(0, 1, fs)
        signal = np.sin(2 * np.pi * 10 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t))

        # Create ASA
        asa = nel.AnalogSignalArray(signal[np.newaxis, :], fs=fs)

        # Test envelope with ASA input
        envelope_asa = nel.utils.signal_envelope_1d(asa, sigma=0.01)

        assert isinstance(envelope_asa, nel.AnalogSignalArray)
        assert envelope_asa.fs == fs

    def test_gaussian_filter_mode_parameters(self):
        """Test gaussian_filter with different mode parameters."""
        data = np.random.randn(1, 100)
        asa = nel.AnalogSignalArray(data, fs=100)

        # Test different modes
        for mode in ["reflect", "constant", "nearest", "wrap"]:
            filtered = nel.utils.gaussian_filter(asa, sigma=0.01, mode=mode)
            assert isinstance(filtered, nel.AnalogSignalArray)

    def test_gaussian_filter_cval_parameter(self):
        """Test gaussian_filter with cval parameter."""
        data = np.random.randn(1, 100)
        asa = nel.AnalogSignalArray(data, fs=100)

        # Test with constant mode and cval
        filtered = nel.utils.gaussian_filter(asa, sigma=0.01, mode="constant", cval=0.5)
        assert isinstance(filtered, nel.AnalogSignalArray)

    def test_deprecated_bw_parameter(self):
        """Test deprecated 'bw' parameter in gaussian_filter."""
        data = np.random.randn(1, 100)
        asa = nel.AnalogSignalArray(data, fs=100)

        # Test deprecated bw parameter - should emit DeprecationWarning
        with pytest.warns(DeprecationWarning, match="deprecated"):
            filtered = nel.utils.gaussian_filter(asa, sigma=0.01, bw=4)
            assert isinstance(filtered, nel.AnalogSignalArray)


class TestMultiDimensionalSmoothing:
    """Test multi-dimensional smoothing functionality for comprehensive coverage."""

    def test_3d_asa_smoothing(self):
        """Test smoothing of 3D AnalogSignalArray."""
        # Create 3D data
        data = np.random.randn(3, 100)  # 3 signals, 100 time points
        asa = nel.AnalogSignalArray(data, fs=100)

        filtered = nel.utils.gaussian_filter(asa, sigma=0.01)
        assert filtered.data.shape == data.shape

    def test_smoothing_with_multiple_epochs(self):
        """Test smoothing across multiple epochs."""
        data = np.random.randn(2, 200)
        support = nel.EpochArray([[0, 1], [1.5, 2.5]])
        asa = nel.AnalogSignalArray(data, fs=100, support=support)

        # Test smoothing within intervals - data shape may change due to support
        filtered = nel.utils.gaussian_filter(asa, sigma=0.01, within_intervals=True)
        assert filtered.data.shape[0] == data.shape[0]  # Same number of signals
        assert filtered.n_epochs == 2


class TestEdgeCasesAndErrorPaths:
    """Test edge cases and error paths for maximum coverage."""

    def test_gaussian_filter_with_fs_parameter(self):
        """Test gaussian_filter with explicit fs parameter."""
        data = np.random.randn(1, 100)
        asa = nel.AnalogSignalArray(data, fs=100)

        # Test with explicit fs parameter
        filtered = nel.utils.gaussian_filter(asa, sigma=0.01, fs=200)  # Different fs
        assert isinstance(filtered, nel.AnalogSignalArray)

    def test_single_sample_signal(self):
        """Test with single sample signal."""
        data = np.array([[1.0, 2.0]])  # Single row, two samples minimum
        asa = nel.AnalogSignalArray(data, fs=100)

        filtered = nel.utils.gaussian_filter(asa, sigma=0.01)
        assert filtered.data.shape == (1, 2)

    def test_zero_sigma_smoothing(self):
        """Test smoothing with zero sigma."""
        data = np.random.randn(2, 50)
        asa = nel.AnalogSignalArray(data, fs=100)

        # Zero sigma should return nearly identical result
        filtered = nel.utils.gaussian_filter(asa, sigma=0.0)
        np.testing.assert_allclose(filtered.data, asa.data, rtol=1e-10)

    def test_very_large_sigma(self):
        """Test smoothing with very large sigma."""
        data = np.random.randn(1, 100)
        asa = nel.AnalogSignalArray(data, fs=100)

        # Large sigma should heavily smooth the signal
        filtered = nel.utils.gaussian_filter(
            asa, sigma=1.0
        )  # Very large relative to signal
        assert isinstance(filtered, nel.AnalogSignalArray)

        # The variance should be reduced due to smoothing
        assert np.var(filtered.data) < np.var(asa.data)
