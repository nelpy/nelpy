import numpy as np
import pytest
from scipy.signal import savgol_filter as scipy_savgol_filter

import nelpy as nel


class TestSavgolFilter:
    def test_matches_scipy_on_one_interval_and_keeps_metadata(self):
        rng = np.random.default_rng(0)
        data = rng.normal(size=(2, 400))
        asa = nel.AnalogSignalArray(data, fs=200)

        filtered = nel.utils.savgol_filter(asa, window=0.25, polyorder=3)

        np.testing.assert_allclose(
            filtered.data, scipy_savgol_filter(data, 51, 3, axis=-1, mode="interp")
        )
        np.testing.assert_array_equal(filtered.abscissa_vals, asa.abscissa_vals)
        np.testing.assert_array_equal(filtered.support.data, asa.support.data)
        assert filtered.fs == asa.fs
        np.testing.assert_array_equal(asa.data, data)

    def test_even_window_samples_are_increased_to_odd(self):
        asa = nel.AnalogSignalArray(np.random.default_rng(1).normal(size=100), fs=10)

        even = nel.utils.savgol_filter(asa, window=0.6)
        odd = nel.utils.savgol_filter(asa, window=0.7)

        np.testing.assert_allclose(even.data, odd.data)

    def test_preserves_polynomials_up_to_polyorder(self):
        time = np.arange(0, 2, 0.01)
        asa = nel.AnalogSignalArray(1 - 2 * time + 3 * time**2, fs=100)

        filtered = nel.utils.savgol_filter(asa, window=0.31, polyorder=2)

        np.testing.assert_allclose(filtered.data, asa.data)

    def test_derivatives_are_per_base_unit(self):
        time = np.arange(0, 2, 0.01)
        asa = nel.AnalogSignalArray(
            np.vstack((1 + 3 * time, time**2)), timestamps=time, fs=100
        )

        first = nel.utils.savgol_filter(asa, window=0.21, polyorder=2, deriv=1)
        second = nel.utils.savgol_filter(asa, window=0.21, polyorder=2, deriv=2)

        np.testing.assert_allclose(first.data[0], 3)
        np.testing.assert_allclose(first.data[1], 2 * time, atol=1e-10)
        np.testing.assert_allclose(second.data[1], 2)

    def test_filters_each_support_interval_independently(self):
        rng = np.random.default_rng(2)
        timestamps = np.r_[np.arange(0, 1, 0.1), np.arange(2, 3, 0.1)]
        data = np.r_[rng.normal(size=10), 100 + rng.normal(size=10)]
        asa = nel.AnalogSignalArray(
            data,
            timestamps=timestamps,
            fs=10,
            support=nel.EpochArray([[0, 1], [2, 3]]),
        )

        filtered = nel.utils.savgol_filter(asa, window=0.5, polyorder=2)

        for start, stop in [(0, 10), (10, 20)]:
            np.testing.assert_allclose(
                filtered.data[0, start:stop],
                scipy_savgol_filter(data[start:stop], 5, 2, mode="interp"),
            )
        np.testing.assert_array_equal(filtered.lengths, asa.lengths)

    def test_window_longer_than_interval_uses_largest_odd_length(self):
        data = np.random.default_rng(3).normal(size=8)
        asa = nel.AnalogSignalArray(data, fs=10)

        filtered = nel.utils.savgol_filter(asa, window=2, polyorder=2)

        np.testing.assert_allclose(
            filtered.data[0], scipy_savgol_filter(data, 7, 2, mode="interp")
        )

    def test_filters_complex_real_and_imaginary_parts(self):
        rng = np.random.default_rng(4)
        real, imag = rng.normal(size=(2, 60))
        asa = nel.AnalogSignalArray(real + 1j * imag, fs=10)

        filtered = nel.utils.savgol_filter(asa, window=0.9, polyorder=2)

        assert np.iscomplexobj(filtered.data)
        np.testing.assert_allclose(
            filtered.data[0],
            scipy_savgol_filter(real, 9, 2, mode="interp")
            + 1j * scipy_savgol_filter(imag, 9, 2, mode="interp"),
        )

    def test_inplace_updates_and_returns_input(self):
        asa = nel.AnalogSignalArray([0, 3, 0, 4, 1, 5, 2, 6], fs=10)
        original = asa.data.copy()

        returned = nel.utils.savgol_filter(asa, window=0.5, inplace=True)

        assert returned is asa
        assert not np.array_equal(asa.data, original)

    @pytest.mark.parametrize(
        ("kwargs", "error", "match"),
        [
            ({"window": 0}, ValueError, "positive finite"),
            ({"window": "wide"}, TypeError, "positive scalar"),
            ({"window": 0.3, "polyorder": 3}, ValueError, "more samples than"),
            ({"window": 0.5, "polyorder": 2.5}, TypeError, "integers"),
            ({"window": 0.5, "polyorder": -1}, ValueError, "non-negative"),
            ({"window": 0.5, "polyorder": 1, "deriv": 2}, ValueError, "exceed"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, error, match):
        asa = nel.AnalogSignalArray(np.arange(20, dtype=float), fs=10)

        with pytest.raises(error, match=match):
            nel.utils.savgol_filter(asa, **kwargs)

    def test_rejects_unsupported_or_invalid_data(self):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            nel.utils.savgol_filter([1, 2, 3], window=1)

        asa = nel.AnalogSignalArray([0, np.nan, 1, 2, 3], fs=10)
        with pytest.raises(ValueError, match="NaN or infinite"):
            nel.utils.savgol_filter(asa, window=0.3)

    def test_rejects_support_interval_too_short_for_polyorder(self):
        asa = nel.AnalogSignalArray(
            np.arange(12, dtype=float),
            timestamps=np.r_[np.arange(0, 1, 0.1), 2, 2.1],
            fs=10,
            support=nel.EpochArray([[0, 1], [2, 2.2]]),
        )

        with pytest.raises(ValueError, match="each support interval"):
            nel.utils.savgol_filter(asa, window=0.5, polyorder=2)
