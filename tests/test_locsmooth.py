import numpy as np
import pytest

import nelpy as nel


def _reference_running_line(data, window_samples, step_samples):
    """Straightforward running-line smoother used to check the compiled kernel."""
    n_samples = data.shape[1]
    window_samples = min(window_samples, n_samples)
    starts = list(range(0, n_samples - window_samples + 1, step_samples))
    if starts[-1] != n_samples - window_samples:
        starts.append(n_samples - window_samples)

    positions = np.arange(1, window_samples + 1, dtype=float)
    centered = positions - positions.mean()
    scaled = (positions - window_samples / 2) / (window_samples / 2)
    weights = (1 - np.abs(scaled) ** 3) ** 3

    smoothed = np.zeros(data.shape, dtype=np.result_type(data.dtype, float))
    normalization = np.zeros(n_samples)
    fallback = np.empty_like(smoothed)
    for start in starts:
        segment = data[:, start : start + window_samples]
        slope = segment @ centered / np.dot(centered, centered)
        fitted = segment.mean(axis=1, keepdims=True) + slope[:, None] * centered
        smoothed[:, start : start + window_samples] += fitted * weights
        normalization[start : start + window_samples] += weights
        fallback[:, start : start + window_samples] = fitted

    valid = normalization > 0
    smoothed[:, valid] /= normalization[valid]
    smoothed[:, ~valid] = fallback[:, ~valid]
    return smoothed


class TestLocsmooth:
    @pytest.mark.parametrize(
        ("n_samples", "window", "step"),
        [
            (n_samples, window, step)
            for n_samples in (2, 3, 7, 50)
            for window in (2, 3, 6, 25)
            for step in sorted({1, max(1, window // 2), window - 1})
            if step < window
        ],
    )
    @pytest.mark.parametrize("complex_data", [False, True])
    def test_matches_reference_running_line(
        self, n_samples, window, step, complex_data
    ):
        rng = np.random.default_rng(n_samples * 100 + window * 10 + step)
        data = rng.normal(size=(3, n_samples))
        if complex_data:
            data = data + 1j * rng.normal(size=(3, n_samples))
        asa = nel.AnalogSignalArray(data, fs=1)

        smoothed = nel.utils.locsmooth(asa, window=window, step=step)

        np.testing.assert_allclose(
            smoothed.data,
            _reference_running_line(data, window, step),
            rtol=1e-10,
            atol=1e-12,
        )

    def test_running_line_reference_values_and_metadata(self):
        data = np.array(
            [
                0,
                1,
                4,
                2,
                5,
                3,
                7,
                6,
                8,
                5,
                9,
                12,
                10,
                13,
                11,
                14,
                17,
                15,
                18,
                16,
            ],
            dtype=float,
        )
        asa = nel.AnalogSignalArray(data, fs=1)

        smoothed = nel.utils.locsmooth(asa, window=6, step=3)

        expected = np.array(
            [
                0.714285714286,
                1.428571428571,
                2.142857142857,
                2.7635748877,
                3.578279010397,
                4.638095238095,
                5.673851130794,
                6.574269539205,
                7.390476190476,
                7.958059380085,
                8.531664376821,
                9.428571428571,
                10.544694865874,
                11.645781324603,
                12.73729327463,
                13.785714285714,
                14.765112615764,
                15.666666666667,
                16.666666666667,
                17.666666666667,
            ]
        )

        np.testing.assert_allclose(smoothed.data[0], expected)
        np.testing.assert_array_equal(smoothed.abscissa_vals, asa.abscissa_vals)
        np.testing.assert_array_equal(smoothed.support.data, asa.support.data)
        assert smoothed.fs == asa.fs
        np.testing.assert_array_equal(asa.data[0], data)

    def test_preserves_linear_and_complex_signals(self):
        samples = np.arange(20, dtype=float)
        asa = nel.AnalogSignalArray(np.vstack((samples, 3 * samples - 2)), fs=10)

        smoothed = nel.utils.locsmooth(asa, window=0.6)

        np.testing.assert_allclose(smoothed.data, asa.data)

        complex_asa = nel.AnalogSignalArray(2 * samples + 1j * (3 * samples - 2), fs=10)
        complex_smoothed = nel.utils.locsmooth(complex_asa, window=0.6)
        np.testing.assert_allclose(complex_smoothed.data, complex_asa.data)
        assert np.iscomplexobj(complex_smoothed.data)

    def test_uses_matlab_half_up_rounding_for_window_samples(self):
        asa = nel.AnalogSignalArray(np.arange(10, dtype=float) ** 2, fs=10)

        half_up = nel.utils.locsmooth(asa, window=0.25, step=0.1)
        three_samples = nel.utils.locsmooth(asa, window=0.3, step=0.1)

        np.testing.assert_allclose(half_up.data, three_samples.data)

    def test_smooths_each_support_interval_independently(self):
        timestamps = np.r_[np.arange(0, 1, 0.1), np.arange(2, 3, 0.1)]
        asa = nel.AnalogSignalArray(
            np.r_[np.zeros(10), np.full(10, 100.0)],
            timestamps=timestamps,
            fs=10,
            support=nel.EpochArray([[0, 1], [2, 3]]),
        )

        smoothed = nel.utils.locsmooth(asa, window=0.6)

        np.testing.assert_allclose(smoothed.data[0, :10], 0)
        np.testing.assert_allclose(smoothed.data[0, 10:], 100)
        np.testing.assert_array_equal(smoothed.lengths, asa.lengths)

    def test_inplace_updates_and_returns_input(self):
        asa = nel.AnalogSignalArray([0, 3, 0, 4, 1, 5, 2, 6], fs=10)
        original = asa.data.copy()

        returned = nel.utils.locsmooth(asa, window=0.4, inplace=True)

        assert returned is asa
        assert not np.array_equal(asa.data, original)

    @pytest.mark.parametrize(
        ("window", "step", "error", "match"),
        [
            (0, None, ValueError, "positive finite"),
            (0.1, None, ValueError, "at least two samples"),
            (0.4, 0.4, ValueError, "shorter than window"),
            (0.4, 0, ValueError, "positive finite"),
        ],
    )
    def test_rejects_invalid_window_parameters(self, window, step, error, match):
        asa = nel.AnalogSignalArray(np.arange(10, dtype=float), fs=10)

        with pytest.raises(error, match=match):
            nel.utils.locsmooth(asa, window=window, step=step)

    def test_rejects_unsupported_or_invalid_data(self):
        with pytest.raises(NotImplementedError, match="not yet supported"):
            nel.utils.locsmooth([1, 2, 3], window=1)

        asa = nel.AnalogSignalArray([0, np.nan, 1, 2], fs=10)
        with pytest.raises(ValueError, match="NaN or infinite"):
            nel.utils.locsmooth(asa, window=0.2)

    def test_rejects_support_interval_with_one_sample(self):
        asa = nel.AnalogSignalArray(
            [0, 1, 2, 3],
            timestamps=[0, 0.1, 0.2, 1],
            fs=10,
            support=nel.EpochArray([[0, 0.3], [1, 1.1]]),
        )

        with pytest.raises(ValueError, match="per support interval"):
            nel.utils.locsmooth(asa, window=0.2)
