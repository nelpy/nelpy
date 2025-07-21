import numpy as np
import pytest

import nelpy as nel
from nelpy.analysis import replay


class TestLinregressFunctions:
    """Test linear regression functions."""

    def test_linregress_array_basic(self):
        """Test linregress_array with basic linear data."""
        # Create a simple linear posterior
        posterior = np.zeros((10, 5))  # 10 positions, 5 time bins
        # Set mode path to be linear
        for i in range(5):
            posterior[i, i] = 1.0  # Diagonal pattern

        slope, intercept, r2 = replay.linregress_array(posterior)

        assert isinstance(slope, float)
        assert isinstance(intercept, float)
        assert isinstance(r2, float)
        assert not np.isnan(slope)
        assert not np.isnan(intercept)
        assert not np.isnan(r2)

    def test_linregress_array_all_nan(self):
        """Test linregress_array with all NaN data."""
        posterior = np.full((5, 5), np.nan)

        slope, intercept, r2 = replay.linregress_array(posterior)

        assert np.isnan(slope)
        assert np.isnan(intercept)
        assert np.isnan(r2)

    @pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide")
    @pytest.mark.filterwarnings("ignore:invalid value encountered in sqrt")
    def test_linregress_array_single_bin(self):
        """Test linregress_array with single time bin."""
        posterior = np.zeros((10, 1))
        posterior[5, 0] = 1.0  # Single peak

        slope, intercept, r2 = replay.linregress_array(posterior)

        # Should return NaN for slope and intercept, but r2 might be 0 for single point
        assert np.isnan(slope)
        assert np.isnan(intercept)
        # r2 can be 0.0 for single point regression
        assert r2 == 0.0 or np.isnan(r2)

    def test_linregress_bst_basic(self):
        """Test linregress_bst with basic data."""
        # Create a simple tuning curve with 2 units to match the binned spike train
        positions = np.linspace(0, 10, 11)
        rates = np.exp(-((positions - 5) ** 2) / 2)  # Gaussian tuning
        # Create tuning curve with 2 units
        ratemap_2d = np.vstack([rates, rates])  # 2 units, same tuning
        tuningcurve = nel.TuningCurve1D(ratemap=ratemap_2d)

        # Create a simple binned spike train
        spike_times = [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]]
        bst = nel.BinnedSpikeTrainArray(nel.SpikeTrainArray(spike_times), ds=0.1)

        slopes, intercepts, r2values = replay.linregress_bst(bst, tuningcurve)

        assert isinstance(slopes, np.ndarray)
        assert isinstance(intercepts, np.ndarray)
        assert isinstance(r2values, np.ndarray)
        assert len(slopes) == bst.n_epochs
        assert len(intercepts) == bst.n_epochs
        assert len(r2values) == bst.n_epochs


class TestShufflingFunctions:
    """Test shuffling functions."""

    def test_time_swap_array_basic(self):
        """Test time_swap_array with basic data."""
        posterior = np.random.rand(10, 5)
        original_sum = np.sum(posterior)

        swapped = replay.time_swap_array(posterior)

        assert swapped.shape == posterior.shape
        assert np.sum(swapped) == pytest.approx(original_sum, rel=1e-10)
        # Should have same values but different arrangement
        assert not np.array_equal(swapped, posterior)

    def test_time_swap_array_single_column(self):
        """Test time_swap_array with single time column."""
        posterior = np.random.rand(10, 1)

        swapped = replay.time_swap_array(posterior)

        assert swapped.shape == posterior.shape
        assert np.array_equal(
            swapped, posterior
        )  # Should be identical for single column

    def test_column_cycle_array_basic(self):
        """Test column_cycle_array with basic data."""
        posterior = np.random.rand(10, 5)
        original_sum = np.sum(posterior)

        cycled = replay.column_cycle_array(posterior)

        assert cycled.shape == posterior.shape
        assert np.sum(cycled) == pytest.approx(original_sum, rel=1e-10)
        # Should have same values but cycled arrangement
        assert not np.array_equal(cycled, posterior)

    def test_column_cycle_array_with_amount(self):
        """Test column_cycle_array with specific amount."""
        posterior = np.random.rand(10, 5)
        amount = np.array([2, 1, 3, 0, 4])  # Array of amounts for each column

        cycled = replay.column_cycle_array(posterior, amt=amount)

        assert cycled.shape == posterior.shape
        # Check that columns are cycled by the specified amounts
        for i, amt_val in enumerate(amount):
            expected = np.roll(posterior[:, i], amt_val)
            np.testing.assert_array_equal(cycled[:, i], expected)


class TestTrajectoryScoring:
    """Test trajectory scoring functions."""

    def test_trajectory_score_array_basic(self):
        """Test trajectory_score_array with basic data."""
        posterior = np.random.rand(10, 5)
        slope = 1.0
        intercept = 0.0

        score = replay.trajectory_score_array(
            posterior, slope=slope, intercept=intercept
        )

        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_trajectory_score_array_with_weights(self):
        """Test trajectory_score_array with weights."""
        posterior = np.random.rand(10, 5)
        slope = 1.0
        intercept = 0.0
        weights = np.ones(5)  # Equal weights

        score = replay.trajectory_score_array(
            posterior, slope=slope, intercept=intercept, weights=weights
        )

        assert isinstance(score, float)
        assert not np.isnan(score)

    def test_trajectory_score_array_normalize(self):
        """Test trajectory_score_array with normalization."""
        posterior = np.random.rand(10, 5)
        slope = 1.0
        intercept = 0.0

        score_normalized = replay.trajectory_score_array(
            posterior, slope=slope, intercept=intercept, normalize=True
        )
        score_not_normalized = replay.trajectory_score_array(
            posterior, slope=slope, intercept=intercept, normalize=False
        )

        assert isinstance(score_normalized, float)
        assert isinstance(score_not_normalized, float)
        # Normalized score should be different (but not necessarily smaller)
        assert score_normalized != score_not_normalized

    def test_trajectory_score_bst_basic(self):
        """Test trajectory_score_bst with basic data."""
        # Create a simple tuning curve with 2 units to match the binned spike train
        positions = np.linspace(0, 10, 11)
        rates = np.exp(-((positions - 5) ** 2) / 2)
        ratemap_2d = np.vstack([rates, rates])  # 2 units, same tuning
        tuningcurve = nel.TuningCurve1D(ratemap=ratemap_2d)

        # Create a simple binned spike train
        spike_times = [[0.1, 0.2, 0.3], [0.6, 0.7, 0.8]]
        bst = nel.BinnedSpikeTrainArray(nel.SpikeTrainArray(spike_times), ds=0.1)

        scores, scores_time_swap, scores_col_cycle = replay.trajectory_score_bst(
            bst, tuningcurve, n_shuffles=10
        )

        assert isinstance(scores, np.ndarray)
        assert isinstance(scores_time_swap, np.ndarray)
        assert isinstance(scores_col_cycle, np.ndarray)
        assert len(scores) == bst.n_epochs
        assert scores_time_swap.shape[1] == bst.n_epochs
        assert scores_col_cycle.shape[1] == bst.n_epochs


class TestSignificantEvents:
    """Test significant events detection."""

    def test_get_significant_events_basic(self):
        """Test get_significant_events with basic data."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        shuffled_scores = np.random.rand(100, 5)  # 100 shuffles, 5 events

        sig_events, pvalues = replay.get_significant_events(
            scores, shuffled_scores, q=95
        )

        assert isinstance(sig_events, np.ndarray)
        assert isinstance(pvalues, np.ndarray)
        assert len(pvalues) == len(scores)
        assert all(0 <= p <= 1 for p in pvalues)

    def test_get_significant_events_all_significant(self):
        """Test get_significant_events when all events are significant."""
        scores = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # All very high
        shuffled_scores = np.random.rand(100, 5) * 5  # All low

        sig_events, pvalues = replay.get_significant_events(
            scores, shuffled_scores, q=95
        )

        assert len(sig_events) == len(scores)  # All should be significant
        assert all(p < 0.05 for p in pvalues)  # All p-values should be small

    def test_get_significant_events_none_significant(self):
        """Test get_significant_events when no events are significant."""
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])  # All low
        shuffled_scores = np.random.rand(100, 5) * 10  # All high

        sig_events, pvalues = replay.get_significant_events(
            scores, shuffled_scores, q=95
        )

        assert len(sig_events) == 0  # None should be significant
        assert all(p > 0.05 for p in pvalues)  # All p-values should be large

    def test_get_significant_events_single_event(self):
        """Test get_significant_events with single event."""
        scores = np.array([5.0])
        shuffled_scores = np.random.rand(100, 1)

        sig_events, pvalues = replay.get_significant_events(
            scores, shuffled_scores, q=95
        )

        assert isinstance(sig_events, np.ndarray)
        assert isinstance(pvalues, np.ndarray)
        assert len(pvalues) == 1


class TestConsecutiveBins:
    """Test consecutive bins detection."""

    def test_three_consecutive_bins_above_q_basic(self):
        """Test three_consecutive_bins_above_q with basic data."""
        pvals = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95, 0.8, 0.7, 0.6])
        lengths = np.array([9])
        q = 0.75

        result = replay.three_consecutive_bins_above_q(
            pvals, lengths, q=q, n_consecutive=3
        )

        assert isinstance(result, np.ndarray)
        assert len(result) <= len(pvals)

    def test_three_consecutive_bins_above_q_no_consecutive(self):
        """Test three_consecutive_bins_above_q with no consecutive bins above threshold."""
        pvals = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        lengths = np.array([9])
        q = 95  # Very high threshold (function uses 100 * (1 - pvals) > q)

        result = replay.three_consecutive_bins_above_q(
            pvals, lengths, q=q, n_consecutive=3
        )

        assert len(result) == 0  # No consecutive bins above threshold

    def test_three_consecutive_bins_above_q_multiple_epochs(self):
        """Test three_consecutive_bins_above_q with multiple epochs."""
        pvals = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 0.95, 0.1, 0.2, 0.3])
        lengths = np.array([6, 3])  # Two epochs

        result = replay.three_consecutive_bins_above_q(
            pvals, lengths, q=0.75, n_consecutive=3
        )

        assert isinstance(result, np.ndarray)


class TestTransitionMatrixShuffling:
    """Test transition matrix shuffling functions."""

    def test_shuffle_transmat_basic(self):
        """Test shuffle_transmat with basic transition matrix."""
        transmat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])

        shuffled = replay.shuffle_transmat(transmat)

        assert shuffled.shape == transmat.shape
        # Should preserve row sums (stochasticity)
        np.testing.assert_allclose(np.sum(shuffled, axis=1), np.sum(transmat, axis=1))

    def test_shuffle_transmat_identity(self):
        """Test shuffle_transmat with identity matrix."""
        transmat = np.eye(3)

        shuffled = replay.shuffle_transmat(transmat)

        assert shuffled.shape == transmat.shape
        # Should preserve row sums
        np.testing.assert_allclose(np.sum(shuffled, axis=1), np.sum(transmat, axis=1))

    def test_shuffle_transmat_Kourosh_breaks_stochasticity(self):
        """Test shuffle_transmat_Kourosh_breaks_stochasticity."""
        transmat = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7]])

        shuffled = replay.shuffle_transmat_Kourosh_breaks_stochasticity(transmat)

        assert shuffled.shape == transmat.shape
        # This function may break stochasticity, so we don't check row sums


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_linregress_array_empty(self):
        """Test linregress_array with empty array."""
        posterior = np.array([])

        with pytest.raises(ValueError):
            replay.linregress_array(posterior)

    def test_time_swap_array_empty(self):
        """Test time_swap_array with empty array."""
        posterior = np.array([])

        with pytest.raises(ValueError):
            replay.time_swap_array(posterior)

    def test_get_significant_events_empty_scores(self):
        """Test get_significant_events with empty scores."""
        scores = np.array([])
        shuffled_scores = np.random.rand(100, 0)

        sig_events, pvalues = replay.get_significant_events(
            scores, shuffled_scores, q=95
        )

        assert len(sig_events) == 0
        assert len(pvalues) == 0

    def test_three_consecutive_bins_above_q_empty(self):
        """Test three_consecutive_bins_above_q with empty data."""
        pvals = np.array([])
        lengths = np.array([])

        result = replay.three_consecutive_bins_above_q(
            pvals, lengths, q=0.75, n_consecutive=3
        )

        assert len(result) == 0


class TestInputValidation:
    """Test input validation for various functions."""

    def test_linregress_array_invalid_input(self):
        """Test linregress_array with invalid input types."""
        with pytest.raises(AttributeError):
            replay.linregress_array("not an array")

        with pytest.raises(AttributeError):
            replay.linregress_array(None)

    def test_time_swap_array_invalid_input(self):
        """Test time_swap_array with invalid input types."""
        with pytest.raises(AttributeError):
            replay.time_swap_array("not an array")

        with pytest.raises(AttributeError):
            replay.time_swap_array(None)

    def test_get_significant_events_invalid_percentile(self):
        """Test get_significant_events with invalid percentile."""
        scores = np.array([1.0, 2.0, 3.0])
        shuffled_scores = np.random.rand(100, 3)

        with pytest.raises(ValueError):
            replay.get_significant_events(
                scores, shuffled_scores, q=150
            )  # Invalid percentile

        with pytest.raises(ValueError):
            replay.get_significant_events(
                scores, shuffled_scores, q=-10
            )  # Invalid percentile
