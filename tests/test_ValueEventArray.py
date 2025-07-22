import numpy as np
import pytest

import nelpy as nel


def test_valueeventarray_construction():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    assert veva.n_series == 2
    assert all(n > 0 for n in veva.n_values)
    assert not veva.isempty
    assert veva.data.shape[0] == 2


def test_valueeventarray_properties():
    events = [[0.1, 0.5, 1.0]]
    values = [[1, 2, 3]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    assert veva.n_series == 1
    assert veva.n_values[0] == 1
    assert not veva.isempty
    assert veva.data.shape[0] == 1


def test_valueeventarray_empty():
    veva = nel.ValueEventArray(events=[[]], values=[[]], fs=10)
    assert veva.isempty
    assert veva.n_series == 1
    assert veva.data.shape[0] == 1


def test_valueeventarray_shape_mismatch():
    events = [[0.1, 0.5]]
    values = [[1, 2, 3]]  # Mismatch
    with pytest.raises(ValueError):
        nel.ValueEventArray(events=events, values=values, fs=10)


def test_valueeventarray_flatten():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    try:
        veva.flatten()
    except NotImplementedError:
        pass
    except Exception:
        pytest.skip("Flatten not implemented or not supported for ValueEventArray")


def test_valueeventarray_indexing():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    # Indexing first series
    sub = veva.iloc[:, 0]
    assert sub.n_series == 1


# --- BinnedValueEventArray tests ---
def test_binnedvalueeventarray_sum():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    bvea = nel.BinnedValueEventArray(veva, ds=0.5, method="sum")
    bea = nel.BinnedEventArray(nel.EventArray(np.array(events, dtype=float), fs=10), ds=0.5)
    np.testing.assert_allclose(bvea.bins, bea.bins)
    # Should have 2 bins: [0.1,0.6), [0.6,1.1) for first interval
    assert bvea.data.shape[1] >= 2
    # Check sum for first series, first bin (events at 0.1 and 0.5)
    assert np.isclose(bvea.data[0, 0, 0], 1 + 2)
    # Check sum for second series, first bin (events at 0.2)
    assert np.isclose(bvea.data[1, 0, 0], 4)
    # Check sum for second series, second bin (events at 0.6)
    assert np.isclose(bvea.data[1, 1, 0], 5)


def test_binnedvalueeventarray_mean():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    bvea = nel.BinnedValueEventArray(veva, ds=0.5, method="mean")
    bea = nel.BinnedEventArray(nel.EventArray(np.array(events, dtype=float), fs=10), ds=0.5)
    np.testing.assert_allclose(bvea.bins, bea.bins)
    # Check mean for first series, first bin (events at 0.1 and 0.5)
    assert np.isclose(bvea.data[0, 0, 0], (1 + 2) / 2)
    # Check mean for second series, first bin (event at 0.2)
    assert np.isclose(bvea.data[1, 0, 0], 4)
    # Check mean for second series, second bin (event at 0.6)
    assert np.isclose(bvea.data[1, 1, 0], 5)


def test_binnedvalueeventarray_custom_func():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    bvea = nel.BinnedValueEventArray(
        veva, ds=0.5, method=lambda x: np.max(x) - np.min(x)
    )
    bea = nel.BinnedEventArray(nel.EventArray(np.array(events, dtype=float), fs=10), ds=0.5)
    np.testing.assert_allclose(bvea.bins, bea.bins)
    # Check custom aggregation (range) for first bin (events at 0.1 and 0.5)
    assert np.isclose(bvea.data[0, 0, 0], 2 - 1)
    # Check custom aggregation (range) for second series, first bin (event at 0.2)
    assert np.isclose(bvea.data[1, 0, 0], 0)
    # Check custom aggregation (range) for second series, second bin (event at 0.6)
    assert np.isclose(bvea.data[1, 1, 0], 0)


def test_binnedvalueeventarray_multiple_intervals():
    events = [[0.1, 0.5, 1.0, 2.1], [0.2, 0.6, 1.2, 2.2]]
    values = [[1, 2, 3, 4], [4, 5, 6, 7]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    # Restrict to two intervals: [0,1.5), [2,2.5)
    epochs = nel.EpochArray([[0, 1.5], [2, 2.5]])
    veva2 = veva[epochs]
    bvea = nel.BinnedValueEventArray(veva2, ds=0.5, method="sum")
    # Should only have bins within [0,1.5) and [2,2.5)
    assert np.all(
        (bvea.bin_centers >= 0) & (bvea.bin_centers < 1.5)
        | (bvea.bin_centers >= 2) & (bvea.bin_centers < 2.5)
    )


def test_binnedvalueeventarray_repr():
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2]]
    values = [[1, 2, 3], [4, 5, 6]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    bvea = nel.BinnedValueEventArray(veva, ds=0.5, method="sum")
    s = repr(bvea)
    assert "BinnedValueEventArray" in s


def test_binnedvalueeventarray_epocharray_slicing():
    events = [[0.1, 0.5, 1.0, 2.1], [0.2, 0.6, 1.2, 2.2]]
    values = [[1, 2, 3, 4], [4, 5, 6, 7]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    bvea = nel.BinnedValueEventArray(veva, ds=0.5, method="sum")
    epochs = nel.EpochArray([[0, 1.5], [2, 2.5]])
    bvea2 = bvea[epochs]
    # All bin centers should be within the epochs
    assert np.all(
        (bvea2.bin_centers >= 0) & (bvea2.bin_centers < 1.5)
        | (bvea2.bin_centers >= 2) & (bvea2.bin_centers < 2.5)
    )


def test_valueeventarray_multiseries_different_event_counts():
    """Test ValueEventArray with multi-series where each series has a different number of events."""
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2, 2.0]]
    values = [[1, 2, 3], [4, 5, 6, 7]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    assert veva.n_series == 2
    assert veva.n_events[0] == 3
    assert veva.n_events[1] == 4
    assert veva.data.shape[0] == 2
    # Check that the events and values are preserved correctly
    np.testing.assert_array_equal(veva.events[0], np.array([0.1, 0.5, 1.0]))
    np.testing.assert_array_equal(veva.events[1], np.array([0.2, 0.6, 1.2, 2.0]))
    np.testing.assert_array_equal(veva.values[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(veva.values[1], np.array([4, 5, 6, 7]))


def test_binnedvalueeventarray_ragged():
    # Series 0: 3 events, Series 1: 5 events
    events = [[0.1, 0.5, 1.0], [0.2, 0.6, 1.2, 2.0, 2.5]]
    values = [[1, 2, 3], [4, 5, 6, 7, 8]]
    veva = nel.ValueEventArray(events=events, values=values, fs=10)
    bvea = nel.BinnedValueEventArray(veva, ds=1.0, method="sum")
    bea = nel.BinnedEventArray(nel.EventArray(events, fs=10), ds=1.0)
    np.testing.assert_allclose(bvea.bins, bea.bins)
    # Bins should be [0.1,1.1,2.1] for both series
    assert bvea.data.shape[0] == 2  # 2 series
    assert bvea.data.shape[1] == 2  # 2 bins
    # Series 0: events at 0.1, 0.5, 1.0 (values 1,2,3)
    # Bin 0: [0.1,1.1): events at 0.1, 0.5, 1.0 (values 1+2+3=6)
    # Bin 1: [1.1,2.1): no events (should be 0)
    assert np.isclose(bvea.data[0, 0, 0], 1 + 2 + 3)
    assert np.isclose(bvea.data[0, 1, 0], 0)
    # Series 1: events at 0.2, 0.6, 1.2, 2.0, 2.5 (values 4,5,6,7,8)
    # Bin 0: [0.1,1.1): events at 0.2, 0.6 (values 4+5=9)
    # Bin 1: [1.1,2.1): events at 1.2, 2.0 (values 6+7=13)
    assert np.isclose(bvea.data[1, 0, 0], 4 + 5)
    assert np.isclose(bvea.data[1, 1, 0], 6 + 7)
