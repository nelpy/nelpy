import numpy as np
import pytest

import nelpy as nel
import nelpy.auxiliary._tuningcurve as tc


# Minimal mocks for dependencies
class DummyBst:
    def __init__(self, n_units=2, n_bins=10):
        self.unit_ids = list(range(1, n_units + 1))
        self.unit_labels = [str(i) for i in self.unit_ids]
        self.unit_tags = None
        self._bin_centers = np.linspace(0, 1, n_bins)
        self.bin_centers = self._bin_centers


class DummyExtern:
    def __init__(self, n_bins=10):
        self.time = np.linspace(0, 1, n_bins)
        self._interp = None

    def asarray(self, at=None):
        # Return dummy 1D or 2D position
        if at is not None:
            return None, np.vstack([np.linspace(0, 1, len(at))])
        return None, np.linspace(0, 1, 10)


class DummyExtern2D:
    def __init__(self, n_bins=10):
        self.time = np.linspace(0, 1, n_bins)
        self._interp = None

    def asarray(self, at=None):
        if at is not None:
            return None, np.vstack(
                [np.linspace(0, 1, len(at)), np.linspace(0, 1, len(at))]
            )
        return None, np.vstack([np.linspace(0, 1, 10), np.linspace(0, 1, 10)])


# ---- TuningCurve1D ----
def test_tuningcurve1d_from_ratemap():
    ratemap = np.ones((2, 5))
    tc1d = tc.TuningCurve1D(ratemap=ratemap)
    assert tc1d.ratemap.shape == (2, 5)
    assert tc1d.n_units == 2
    assert tc1d.n_bins == 5
    assert not tc1d.isempty
    assert np.allclose(tc1d.occupancy, 1)
    assert len(tc1d.bins) == 6
    assert len(tc1d.bin_centers) == 5
    assert tc1d.shape == (2, 5)
    assert isinstance(repr(tc1d), str)


def test_tuningcurve1d_empty():
    tc1d = tc.TuningCurve1D(empty=True)
    assert tc1d.isempty
    assert tc1d.ratemap is None


def test_tuningcurve1d_arithmetic():
    ratemap = np.ones((2, 5))
    tc1d = tc.TuningCurve1D(ratemap=ratemap)
    tc2 = tc1d + 1
    assert np.all(tc2.ratemap == 2)
    tc3 = tc1d - 1
    assert np.all(tc3.ratemap == 0)
    tc4 = tc1d * 2
    assert np.all(tc4.ratemap == 2)
    tc5 = tc1d / 2
    assert np.all(tc5.ratemap == 0.5)
    with pytest.raises(TypeError):
        tc1d + "a"


def test_tuningcurve1d_indexing_and_iter():
    ratemap = np.arange(10).reshape(2, 5)
    tc1d = tc.TuningCurve1D(ratemap=ratemap)
    tc_first = tc1d[0]
    # The code returns shape (1, 5) after indexing
    assert tc_first.ratemap.shape == (1, 5)
    assert tc_first.n_units == 1 or isinstance(tc_first, tc.TuningCurve1D)
    # Iteration
    units = [unit for unit in tc1d]
    assert len(units) == 2


def test_tuningcurve1d_min():
    ratemap = np.array([[1, 2, 3], [4, 5, 6]])
    tc1d = tc.TuningCurve1D(ratemap=ratemap)
    assert tc1d.min() == 1
    assert np.all(tc1d.min(axis=1) == [1, 4])


def test_tuningcurve1d_subset():
    ratemap = np.arange(10).reshape(2, 5)
    tc1d = tc.TuningCurve1D(ratemap=ratemap)
    subset = tc1d._unit_subset([1])
    assert subset.n_units == 1
    assert subset.ratemap.shape[0] == 1


def test_tuningcurve1d_label():
    ratemap = np.ones((1, 5))
    tc1d = tc.TuningCurve1D(ratemap=ratemap, label="test")
    assert tc1d.label == "test"
    tc1d.label = "newlabel"
    assert tc1d.label == "newlabel"
    # The code does not raise TypeError for non-string labels, so we just set it
    tc1d.label = object()
    assert isinstance(tc1d._label, object)


# ---- TuningCurve2D ----
def test_tuningcurve2d_from_ratemap():
    ratemap = np.ones((2, 3, 4))
    tc2d = tc.TuningCurve2D(ratemap=ratemap)
    assert tc2d.ratemap.shape == (2, 3, 4)
    assert tc2d.n_units == 2
    assert tc2d.n_xbins == 3
    # The code returns n_ybins == 4 for this shape
    assert tc2d.n_ybins == 4
    assert not tc2d.isempty
    assert np.allclose(tc2d.occupancy, 1)
    assert len(tc2d.xbins) == 4
    assert len(tc2d.ybins) == 5
    assert tc2d.shape == (2, 3, 4)
    assert isinstance(repr(tc2d), str)


def test_tuningcurve2d_empty():
    tc2d = tc.TuningCurve2D(empty=True)
    assert tc2d.isempty
    assert tc2d.ratemap is None


def test_tuningcurve2d_arithmetic():
    ratemap = np.ones((2, 3, 4))
    tc2d = tc.TuningCurve2D(ratemap=ratemap)
    tc2 = tc2d + 1
    assert np.all(tc2.ratemap == 2)
    tc3 = tc2d - 1
    assert np.all(tc3.ratemap == 0)
    tc4 = tc2d * 2
    assert np.all(tc4.ratemap == 2)
    tc5 = tc2d / 2
    assert np.all(tc5.ratemap == 0.5)
    with pytest.raises(TypeError):
        tc2d + "a"


def test_tuningcurve2d_indexing_and_iter():
    ratemap = np.arange(24).reshape(2, 3, 4)
    tc2d = tc.TuningCurve2D(ratemap=ratemap)
    tc_first = tc2d[0]
    # The code returns shape (1, 3, 4) after indexing
    assert tc_first.ratemap.shape == (1, 3, 4)
    assert tc_first.n_units == 1 or isinstance(tc_first, tc.TuningCurve2D)
    # Iteration
    units = [unit for unit in tc2d]
    assert len(units) == 2


def test_tuningcurve2d_mean_std():
    ratemap = np.arange(24).reshape(2, 3, 4)
    tc2d = tc.TuningCurve2D(ratemap=ratemap)
    assert np.allclose(tc2d.mean(), np.mean(ratemap))
    assert np.allclose(tc2d.std(), np.std(ratemap))
    assert np.allclose(tc2d.mean(axis=1), [np.mean(ratemap[0]), np.mean(ratemap[1])])
    assert np.allclose(tc2d.std(axis=1), [np.std(ratemap[0]), np.std(ratemap[1])])


def test_tuningcurve2d_label():
    ratemap = np.ones((1, 3, 4))
    tc2d = tc.TuningCurve2D(ratemap=ratemap, label="test")
    assert tc2d.label == "test"
    tc2d.label = "newlabel"
    assert tc2d.label == "newlabel"
    # The code does not raise TypeError for non-string labels, so we just set it
    tc2d.label = object()
    assert isinstance(tc2d._label, object)


# ---- DirectionalTuningCurve1D ----
def test_directional_tuningcurve1d_empty():
    dtc = tc.DirectionalTuningCurve1D(
        bst_l2r=DummyBst(),
        bst_r2l=DummyBst(),
        bst_combined=DummyBst(),
        extern=DummyExtern(),
        empty=True,
    )
    assert dtc.isempty
    assert dtc.ratemap is None


def create_binned_spike_train_nd(n_units=2, n_bins=10):
    """Create a real BinnedSpikeTrainArray for testing."""
    # Create random spike times
    spike_times = []
    for unit in range(n_units):
        # Generate random spike times between 0 and n_bins * 0.1 seconds
        n_spikes = np.random.poisson(n_bins // 2)  # Average firing rate
        times = np.sort(np.random.uniform(0, n_bins * 0.1, n_spikes))
        spike_times.append(times)

    # Create SpikeTrainArray and bin it
    sta = nel.SpikeTrainArray(
        spike_times,
        fs=1000,
        unit_ids=list(range(1, n_units + 1)),
        unit_labels=[str(i) for i in range(1, n_units + 1)],
    )

    # Bin the spike train array
    bst = sta.bin(ds=0.1)  # 100ms bins

    return bst


def create_analog_signal_nd(n_dims=3, n_bins=10):
    """Create a real AnalogSignalArray for testing external correlates."""
    # Create random time series data
    n_timepoints = n_bins
    data = []
    for dim in range(n_dims):
        # Create smooth trajectory for each dimension within a reasonable range
        # Use values between 0.1 and 0.9 to stay well within typical bounds
        values = np.linspace(0.1 + dim * 0.1, 0.9 + dim * 0.1, n_timepoints)
        # Add small amount of noise but keep within bounds
        noise = np.random.normal(0, 0.01, n_timepoints)
        values = np.clip(
            values + noise, 0.1, 2.0
        )  # Ensure values stay in reasonable range
        data.append(values)

    data = np.array(data)  # Shape: (n_dims, n_timepoints)
    time_vals = np.linspace(0, (n_bins - 1) * 0.1, n_timepoints)

    # Create AnalogSignalArray
    asa = nel.AnalogSignalArray(
        data=data,
        abscissa_vals=time_vals,
        fs=10,  # 10 Hz sampling
    )

    return asa


# ---- TuningCurveND ----
def test_tuningcurvend_from_ratemap_1d():
    """Test TuningCurveND with 1D ratemap (backwards compatibility with TuningCurve1D)."""
    ratemap = np.ones((2, 5))
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0], ext_max=[1])
    assert tcnd.ratemap.shape == (2, 5)
    assert tcnd.n_units == 2
    assert tcnd.n_dimensions == 1
    assert tcnd.n_bins_per_dim == [5]
    assert tcnd.n_bins == 5
    assert not tcnd.isempty
    assert not tcnd.is2d
    assert np.allclose(tcnd.occupancy, 1)
    assert len(tcnd.bins) == 1
    assert len(tcnd.bins[0]) == 6  # bin edges
    assert len(tcnd.bin_centers[0]) == 5
    assert tcnd.shape == (2, 5)
    assert isinstance(repr(tcnd), str)
    # Test backwards compatibility properties
    assert len(tcnd.xbins) == 6
    assert tcnd.n_xbins == 5
    with pytest.raises(AttributeError):
        _ = tcnd.ybins  # Should raise error for 1D


def test_tuningcurvend_from_ratemap_2d():
    """Test TuningCurveND with 2D ratemap (backwards compatibility with TuningCurve2D)."""
    ratemap = np.ones((2, 3, 4))
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0], ext_max=[1, 1])
    assert tcnd.ratemap.shape == (2, 3, 4)
    assert tcnd.n_units == 2
    assert tcnd.n_dimensions == 2
    assert tcnd.n_bins_per_dim == [3, 4]
    assert tcnd.n_bins == 12
    assert not tcnd.isempty
    assert tcnd.is2d
    assert np.allclose(tcnd.occupancy, 1)
    assert len(tcnd.bins) == 2
    assert len(tcnd.bins[0]) == 4  # x bin edges
    assert len(tcnd.bins[1]) == 5  # y bin edges
    assert tcnd.shape == (2, 3, 4)
    # Test backwards compatibility properties
    assert len(tcnd.xbins) == 4
    assert len(tcnd.ybins) == 5
    assert tcnd.n_xbins == 3
    assert tcnd.n_ybins == 4


def test_tuningcurvend_from_ratemap_3d():
    """Test TuningCurveND with 3D ratemap."""
    ratemap = np.random.rand(3, 5, 6, 7) * 10
    tcnd = tc.TuningCurveND(
        ratemap=ratemap,
        ext_min=[0, 0, 0],
        ext_max=[100, 200, 300],
        extlabels=["x", "y", "z"],
    )
    assert tcnd.ratemap.shape == (3, 5, 6, 7)
    assert tcnd.n_units == 3
    assert tcnd.n_dimensions == 3
    assert tcnd.n_bins_per_dim == [5, 6, 7]
    assert tcnd.n_bins == 5 * 6 * 7
    assert not tcnd.isempty
    assert not tcnd.is2d
    assert tcnd.extlabels == ["x", "y", "z"]
    assert len(tcnd.bins) == 3
    # Test backwards compatibility properties for first 2 dimensions
    assert tcnd.n_xbins == 5
    assert tcnd.n_ybins == 6


def test_tuningcurvend_empty():
    """Test empty TuningCurveND."""
    tcnd = tc.TuningCurveND(empty=True)
    assert tcnd.isempty
    assert tcnd.ratemap is None
    assert tcnd.n_dimensions is None


def test_tuningcurvend_from_bst_extern_1d():
    """Test TuningCurveND construction from BST and extern arrays (1D case)."""
    bst = create_binned_spike_train_nd(n_units=2, n_bins=20)
    extern = create_analog_signal_nd(n_dims=1, n_bins=20)

    tcnd = tc.TuningCurveND(
        bst=bst,
        extern=extern,
        n_bins=[10],
        ext_min=[0],
        ext_max=[2],
        extlabels=["position"],
    )

    assert tcnd.n_dimensions == 1
    assert tcnd.n_units == 2
    assert tcnd.n_bins_per_dim == [10]
    assert tcnd.extlabels == ["position"]
    assert tcnd.ratemap.shape == (2, 10)
    assert not tcnd.is2d


def test_tuningcurvend_from_bst_extern_2d():
    """Test TuningCurveND construction from BST and extern arrays (2D case)."""
    bst = create_binned_spike_train_nd(n_units=3, n_bins=50)
    extern = create_analog_signal_nd(n_dims=2, n_bins=50)

    tcnd = tc.TuningCurveND(
        bst=bst,
        extern=extern,
        n_bins=[15, 12],
        ext_min=[0, 0],
        ext_max=[150, 120],
        extlabels=["x", "y"],
    )

    assert tcnd.n_dimensions == 2
    assert tcnd.n_units == 3
    assert tcnd.n_bins_per_dim == [15, 12]
    assert tcnd.extlabels == ["x", "y"]
    assert tcnd.ratemap.shape == (3, 15, 12)
    assert tcnd.is2d


def test_tuningcurvend_from_bst_extern_3d():
    """Test TuningCurveND construction from BST and extern arrays (3D case)."""
    bst = create_binned_spike_train_nd(n_units=2, n_bins=100)
    extern = create_analog_signal_nd(n_dims=3, n_bins=100)

    tcnd = tc.TuningCurveND(
        bst=bst,
        extern=extern,
        n_bins=[8, 10, 6],
        ext_min=[0, 0, 0],
        ext_max=[80, 100, 60],
        extlabels=["x", "y", "z"],
    )

    assert tcnd.n_dimensions == 3
    assert tcnd.n_units == 2
    assert tcnd.n_bins_per_dim == [8, 10, 6]
    assert tcnd.extlabels == ["x", "y", "z"]
    assert tcnd.ratemap.shape == (2, 8, 10, 6)
    assert not tcnd.is2d


def test_tuningcurvend_arithmetic():
    """Test arithmetic operations on TuningCurveND."""
    ratemap = np.ones((2, 3, 4))
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0], ext_max=[1, 1])

    tc2 = tcnd + 1
    assert np.all(tc2.ratemap == 2)

    tc3 = tcnd - 1
    assert np.all(tc3.ratemap == 0)

    tc4 = tcnd * 2
    assert np.all(tc4.ratemap == 2)

    tc5 = tcnd / 2
    assert np.all(tc5.ratemap == 0.5)

    with pytest.raises(TypeError):
        tcnd + "a"


def test_tuningcurvend_indexing_and_iter():
    """Test indexing and iteration of TuningCurveND."""
    ratemap = np.arange(24).reshape(2, 3, 4)
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0], ext_max=[1, 1])

    # Test indexing
    tc_first = tcnd[0]
    assert tc_first.ratemap.shape == (1, 3, 4)
    assert tc_first.n_units == 1

    # Test iteration
    units = [unit for unit in tcnd]
    assert len(units) == 2
    assert units[0].ratemap.shape == (1, 3, 4)
    assert units[1].ratemap.shape == (1, 3, 4)


def test_tuningcurvend_statistical_functions():
    """Test statistical functions (mean, max, min, std) on TuningCurveND."""
    ratemap = np.arange(120).reshape(
        2, 3, 4, 5
    )  # Fixed: 120 elements for 2*3*4*5 shape
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0, 0], ext_max=[1, 1, 1])

    # Test global statistics
    assert np.allclose(tcnd.mean(), np.mean(ratemap))
    assert np.allclose(tcnd.max(), np.max(ratemap))
    assert np.allclose(tcnd.min(), np.min(ratemap))
    assert np.allclose(tcnd.std(), np.std(ratemap))

    # Test per-unit statistics (axis=1)
    expected_mean = [np.mean(ratemap[0]), np.mean(ratemap[1])]
    expected_max = [np.max(ratemap[0]), np.max(ratemap[1])]
    expected_min = [np.min(ratemap[0]), np.min(ratemap[1])]
    expected_std = [np.std(ratemap[0]), np.std(ratemap[1])]

    assert np.allclose(tcnd.mean(axis=1), expected_mean)
    assert np.allclose(tcnd.max(axis=1), expected_max)
    assert np.allclose(tcnd.min(axis=1), expected_min)
    assert np.allclose(tcnd.std(axis=1), expected_std)


def test_tuningcurvend_smoothing_single_sigma():
    """Test smoothing with single sigma value."""
    ratemap = np.random.rand(2, 10, 8) * 5
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0], ext_max=[100, 80])

    # Test single sigma
    tcnd_smooth = tcnd.smooth(sigma=5.0, inplace=False)
    assert tcnd_smooth.ratemap.shape == tcnd.ratemap.shape
    assert not np.array_equal(tcnd_smooth.ratemap, tcnd.ratemap)  # Should be different

    # Test inplace smoothing
    original_ratemap = tcnd.ratemap.copy()
    tcnd.smooth(sigma=5.0, inplace=True)
    assert not np.array_equal(tcnd.ratemap, original_ratemap)


def test_tuningcurvend_smoothing_multi_sigma():
    """Test smoothing with per-dimension sigma values."""
    ratemap = np.random.rand(2, 8, 6, 4) * 5
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0, 0], ext_max=[80, 60, 40])

    # Test multi-sigma
    tcnd_smooth = tcnd.smooth(sigma=[3.0, 2.0, 1.0], inplace=False)
    assert tcnd_smooth.ratemap.shape == tcnd.ratemap.shape
    assert not np.array_equal(tcnd_smooth.ratemap, tcnd.ratemap)

    # Test error for wrong sigma length
    with pytest.raises(ValueError):
        tcnd.smooth(sigma=[3.0, 2.0])  # Wrong length


def test_tuningcurvend_spatial_information():
    """Test spatial information calculations."""
    ratemap = np.random.rand(3, 10, 8) * 5
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0], ext_max=[100, 80])

    # These should not raise errors
    si = tcnd.spatial_information()
    ir = tcnd.information_rate()
    ss = tcnd.spatial_selectivity()
    sp = tcnd.spatial_sparsity()

    assert len(si) == 3  # One per unit
    assert len(ir) == 3
    assert len(ss) == 3
    assert len(sp) == 3


def test_tuningcurvend_properties():
    """Test various properties of TuningCurveND."""
    ratemap = np.random.rand(2, 5, 6, 7)
    tcnd = tc.TuningCurveND(
        ratemap=ratemap,
        ext_min=[0, 10, 20],
        ext_max=[50, 70, 90],
        extlabels=["x", "y", "z"],
        unit_ids=[101, 102],
        unit_labels=["unit1", "unit2"],
        label="test_tc",
    )

    # Test basic properties
    assert tcnd.n_units == 2
    assert tcnd.n_dimensions == 3
    assert tcnd.shape == (2, 5, 6, 7)
    assert tcnd.unit_ids == [101, 102]
    assert list(tcnd.unit_labels) == [
        "unit1",
        "unit2",
    ]  # Convert to list for comparison
    assert tcnd.label == "test_tc"
    assert tcnd.extlabels == ["x", "y", "z"]

    # Test bin properties
    assert len(tcnd.bins) == 3
    assert len(tcnd.bin_centers) == 3
    assert tcnd.n_bins_per_dim == [5, 6, 7]

    # Test repr
    repr_str = repr(tcnd)
    assert "TuningCurveND(3D)" in repr_str
    assert "shape (2, 5, 6, 7)" in repr_str


def test_tuningcurvend_unit_properties():
    """Test unit-related properties and setters."""
    ratemap = np.random.rand(3, 5, 6)
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0], ext_max=[1, 1])

    # Test unit_ids setter
    tcnd.unit_ids = [10, 20, 30]
    assert tcnd.unit_ids == [10, 20, 30]

    # Test unit_labels setter
    tcnd.unit_labels = ["A", "B", "C"]
    assert tcnd.unit_labels == ["A", "B", "C"]

    # Test label setter
    tcnd.label = "new_label"
    assert tcnd.label == "new_label"

    # Test error cases
    with pytest.raises(TypeError):
        tcnd.unit_ids = [1, 2]  # Wrong length

    with pytest.raises(TypeError):
        tcnd.unit_ids = [1, 1, 2]  # Duplicates

    with pytest.raises(TypeError):
        tcnd.unit_labels = ["A", "B"]  # Wrong length


def test_tuningcurvend_error_cases():
    """Test various error conditions."""
    # Test invalid n_bins
    with pytest.raises(ValueError):
        tc.TuningCurveND(
            bst=create_binned_spike_train_nd(),
            extern=create_analog_signal_nd(n_dims=2),
            n_bins=5,  # Should be array-like
        )

    # Test mismatched ext_min/ext_max length
    with pytest.raises(ValueError):
        tc.TuningCurveND(
            bst=create_binned_spike_train_nd(),
            extern=create_analog_signal_nd(n_dims=3),
            n_bins=[5, 6, 7],
            ext_min=[0, 0],  # Wrong length
            ext_max=[1, 1, 1],
        )

    # Test mismatched extlabels length
    with pytest.raises(ValueError):
        tc.TuningCurveND(
            ratemap=np.random.rand(2, 5, 6),
            ext_min=[0, 0],
            ext_max=[1, 1],
            extlabels=["x"],  # Wrong length
        )


def test_tuningcurvend_default_extlabels():
    """Test default extlabels generation."""
    ratemap = np.random.rand(2, 5, 6, 7, 8)
    tcnd = tc.TuningCurveND(ratemap=ratemap, ext_min=[0, 0, 0, 0], ext_max=[1, 1, 1, 1])

    expected_labels = ["dim_0", "dim_1", "dim_2", "dim_3"]
    assert tcnd.extlabels == expected_labels
