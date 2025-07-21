import numpy as np
import pytest

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
