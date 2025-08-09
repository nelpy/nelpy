import matplotlib.pyplot as plt
import numpy as np
import pytest

from nelpy import core
from nelpy.plotting import core as plotting_core


@pytest.fixture
def epocharray_and_data():
    # Create a simple EpochArray and matching data
    epochs = core.EpochArray([[0, 1], [2, 3], [4, 5]])
    data = np.array([10, 20, 30])
    return epochs, data


@pytest.mark.mpl_image_compare(baseline_dir=None, tolerance=10)
def test_plot_epocharray_calls_plot_old(epocharray_and_data):
    epochs, data = epocharray_and_data
    fig, ax = plt.subplots()
    # Should not raise and should plot lines for each epoch
    plotting_core.plot(epochs, data, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=None, tolerance=10)
def test_plot_old_epocharray(epocharray_and_data):
    epochs, data = epocharray_and_data
    fig, ax = plt.subplots()
    plotting_core.plot_old(epochs, data, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=None, tolerance=10)
def test_plot_old_intervalarray():
    intervals = core.IntervalArray([[0, 1], [2, 3], [4, 5]])
    data = np.array([1, 2, 3])
    fig, ax = plt.subplots()
    plotting_core.plot_old(intervals, data, ax=ax)
    return fig


@pytest.mark.mpl_image_compare(baseline_dir=None, tolerance=10)
def test_plot_regularlysampledanalogsignalarray():
    # Create a simple RegularlySampledAnalogSignalArray
    times = np.linspace(0, 1, 100)
    data = np.sin(2 * np.pi * times)
    rsa = core.RegularlySampledAnalogSignalArray(data=data, fs=100)
    fig, ax = plt.subplots()
    plotting_core.plot(rsa, ax=ax)
    return fig
