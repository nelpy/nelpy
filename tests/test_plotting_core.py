import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from nelpy import core
from nelpy.core import EpochArray, SpikeTrainArray
from nelpy.plotting import core as plotting_core

matplotlib.use("Agg")


@pytest.fixture
def epocharray_and_data():
    # Create a simple EpochArray and matching data
    epochs = core.EpochArray([[0, 1], [2, 3], [4, 5]])
    data = np.array([10, 20, 30])
    return epochs, data


def test_plot_epocharray_calls_plot_old(epocharray_and_data):
    epochs, data = epocharray_and_data
    fig, ax = plt.subplots()
    # Should not raise and should plot lines for each epoch
    plotting_core.plot(epochs, data, ax=ax)


def test_plot_old_epocharray(epocharray_and_data):
    epochs, data = epocharray_and_data
    fig, ax = plt.subplots()
    plotting_core.plot_old(epochs, data, ax=ax)


def test_plot_old_intervalarray():
    intervals = core.IntervalArray([[0, 1], [2, 3], [4, 5]])
    data = np.array([1, 2, 3])
    fig, ax = plt.subplots()
    plotting_core.plot_old(intervals, data, ax=ax)


def test_plot_regularlysampledanalogsignalarray():
    # Create a simple RegularlySampledAnalogSignalArray
    times = np.linspace(0, 1, 100)
    data = np.sin(2 * np.pi * times)
    rsa = core.RegularlySampledAnalogSignalArray(data=data, fs=100)
    fig, ax = plt.subplots()
    plotting_core.plot(rsa, ax=ax)


# Additional tests for other plotting functions


def test_plot2d_numpy():
    arr = np.array([[0, 0], [1, 1], [2, 0]])
    fig, ax = plt.subplots()
    plotting_core.plot2d(arr, ax=ax)


def test_colorline():
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    plotting_core.colorline(x, y, ax=ax)


def test_plot_tuning_curves1D():
    from nelpy.auxiliary import TuningCurve1D

    ratemap = np.array([[1, 2, 3], [2, 1, 0]])
    tc = TuningCurve1D(ratemap=ratemap)
    fig, ax = plt.subplots()
    plotting_core.plot_tuning_curves1D(tc, ax=ax)


def test_psdplot():
    times = np.linspace(0, 1, 1000)
    data = np.sin(2 * np.pi * 10 * times)
    rsa = core.RegularlySampledAnalogSignalArray(data=data, fs=1000)
    fig, ax = plt.subplots()
    plotting_core.psdplot(rsa, ax=ax)


def test_imagesc():
    data = np.random.rand(10, 10)
    fig, ax = plt.subplots()
    _, img = plotting_core.imagesc(data, ax=ax)


def test_matshow():
    data = np.sort(np.random.rand(5))
    epochs = EpochArray([[0, 5]])
    st = SpikeTrainArray(data, support=epochs)
    binned = st.bin()
    fig, ax = plt.subplots()
    plotting_core.matshow(binned, ax=ax)


def test_epochplot():
    epochs = core.EpochArray([[0, 1], [2, 3], [4, 5]])
    fig, ax = plt.subplots()
    plotting_core.epochplot(epochs, ax=ax)


def test_overviewstrip():
    epochs = core.EpochArray([[0, 1], [2, 3], [4, 5]])
    fig, ax = plt.subplots()
    plotting_core.overviewstrip(epochs, ax=ax)
