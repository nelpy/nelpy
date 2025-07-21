import numpy as np
import pytest

import nelpy as nel


def test_sosfiltfilt_numpy_lowpass():
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(t.size)
    y = nel.filtering.sosfiltfilt(x, fs=1000, fl=None, fh=10)
    assert y.shape == x.shape
    assert isinstance(y, np.ndarray)
    # Should reduce high-frequency noise
    assert np.std(y) < np.std(x)

def test_sosfiltfilt_asa_lowpass():
    t = np.linspace(0, 1, 1000, endpoint=False)
    data = np.vstack([np.sin(2 * np.pi * 5 * t), np.cos(2 * np.pi * 5 * t)])
    asa = nel.AnalogSignalArray(data=data, abscissa_vals=t, fs=1000)
    filtered = nel.filtering.sosfiltfilt(asa, fl=None, fh=10)
    assert filtered.data.shape == asa.data.shape
    # Should reduce high-frequency noise
    assert np.std(filtered.data) < np.std(asa.data)

def test_sosfiltfilt_invalid_input():
    with pytest.raises(TypeError):
        nel.filtering.sosfiltfilt("not an array", fs=1000, fl=None, fh=10)

def test_sosfiltfilt_missing_fs():
    arr = np.arange(10)
    with pytest.raises(ValueError):
        nel.filtering.sosfiltfilt(arr, fl=None, fh=10)

def test_sosfiltfilt_bandpass():
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 5 * t)
    y = nel.filtering.sosfiltfilt(x, fs=1000, fl=40, fh=60)
    # Should suppress the 5 Hz component
    assert np.abs(np.mean(y)) < 1 