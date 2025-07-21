import numpy as np

import nelpy as nel


def test_standardize_asa_numpy():
    @nel.preprocessing.standardize_asa(asa="X")
    def dummy(X=None, **kwargs):
        return X

    arr = np.arange(10)
    result = dummy(X=arr)
    assert isinstance(result, np.ndarray)
    assert np.all(result == arr)


def test_standardize_asa_asa():
    @nel.preprocessing.standardize_asa(
        asa="X", lengths="lengths", timestamps="timestamps", fs="fs"
    )
    def dummy(X=None, lengths=None, timestamps=None, fs=None, **kwargs):
        return X, lengths, timestamps, fs

    t = np.linspace(0, 1, 10)
    data = np.sin(2 * np.pi * t)
    asa = nel.AnalogSignalArray(data=data, abscissa_vals=t, fs=10)
    X, lengths, timestamps, fs = dummy(X=asa)
    assert isinstance(X, np.ndarray)
    assert X.shape == (10,)
    assert np.allclose(timestamps, t)
    assert fs == 10
    assert lengths is not None


def test_datawindow_transform():
    X = np.arange(20).reshape(-1, 2)  # 2D array: 10 samples, 2 features
    w = nel.preprocessing.DataWindow(1, 1, 1, 1)
    win = w.fit(X)
    out, _ = win.transform(X)
    assert out.shape[0] > 0
    # Check that the windowed output contains the original data in some form
    assert np.any(np.isin(X, out))


def test_standardscaler():
    X = np.array([[1, 2], [4, 5], [3, 6]])  # shape (3, 2)
    scaler = nel.preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Mean should be ~0, std ~1
    np.testing.assert_allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
    np.testing.assert_allclose(X_scaled.std(axis=0), 1, atol=1e-7)
