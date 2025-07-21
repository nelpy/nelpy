import numpy as np
import pytest

import nelpy as nel


def test_decode1D_simple():
    ratemap = np.array([[0.1, 1.0, 0.1]])
    # Events at times that will bin into 3 bins
    ea = nel.EventArray([[0.2, 1.2, 2.2]], fs=1)
    bst = nel.BinnedSpikeTrainArray(ea, ds=1)
    posterior, cum_posterior_lengths, mode_pth, mean_pth = nel.decoding.decode1D(
        bst, ratemap
    )
    # Should decode to the bin with the highest rate (bin 1)
    assert np.all(mode_pth == 1)


def test_get_mode_pth_from_array():
    posterior = np.array([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]])
    mode = nel.decoding.get_mode_pth_from_array(posterior)
    assert np.all(mode == np.argmax(posterior, axis=0))


def test_bayesian_decoder_fit_predict():
    ratemap = np.array([[0.1, 1.0, 0.1]])
    ea = nel.EventArray([[0.2, 1.2, 2.2]], fs=1)
    bst = nel.BinnedSpikeTrainArray(ea, ds=1)
    decoder = nel.decoding.BayesianDecoder()
    decoder.fit(ratemap)
    # Test predict_proba
    posterior = decoder.predict_proba(bst)
    assert posterior.shape[1] == bst.data.shape[1]
    # Test predict (mode path)
    mode_pth = decoder.predict(bst)
    assert mode_pth.shape[0] == bst.data.shape[1]
    # Test predict_asa (mean path as AnalogSignalArray)
    asa = decoder.predict_asa(bst)
    from nelpy.core import AnalogSignalArray

    assert isinstance(asa, AnalogSignalArray)
    # abscissa_vals should match bin_centers
    assert np.allclose(asa.abscissa_vals, bst.bin_centers)


def test_decode1D_invalid_input():
    ratemap = np.array([[0.1, 1.0]])
    bst = np.zeros((2, 2))  # 2 units, but ratemap has 1
    try:
        nel.decoding.decode1D(bst, ratemap)
    except Exception:
        pass
    else:
        assert False, "Expected an exception for shape mismatch"


def test_decode2D_simple():
    from nelpy.auxiliary._tuningcurve import TuningCurve2D

    # Create a simple ratemap and bins
    n_units, ext_nx, ext_ny = 1, 2, 2
    ratemap = np.ones((n_units, ext_ny, ext_nx))
    tc2d = TuningCurve2D(
        ratemap=ratemap,
        ext_xmin=0,
        ext_xmax=2,
        ext_ymin=0,
        ext_ymax=2,
        ext_nx=ext_nx,
        ext_ny=ext_ny,
    )
    ea = nel.EventArray([[0.2, 0.8, 1.2, 1.8]], fs=1)
    bst = nel.BinnedSpikeTrainArray(ea, ds=1)
    posterior, cum_posterior_lengths, mode_pth, mean_pth = nel.decoding.decode2D(
        bst, tc2d
    )
    assert posterior.shape[0] == ext_nx and posterior.shape[1] == ext_ny
    assert mode_pth.shape[1] == posterior.shape[2]
    assert mean_pth.shape[1] == posterior.shape[2]


def test_get_mean_pth_from_array():
    posterior = np.array([[0.1, 0.8, 0.1], [0.7, 0.1, 0.2]])
    mean = nel.decoding.get_mean_pth_from_array(posterior, tuningcurve=None)
    assert mean.shape == (3,)
    assert np.all((mean >= 0) & (mean <= 1))


def test_decode_bayesian_memoryless_nd():
    pytest.skip("decode_bayesian_memoryless_nd is not public or not available")


def test_k_fold_cross_validation_runs():
    # Use a list of indices and k
    X = list(range(10))
    folds = list(nel.decoding.k_fold_cross_validation(X, k=2))
    assert len(folds) == 2
    for train, val in folds:
        assert set(train + val) == set(X)
        assert len(set(train).intersection(val)) == 0


def test_cumulative_dist_decoding_error():
    # Use a real TuningCurve1D and BinnedSpikeTrainArray
    from nelpy.auxiliary._tuningcurve import TuningCurve1D

    ea = nel.EventArray([[0.2, 0.8, 1.2, 1.8]], fs=1)
    bst = nel.BinnedSpikeTrainArray(ea, ds=1)
    extern = nel.AnalogSignalArray(
        data=np.linspace(0, 1, 4)[np.newaxis, :], abscissa_vals=np.arange(4)
    )
    tc = TuningCurve1D(ratemap=np.ones((1, 4)))
    cumhist, bincenters = nel.decoding.cumulative_dist_decoding_error(
        bst=bst, tuningcurve=tc, extern=extern
    )
    assert cumhist.shape == bincenters.shape
    assert np.all(np.diff(cumhist) >= 0)
