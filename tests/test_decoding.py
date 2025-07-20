import numpy as np

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
