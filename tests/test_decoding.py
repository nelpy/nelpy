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


def _decode_window_posterior(bst, ratemap, ndim, w, **kwargs):
    if ndim == 1:
        posterior, lengths, _, _ = nel.decoding.decode1D(bst, ratemap, w=w, **kwargs)
    else:
        tuningcurve = nel.TuningCurve2D(
            ratemap=ratemap.reshape(2, 2, 2), unit_ids=bst.unit_ids
        )
        posterior, lengths, _, _ = nel.decoding.decode2D(
            bst, tuningcurve, w=w, **kwargs
        )
        posterior = posterior.transpose(1, 0, 2).reshape(4, -1)
    return posterior, lengths


def _poisson_posterior(ratemap, counts, duration):
    likelihood = np.prod(ratemap ** counts[:, np.newaxis], axis=0) * np.exp(
        -duration * ratemap.sum(axis=0)
    )
    return likelihood / likelihood.sum()


@pytest.mark.parametrize("ndim", [1, 2], ids=["1D", "2D"])
@pytest.mark.parametrize("n_bins", [1, 2, 3, 5])
def test_decode_window_uses_observed_duration(ndim, n_bins):
    ds = 0.25
    w = 3
    ratemap = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]])
    spikes = [
        (np.arange(n_bins) + 0.2) * ds,
        (np.arange(0, n_bins, 2) + 0.4) * ds,
    ]
    bst = nel.SpikeTrainArray(spikes, support=nel.EpochArray([0, n_bins * ds])).bin(
        ds=ds
    )

    posterior, lengths = _decode_window_posterior(bst, ratemap, ndim, w)

    bin_counts = np.vstack((np.ones(n_bins), np.arange(n_bins) % 2 == 0))
    expected = []
    for start in range(max(1, n_bins - w + 1)):
        counts = bin_counts[:, start : start + w].sum(axis=1)
        duration = min(n_bins, w) * ds
        expected.append(_poisson_posterior(ratemap, counts, duration))
    np.testing.assert_allclose(posterior, np.column_stack(expected))
    np.testing.assert_array_equal(lengths, [0, len(expected)])


@pytest.mark.parametrize("ndim", [1, 2], ids=["1D", "2D"])
def test_decode_window_duration_is_independent_for_each_epoch(ndim):
    ds = 0.25
    w = 3
    ratemap = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]])
    n_bins = np.array([1, 3, 5])
    starts = np.array([0.0, 2.0, 4.0])
    support = nel.EpochArray(np.column_stack((starts, starts + n_bins * ds)))
    spikes = np.concatenate(
        [
            start + (np.arange(length) + 0.2) * ds
            for start, length in zip(starts, n_bins)
        ]
    )
    bst = nel.SpikeTrainArray([spikes, spikes], support=support).bin(ds=ds)

    posterior, lengths = _decode_window_posterior(bst, ratemap, ndim, w)

    short = _poisson_posterior(ratemap, np.array([1, 1]), ds)
    full = _poisson_posterior(ratemap, np.array([w, w]), ds * w)
    np.testing.assert_allclose(
        posterior, np.column_stack([short, full, full, full, full])
    )
    np.testing.assert_array_equal(lengths, [0, 1, 2, 5])


@pytest.mark.parametrize("ndim", [1, 2], ids=["1D", "2D"])
@pytest.mark.parametrize("use_prior", [False, True])
def test_decode_short_window_without_spikes_preserves_prior_policy(ndim, use_prior):
    ds = 0.25
    ratemap = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 1.0, 2.0, 3.0]])
    support = nel.EpochArray([[0, ds], [2, 2 + 3 * ds]])
    bst = nel.SpikeTrainArray([[2.1], [2.15]], support=support).bin(ds=ds)

    # Preserve the existing, differing empty-bin flags of the two decoders.
    skip_empty_bins = use_prior if ndim == 1 else not use_prior
    posterior, lengths = _decode_window_posterior(
        bst, ratemap, ndim, w=3, nospk_prior=1.0, _skip_empty_bins=skip_empty_bins
    )

    if use_prior:
        expected = np.full(4, 0.25)
    else:
        expected = _poisson_posterior(ratemap, np.zeros(2), ds)
    np.testing.assert_allclose(posterior[:, 0], expected)
    np.testing.assert_array_equal(lengths, [0, 1, 2])


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
