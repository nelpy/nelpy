import dill as pickle
import pytest

import nelpy as nel
from nelpy import formatters


def _roundtrip_without_attrs(obj, attrs):
    for attr in attrs:
        obj.__dict__.pop(attr, None)

    return pickle.loads(pickle.dumps(obj))


def test_stale_spiketrainarray_pickle_reattaches_indexers():
    sta = nel.SpikeTrainArray(
        [[1, 2, 3, 4], [2, 3, 5]],
        support=nel.EpochArray([[0, 4], [4, 6]]),
        fs=1,
        series_ids=[10, 20],
    )

    loaded = _roundtrip_without_attrs(sta, ["loc", "iloc"])

    assert loaded.loc.obj is loaded
    assert loaded.iloc.obj is loaded
    assert loaded.loc[:, 20].n_series == 1
    assert loaded.iloc[:, 0].n_series == 1


def test_stale_analogsignalarray_pickle_reattaches_slicers():
    asa = nel.AnalogSignalArray(
        [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        fs=1,
        support=nel.EpochArray([[0, 2], [3, 5]]),
    )

    loaded = _roundtrip_without_attrs(
        asa,
        ["_intervalsignalslicer", "_intervaldata", "_intervaltime"],
    )

    assert loaded._intervalsignalslicer.obj is loaded
    assert loaded._intervaldata._parent is loaded
    assert loaded._intervaltime._parent is loaded
    assert loaded[nel.EpochArray([0, 2])].n_signals == 2
    assert loaded[:, 1].n_signals == 1


def test_stale_valueeventarray_pickle_reattaches_indexers():
    veva = nel.ValueEventArray(
        events=[[0.1, 0.5], [0.2, 0.6]],
        values=[[1, 2], [3, 4]],
        fs=10,
        series_ids=[3, 4],
    )

    loaded = _roundtrip_without_attrs(veva, ["loc", "iloc"])

    assert loaded.loc.obj is loaded
    assert loaded.iloc.obj is loaded
    assert loaded.iloc[:, 1].n_series == 1


@pytest.mark.parametrize(
    "obj, formatter, base_unit, interval_label",
    [
        (
            nel.IntervalArray([[0, 1]]),
            formatters.ArbitraryFormatter,
            "a.u.",
            "interval",
        ),
        (nel.EpochArray([[0, 1]]), formatters.PrettyDuration, "s", "epoch"),
        (nel.SpaceArray([[0, 1]]), formatters.PrettySpace, "cm", "interval"),
    ],
)
def test_stale_intervalarray_pickle_restores_display_metadata(
    obj, formatter, base_unit, interval_label
):
    loaded = _roundtrip_without_attrs(
        obj,
        ["__version__", "type_name", "_interval_label", "formatter", "base_unit"],
    )

    assert loaded.__version__ == nel.__version__
    assert loaded.type_name == loaded.__class__.__name__
    assert loaded._interval_label == interval_label
    assert loaded.formatter is formatter
    assert loaded.base_unit == base_unit
