"""Optional event metadata reads must not warn or alter the object."""

import logging

import numpy as np
import pytest

import nelpy as nel


def _event_object(kind, tags=None, label=None):
    kwargs = {
        "fs": 20000,
        "support": nel.EpochArray([0, 1]),
        "series_ids": [10, 20],
        "series_labels": ["CA1", "PFC"],
        "series_tags": tags,
        "label": label,
    }
    if kind in ("spikes", "binned_spikes"):
        obj = nel.SpikeTrainArray([[0.1, 0.4], [0.2, 0.5]], **kwargs)
    else:
        obj = nel.EventArray([[0.1, 0.4], [0.2, 0.5]], **kwargs)
    return obj.bin(ds=0.1) if kind.startswith("binned") else obj


@pytest.mark.parametrize("kind", ["events", "spikes", "binned_events", "binned_spikes"])
def test_missing_optional_metadata_is_quiet_and_unchanged(kind, caplog):
    obj = _event_object(kind)
    original = obj.copy()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            assert obj.series_tags is None
            assert obj.label is None
            if "spikes" in kind:
                assert obj.unit_tags is None
        copied = obj.copy()
        assert copied.series_tags is None
        assert copied.label is None
    assert not caplog.records
    assert obj.fs == original.fs
    assert obj.series_ids == original.series_ids
    assert obj.series_labels == original.series_labels
    np.testing.assert_array_equal(obj.support.data, original.support.data)
    for actual, expected in zip(obj.data, original.data):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", ["events", "spikes", "binned_events", "binned_spikes"])
def test_specified_optional_metadata_survives_copy_and_reconstruction(kind, caplog):
    tags = ["pyramidal", "interneuron"]
    obj = _event_object(kind, tags=tags, label="session")
    with caplog.at_level(logging.WARNING):
        assert obj.series_tags == tags
        assert obj.label == "session"
        copied = obj.copy()
        assert copied.series_tags == tags
        assert copied.label == "session"
        if "spikes" in kind:
            assert obj.unit_tags == tags
    assert not caplog.records
    raw = nel.SpikeTrainArray(
        [[0.1, 0.4], [0.2, 0.5]],
        fs=20000,
        support=nel.EpochArray([0, 1]),
        unit_ids=obj.series_ids,
        unit_labels=obj.series_labels,
        unit_tags=obj.series_tags,
        label=obj.label,
    )
    assert raw.unit_tags == tags
    assert raw.label == "session"
    raw.label = None
    with caplog.at_level(logging.WARNING):
        assert raw.label is None
    assert not caplog.records


@pytest.mark.parametrize("kind", ["events", "spikes", "binned_events", "binned_spikes"])
def test_missing_optional_metadata_remains_available_at_debug(kind, caplog):
    obj = _event_object(kind)
    with caplog.at_level(logging.DEBUG, logger="nelpy.core._eventarray"):
        assert obj.series_tags is None
        assert obj.label is None
    records = [
        record for record in caplog.records if record.name == "nelpy.core._eventarray"
    ]
    assert [record.getMessage() for record in records] == [
        "series tags have not yet been specified",
        "label has not yet been specified",
    ]
    assert all(record.levelno == logging.DEBUG for record in records)


def test_nelpy_logger_level_controls_optional_metadata_diagnostics(caplog):
    obj = _event_object("spikes")
    with caplog.at_level(logging.DEBUG):
        with caplog.at_level(logging.WARNING, logger="nelpy"):
            assert obj.unit_tags is None
            assert obj.label is None
    assert not caplog.records
