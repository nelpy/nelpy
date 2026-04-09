import logging

import nelpy as nel
from nelpy import utils


def test_eventarray_support_clipping_logs_info(caplog):
    ea = nel.EventArray(
        abscissa_vals=[[0.1, 0.3, 1.1]],
        support=nel.EpochArray([0, 2]),
        fs=1,
    )

    with caplog.at_level(logging.INFO, logger="nelpy"):
        _ = ea[nel.EpochArray([0, 1])]

    records = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "ignoring events outside of eventarray support"
    ]
    assert records
    assert all(rec.levelno == logging.INFO for rec in records)


def test_analogsignalarray_support_clipping_logs_info(caplog):
    asa = nel.AnalogSignalArray([0, 1, 2, 3, 4], fs=1)

    with caplog.at_level(logging.INFO, logger="nelpy"):
        _ = asa[nel.EpochArray([0, 3])]

    records = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "ignoring signal outside of support"
    ]
    assert records
    assert all(rec.levelno == logging.INFO for rec in records)


def test_nelpy_logger_level_controls_support_clipping_noise(caplog):
    ea = nel.EventArray(
        abscissa_vals=[[0.1, 0.3, 1.1]],
        support=nel.EpochArray([0, 2]),
        fs=1,
    )

    nelpy_logger = logging.getLogger("nelpy")
    original_level = nelpy_logger.level

    try:
        nelpy_logger.setLevel(logging.ERROR)
        with caplog.at_level(logging.INFO):
            _ = ea[nel.EpochArray([0, 1])]

        records = [
            rec
            for rec in caplog.records
            if rec.getMessage() == "ignoring events outside of eventarray support"
        ]
        assert not records
    finally:
        nelpy_logger.setLevel(original_level)


def test_rsasa_init_estimated_fs_logs_info(caplog):
    with caplog.at_level(logging.INFO, logger="nelpy"):
        _ = nel.AnalogSignalArray(data=[0, 1, 2, 3], abscissa_vals=[0, 0.25, 0.5, 0.75])

    messages = [rec.getMessage() for rec in caplog.records]
    assert "fs was not specified, so we try to estimate it from the data..." in messages
    assert any(msg.startswith("fs was estimated to be ") for msg in messages)

    fs_records = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "fs was not specified, so we try to estimate it from the data..."
        or rec.getMessage().startswith("fs was estimated to be ")
    ]
    assert fs_records
    assert all(rec.levelno == logging.INFO for rec in fs_records)


def test_asa_init_does_not_emit_fs_deprecated_message(caplog):
    with caplog.at_level(logging.INFO, logger="nelpy"):
        _ = nel.AnalogSignalArray(data=[0, 1, 2, 3], abscissa_vals=[0, 0.25, 0.5, 0.75], fs=4)

    messages = [rec.getMessage() for rec in caplog.records]
    assert "'fs' has been deprecated; use 'step' instead" not in messages


def test_get_contiguous_segments_fs_and_step_messages_are_info(caplog):
    with caplog.at_level(logging.INFO, logger="nelpy"):
        _ = utils.get_contiguous_segments([0.0, 0.2, 0.4], fs=5)
        _ = utils.get_contiguous_segments([0.0, 0.2, 0.4], step=0.5)

    fs_deprecated = [
        rec for rec in caplog.records if rec.getMessage() == "'fs' has been deprecated; use 'step' instead"
    ]
    step_small = [
        rec
        for rec in caplog.records
        if rec.getMessage() == "some steps in the data are smaller than the requested step size."
    ]

    assert fs_deprecated
    assert step_small
    assert all(rec.levelno == logging.INFO for rec in fs_deprecated + step_small)
