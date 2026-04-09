import logging

import nelpy as nel


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
