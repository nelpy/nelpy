"""Loads data stored in the formats used by the Neuralynx recording systems."""

# Adapted from nlxio written by Bernard Willards <https://github.com/bwillers/nlxio>

import numpy as np
from .. import auxiliary


def load_position(filename, pixels_per_cm=None):
    """Loads videotracking position as a nelpy PositionArray

    Parameters
    ----------
    filename: str
    pixel_per_cm: tuple
        With (x, y) conversion factors
    Returns
    -------
    position: nelpy.PositionArray
    """

    if not pixels_per_cm:
        pixels_per_cm = (1, 1)

    nvt_data = load_nvt(filename)

    xydata = np.vstack(
        (nvt_data["x"] / pixels_per_cm[0], nvt_data["y"] / pixels_per_cm[1])
    )
    timestamps = nvt_data["time"]
    pos = auxiliary.PositionArray(xydata, timestamps=timestamps)
    return pos


def load_nvt(filename):
    """Loads a neuralynx .nvt file.
    Parameters
    ----------
    filename: str
    Returns
    -------
    nvt_data: dict
        With time, x, and y as keys.
    """
    with open(filename, "rb") as f:

        # Neuralynx files have a 16kbyte header
        # header = f.read(2**14).strip(b"\x00")

        # The format for .nvt files according the the neuralynx docs is
        # uint16 - beginning of the record
        # uint16 - ID for the system
        # uint16 - size of videorec in bytes
        # uint64 - timestamp in microseconds
        # uint32 x 400 - points with the color bitfield values
        # int16 - unused
        # int32 - extracted X location of target
        # int32 - extracted Y location of target
        # int32 - calculated head angle in degrees clockwise from the positive Y axis
        # int32 x 50 - colored targets using the same bitfield format used to extract colors earlier
        dt = np.dtype(
            [
                ("filler1", "<h", 3),
                ("time", "<Q"),
                ("points", "<i", 400),
                ("filler2", "<h"),
                ("x", "<i"),
                ("y", "<i"),
                ("head_angle", "<i"),
                ("targets", "<i", 50),
            ]
        )
        data = np.fromfile(f, dt)

    nvt_data = dict()
    nvt_data["time"] = data["time"] * 1e-6
    nvt_data["x"] = np.array(data["x"], dtype=float)
    nvt_data["y"] = np.array(data["y"], dtype=float)
    nvt_data["head_angle"] = np.array(data["head_angle"], dtype=float)
    nvt_data["targets"] = np.array(data["targets"], dtype=float)

    empty_idx = (data["x"] == 0) & (data["y"] == 0)
    for key in nvt_data:
        nvt_data[key] = nvt_data[key][~empty_idx]

    return nvt_data


def load_ntt(filename, should_d2a=True):
    """Loads a neuralynx .ntt tetrode spike file.

    Parameters
    ----------
    filename: str
    should_d2a: convert from integer to microVolt units (default True)

    Returns
    -------
    spiketimes: np.array
        Spike times as floats (seconds)
    waveforms: np.array
        Spike waveforms as (num_spikes, length_waveform, num_channels)
    fs: float
        Sampling frequency (Hz)
    cellnums: cell numbers

    Usage:
    spiketimes, waveforms, frequency, cellnums = load_ntt('TT13.ntt')
    """

    with open(filename, "rb") as f:

        # A tetrode spike record is as folows:
        # uint64 - timestamp                    bytes 0:8
        # uint32 - acquisition entity number    bytes 8:12
        # uint32 - classified cell number       bytes 12:16
        # 8 * uint32- params                    bytes 16:48
        # 32 * 4 * int16 - waveform points
        # hence total record size is 2432 bits, 304 bytes

        # header is 16kbyte, i.e. 16 * 2^10 = 2^14
        header = f.read(2**14).strip(b"\x00")

        # Read the header and find the conversion factors / sampling frequency
        analog_to_digital = None
        fs = None

        for line in header.split(b"\n"):
            if line.strip().startswith(b"-ADBitVolts"):
                analog_to_digital = np.array(float(line.split(b" ")[1].decode()))
            if line.strip().startswith(b"-SamplingFrequency"):
                fs = float(line.split(b" ")[1].decode())

        f.seek(2**14)  # start of the spike, records
        # Neuralynx writes little endian for some reason
        dt = np.dtype(
            [
                ("spiketimes", "<Q"),
                ("acq", "<i", 1),
                ("cellnums", "<i", 1),
                ("params", "<i", 8),
                ("waveforms", np.dtype("<h"), (32, 4)),
            ]
        )
        data = np.fromfile(f, dt)

        if analog_to_digital is None:
            raise IOError("ADBitVolts not found in .ntt header for " + filename)
        if fs is None:
            raise IOError("Frequency not found in .ntt header for " + filename)

    return (
        data["spiketimes"] / 1e6,
        data["waveforms"] * analog_to_digital,
        fs,
        data["cellnums"],
    )
