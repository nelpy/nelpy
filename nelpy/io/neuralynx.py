"""Loads data stored in the formats used by the Neuralynx recording systems."""
# Adapted from nlxio written by Bernard Willards <https://github.com/bwillers/nlxio>

import numpy as np

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

    with open(filename, 'rb') as f:

        # A tetrode spike record is as folows:
        # uint64 - timestamp                    bytes 0:8
        # uint32 - acquisition entity number    bytes 8:12
        # uint32 - classified cell number       bytes 12:16
        # 8 * uint32- params                    bytes 16:48
        # 32 * 4 * int16 - waveform points
        # hence total record size is 2432 bits, 304 bytes

        # header is 16kbyte, i.e. 16 * 2^10 = 2^14
        header = f.read(2 ** 14).strip(b'\x00')

        # Read the header and find the conversion factors / sampling frequency
        analog_to_digital = None
        fs = None

        for line in header.split(b'\n'):
            if line.strip().startswith(b'-ADBitVolts'):
                analog_to_digital = np.array(float(line.split(b' ')[1].decode()))
            if line.strip().startswith(b'-SamplingFrequency'):
                fs = float(line.split(b' ')[1].decode())

        f.seek(2 ** 14)  # start of the spike, records
        # Neuralynx writes little endian for some reason
        dt = np.dtype([('spiketimes', '<Q'), ('acq', '<i', 1), ('cellnums', '<i', 1), ('params', '<i', 8),
                    ('waveforms', np.dtype('<h'), (32, 4))])
        data = np.fromfile(f, dt)

        if analog_to_digital is None:
            raise IOError("ADBitVolts not found in .ntt header for " + filename)
        if fs is None:
            raise IOError("Frequency not found in .ntt header for " + filename)

    return data['spiketimes'] / 1e6, data['waveforms'] * analog_to_digital, fs, data['cellnums']