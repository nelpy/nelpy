#encoding : utf-8
"""This module implements filtering functionailty for core nelpy objects. 
"""

__all__ = ['sosfiltfilt']

import ctypes
import sys
import warnings
import numpy as np
import scipy.signal as sig
import time

from . import core
from copy import deepcopy
from itertools import repeat
from multiprocessing import Process, Array, cpu_count
from multiprocessing.pool import Pool

def sosfiltfilt(timeseries, *, fl=None, fh=None, fs=None, inplace=False, bandstop=False,
                gpass=None, gstop=None, ftype=None, buffer_len=None, overlap_len=None,
                parallel=True,
                **kwargs):
    """Zero-phase forward backward filtering using second-order-segments.

    # spike  600--6000
    # ripple 150--250
    # delta 1--4
    # theta 6--12
    # gamma 32--100

    # Delta wave – (0.1 – 3 Hz)
    # Theta wave – (4 – 7 Hz)
    # Alpha wave – (8 – 15 Hz)
    # Mu wave – (7.5 – 12.5 Hz)
    # SMR wave – (12.5 – 15.5 Hz)
    # Beta wave – (16 – 31 Hz)
    # Gamma wave – (32 – 100 Hz)

    # slow gamma : 10–50 Hz
    # hippocampal theta : 6–10 Hz
    # motionless but alert theta : 6–7 Hz
    # cat & rabbit theta: 4-6 Hz meow! hop!

    Parameters
    ----------
    timeseries : nelpy.RegularlySampledAnalogSignalArray (preferred), ndarray, or list
        Object or data to filter.
    fs : float, optional only if RegularlySampledAnalogSignalArray is passed
        The sampling frequency (Hz). Obtained from the input timeseries.
    fl : float, optional
        Lower cut-off frequency (in Hz), 0 or None to ignore. Default is None.
    fh : float, optional
        Upper cut-off frequency (in Hz), np.inf or None to ignore. Default is None.
    bandstop : boolean, optional
        If False, passband is between fl and fh. If True, stopband is between
        fl and fh. Default is False.
    gpass : float, optional
        The maximum loss in the passband (dB). Default is 0.1 dB.
    gstop : float, optional
        The minimum attenuation in the stopband (dB). Default is 30 dB.
    ftype : str, optional
        The type of IIR filter to design:
            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2' (Default)
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'
    buffer_len : int, optional
        How much data to process at a time. Default is 2**22 = 4194304 samples.
    overlap_len : int, optional
        How much data we add to the end of each chunk to smooth out filter
        transients.
    kwargs : optional
        Other keyword arguments are passed to scipy.signal's iirdesign method

    Returns
    -------
    out : nelpy.RegularlySampledAnalogSignalArray, ndarray, or list
        Same output type as input timeseries.

    WARNING : The data type of the output object is the same as that of the input.
    Thus it is highly recommended to have your input data be floats before calling
    this function. If the input is an RSASA, you do not need to worry because
    the underlying data are already floats.
    """

    # make sure that fs is specified, unless AnalogSignalArray is passed in
    if isinstance(timeseries, (np.ndarray, list)):
        if fs is None:
            raise ValueError("Sampling frequency, fs, must be specified!")
    elif isinstance(timeseries, core.RegularlySampledAnalogSignalArray):
        if fs is None:
            fs = timeseries.fs
    else:
        raise TypeError('Unsupported input type!')

    try:
        assert fh < fs, "fh must be less than sampling rate!"
    except TypeError:
        pass
    try:
        assert fl < fh, "fl must be less than fh!"
    except TypeError:
        pass

    if inplace:
        out = timeseries
    else:
        out = deepcopy(timeseries)
    if overlap_len is None:
        overlap_len = int(fs*2)
    if buffer_len is None:
        buffer_len = 4194304
    if gpass is None:
        gpass = 0.1 # max loss in passband, dB
    if gstop is None:
        gstop = 30 # min attenuation in stopband (dB)
    if ftype is None:
        ftype = 'cheby2'

    try:
        if np.isinf(fh):
            fh = None
    except TypeError:
        pass
    if fl == 0:
        fl = None

    # Handle cutoff frequencies
    fso2 = fs/2.0
    if (fl is None) and (fh is None):
        raise ValueError('Nonsensical all-pass filter requested...')
    elif fl is None: # lowpass
        wp = fh/fso2
        ws = 1.4*fh/fso2
    elif fh is None: # highpass
        wp = fl/fso2
        ws = 0.8*fl/fso2
    else: # bandpass
        wp = [fl/fso2, fh/fso2] 
        ws = [0.8*fl/fso2,1.4*fh/fso2]
    if bandstop: # notch / bandstop filter
        wp, ws = ws, wp

    sos = sig.iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype=ftype, output='sos', **kwargs)

    # Prepare input and output data structures
    # Output array lives in shared memory and will reduce overhead from pickling/de-pickling
    # data if we're doing parallelized filtering
    if isinstance(timeseries, (np.ndarray, list)):
        temp_array = np.array(timeseries)
        dims = temp_array.shape
        if len(temp_array.shape) > 2:
            raise NotImplementedError('Filtering for >2D ndarray or list is not implemented')
        shared_array_base = Array(ctypes.c_double, temp_array.size, lock=False)
        shared_array_out = np.ctypeslib.as_array(shared_array_base)
        # Force input and output arrays to be 2D (N x T) where N is number of signals
        # and T is number of time points
        if len(temp_array.squeeze().shape) == 1:
            shared_array_out = np.ctypeslib.as_array(shared_array_base).reshape((1, temp_array.size))
            input_asarray = temp_array.reshape((1, temp_array.size))
        else:
            shared_array_out = np.ctypeslib.as_array(shared_array_base).reshape(dims)
            input_asarray = temp_array
    elif isinstance(timeseries, core.RegularlySampledAnalogSignalArray):
        dims = timeseries._data.shape
        shared_array_base = Array(ctypes.c_double, timeseries._data_rowsig.size, lock=False)
        shared_array_out = np.ctypeslib.as_array(shared_array_base).reshape(dims)
        input_asarray = timeseries._data

    # Embedded function to avoid pickling data but need global to make this function
    # module-visible (required by multiprocessing). I know, a bit of a hack
    global filter_chunk
    def filter_chunk(it):
        """The function that performs the chunked filtering"""

        try:
            start, stop, buffer_len, overlap_len, buff_st_idx = it
            buff_nd_idx = int(min(stop, buff_st_idx + buffer_len))
            chk_st_idx = int(max(start, buff_st_idx - overlap_len))
            chk_nd_idx = int(min(stop, buff_nd_idx + overlap_len))
            rel_st_idx = int(buff_st_idx - chk_st_idx)
            rel_nd_idx = int(buff_nd_idx - chk_st_idx)
            this_y_chk = sig.sosfiltfilt(sos, input_asarray[:, chk_st_idx:chk_nd_idx], axis=1)
            shared_array_out[:,buff_st_idx:buff_nd_idx] = this_y_chk[:, rel_st_idx:rel_nd_idx]
        except:
            raise ValueError(("Some epochs were too short to filter. Try dropping those first,"
                              " filtering, and then inserting them back in"))

    # Do the actual parallellized filtering
    t0 = time.time()
    if (sys.platform.startswith('linux') or sys.platform.startswith('darwin')) and parallel:
        pool = Pool(processes=cpu_count())
        if isinstance(timeseries, (np.ndarray, list)):
            # ignore epochs (information not contained in list or array) so filter directly
            start, stop = 0, input_asarray.shape[1]
            pool.map(filter_chunk, zip(repeat(start), repeat(stop), repeat(buffer_len),
                                     repeat(overlap_len), range(start, stop, buffer_len)),
                                    chunksize=1)
        elif isinstance(timeseries, core.RegularlySampledAnalogSignalArray):
            fei = np.insert(np.cumsum(timeseries.lengths), 0, 0) # filter epoch indices, fei
            for ii in range(len(fei)-1): # filter within epochs
                start, stop = fei[ii], fei[ii+1]
                pool.map(filter_chunk, zip(repeat(start), repeat(stop), repeat(buffer_len),
                         repeat(overlap_len), range(start, stop, buffer_len)),
                         chunksize=1)
        pool.close()
        pool.join()
    # No easy parallelized filtering for other OSes
    else:
        if isinstance(timeseries, (np.ndarray, list)):
            # ignore epochs (information not contained in list or array) so filter directly
            start, stop = 0, input_asarray.shape[1]
            iterator = zip(repeat(start), repeat(stop), repeat(buffer_len),
                           repeat(overlap_len), range(start, stop, buffer_len))
            for item in iterator:
                filter_chunk(item)
        elif isinstance(timeseries, core.RegularlySampledAnalogSignalArray):
            fei = np.insert(np.cumsum(timeseries.lengths), 0, 0) # filter epoch indices, fei
            for ii in range(len(fei)-1): # filter within epochs
                start, stop = fei[ii], fei[ii+1]
                iterator = zip(repeat(start), repeat(stop), repeat(buffer_len),
                               repeat(overlap_len), range(start, stop, buffer_len))
                for item in iterator:
                    filter_chunk(item)
    print(time.time() - t0)

    if isinstance(timeseries, np.ndarray):
        out[:] = np.reshape(shared_array_out, dims)
    elif isinstance(timeseries, list):
        out[:] = np.reshape(shared_array_out, dims).tolist()
    elif isinstance(timeseries, core.RegularlySampledAnalogSignalArray):
        out._data[:] = shared_array_out

    return out

def getsos(*, fs, fl=None, fh=None, bandstop=False,
                gpass=None, gstop=None, ftype='cheby2'):
    """Return second-order sections representation of the IIR filter.

    This is useful to plot the frequency response using scipy.signal.sosfreqz,
    for example:

    >>> import matplotlib.pyplot as plt
    >>> from scipy import signal
    >>>
    >>> sos = getsos(...)
    >>> w, h = signal.sosfreqz(sos, worN=1500)
    >>> db = 20*np.log10(np.abs(h))
    >>> freq = w*fs/(2*np.pi)
    >>> plt.subplot(2, 1, 1)
    >>> plt.ylabel('Gain [dB]')
    >>> plt.plot(freq, db)
    >>> plt.subplot(2, 1, 2)
    >>> plt.plot(freq, np.angle(h))
    >>> plt.ylabel('Phase [rad]')

    Although not currently supported, filters can be stacked as well, as follows:
    >>> sos = np.vstack((nel.filtering.getsos(fs=T2.fs, fl=150, fh=250, gstop=10, ftype='cheby2'),\
                 nel.filtering.getsos(fs=T2.fs, fl=150, fh=250, gstop=10, ftype='cheby2'),\
                 nel.filtering.getsos(fs=T2.fs, fl=150, fh=250, gstop=10, ftype='cheby2'),\
                 nel.filtering.getsos(fs=T2.fs, fl=150, fh=250, gstop=1, ftype='butter')))

    Parameters
    ----------
    fs : float
        The sampling frequency (Hz).
    fl : float, optional
        Lower cut-off frequency (in Hz), 0 or None to ignore. Default is None.
    fh : float, optional
        Upper cut-off frequency (in Hz), 0 or None to ignore. Default is None.
    bandstop : boolean, optional
        If False, passband is between fl and fh. If True, stopband is between
        fl and fh. Default is False.
    gpass : float, optional
        The maximum loss in the passband (dB). Default is 0.1 dB.
    gstop : float, optional
        The minimum attenuation in the stopband (dB). Default is 30 dB.
    ftype : str, optional
        The type of IIR filter to design:
            - Butterworth   : 'butter'
            - Chebyshev I   : 'cheby1'
            - Chebyshev II  : 'cheby2' (Default)
            - Cauer/elliptic: 'ellip'
            - Bessel/Thomson: 'bessel'

    Returns
    -------
    sos : ndarray
        Second-order sections representation of the IIR filter.
    """

    try:
        assert fh < fs, "fh must be less than sampling rate!"
    except TypeError:
        pass
    try:
        assert fl < fh, "fl must be less than fh!"
    except TypeError:
        pass

    if gpass is None:
        gpass = 0.1 # max loss in passband, dB
    if gstop is None:
        gstop = 30 # min attenuation in stopband (dB)

    try:
        if np.isinf(fh):
            fh = None
    except TypeError:
        pass
    if fl == 0:
        fl = None

    # Handle cutoff frequencies
    fso2 = fs/2.0
    if (fl is None) and (fh is None):
        raise ValueError('Nonsensical all-pass filter requested...')
    elif fl is None: # lowpass
        wp = fh/fso2
        ws = 1.4*fh/fso2
    elif fh is None: # highpass
        wp = fl/fso2
        ws = 0.8*fl/fso2
    else: # bandpass
        wp = [fl/fso2, fh/fso2]
        ws = [0.8*fl/fso2,1.4*fh/fso2]
    if bandstop: # notch / bandstop filter
        wp, ws = ws, wp

    sos = sig.iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype=ftype, output='sos')

    return sos