#encoding : utf-8
"""This module implements filtering functionailty for core nelpy objects
"""

__all__ = ['sosfiltfilt']

import copy
import numpy as np
import warnings

from .core import AnalogSignalArray

def sosfiltfilt(asa, *, fl=None, fh=None, fs=None, inplace=False, bandstop=False,
                gpass=None, gstop=None, ftype='cheby2', buffer_len=4194304,
                overlap_len=None, max_len=None, **kwargs):
    """Zero-phase forward backward second-order-segment Chebyshev II filter.

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
    asa : nelpy.core.AnalogSignalArray (preferred), ndarray, or list
        Object or data to filter.
    fs : float, optional only if AnalogSignalArray is passed
        The sampling frequency (Hz). Obtained from asa.
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
    buffer_len : int, optional
        How much data to process at a time. Default is 2**22 = 4194304 samples.
    overlap_len : int, optional
        How much data do we add to the end of each chunk to smooth out filter
        transients
    max_len : int, optional
        When max_len == -1 or max_len == None, then argument is effectively
        ignored. If max_len is a positive integer, thenmax_len specifies how
        many samples to process.

    Returns
    -------
    out : nelpy.core.AnalogSignalArray, ndarray, or list
        Same output type as input asa.
    """

    # make sure that fs is specified, unless AnalogSignalArray is passed in
    if isinstance(asa, (np.ndarray, list)):
        if fs is None:
            raise ValueError("sampling frequency, fs, must be specified!")
    elif isinstance(asa, AnalogSignalArray):
        if fs is None:
            fs = asa.fs
    else:
        raise TypeError('unsupported input type!')

    try:
        assert fh < fs, "fh must be less than sampling rate!"
    except TypeError:
        pass
    try:
        assert fl < fh, "fl must be less than fh!"
    except TypeError:
        pass

    from scipy.signal import sosfiltfilt, iirdesign

    if inplace:
        out = asa
    else:
        from copy import deepcopy
        out = deepcopy(asa)

    if overlap_len is None:
        overlap_len = int(fs*2)

    buffer_len = 4194304
    if gpass is None:
        gpass = 0.1 # max loss in passband, dB

    if gstop is None:
        gstop = 30 # min attenuation in stopband (dB)

    fso2 = fs/2.0

    try:
        if np.isinf(fh):
            fh = None
    except TypeError:
        pass
    if fl == 0:
        fl = None

    if (fl is None) and (fh is None):
        raise ValueError('nonsensical all-pass filter requested...')
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

    sos = iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype='cheby2', output='sos')

    if isinstance(asa, (np.ndarray, list)):
        if len(np.array(out).squeeze().shape) > 1:
            raise NotImplementedError('filtering for multidimensional ndarrays and lists not yet implemented; use an AnalogSignalArray, or a single dimensional list or ndarray')
        # ignore epochs (information not contained in list or array) so filter directly
        dims = np.array(out).shape
        out = np.squeeze(out)
        start, stop = 0, np.array(out).shape[-1]
        for buff_st_idx in range(start, stop, buffer_len):
                chk_st_idx = int(max(start, buff_st_idx - overlap_len))
                buff_nd_idx = int(min(stop, buff_st_idx + buffer_len))
                chk_nd_idx = int(min(stop, buff_nd_idx + overlap_len))
                rel_st_idx = int(buff_st_idx - chk_st_idx)
                rel_nd_idx = int(buff_nd_idx - chk_st_idx)
                this_y_chk = sosfiltfilt(sos, out[chk_st_idx:chk_nd_idx])
                out[buff_st_idx:buff_nd_idx] = this_y_chk[rel_st_idx:rel_nd_idx]
        out = np.reshape(out, dims)
        if isinstance(asa, list):
            out = out.tolist()
    elif isinstance(asa, AnalogSignalArray):
        # filter within epochs
        fei = np.insert(np.cumsum(out.lengths), 0, 0) # filter epoch indices, fei
        for ii in range(len(fei)-1):
            start, stop = fei[ii], fei[ii+1]
            for buff_st_idx in range(start, stop, buffer_len):
                chk_st_idx = int(max(start, buff_st_idx - overlap_len))
                buff_nd_idx = int(min(stop, buff_st_idx + buffer_len))
                chk_nd_idx = int(min(stop, buff_nd_idx + overlap_len))
                rel_st_idx = int(buff_st_idx - chk_st_idx)
                rel_nd_idx = int(buff_nd_idx - chk_st_idx)
                this_y_chk = sosfiltfilt(sos, out._ydata_rowsig[:,chk_st_idx:chk_nd_idx])
                out._ydata[:,buff_st_idx:buff_nd_idx] = this_y_chk[:,rel_st_idx:rel_nd_idx]
    return out
