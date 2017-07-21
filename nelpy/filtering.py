#encoding : utf-8
"""This module implements filtering functionailty for core nelpy objects
"""

# NOTE: I found a really good website + implementation of doing out-of-core
# chunked signal filtering in Python that was scalable and efficient,
# but I have since lost the url (I mad a note, but can't find the note).
# Frustrating as that is, here are some other pages to check out:
#
# http://codereview.stackexchange.com/questions/88885/efficiently-filter-a-large-100gb-csv-file-v3
#
# https://www.airpair.com/python/posts/top-mistakes-python-big-data-analytics (see cythonmagic!)
#
# https://github.com/kghose/neurapy/blob/master/neurapy/signal/continuous.py (FFB!)
#
# https://pypi.python.org/pypi/out-of-core-fft/1.0
#
# http://matthewrocklin.com/blog/work/2015/02/17/Towards-OOC-Bag

__all__ = ['butter_bandpass_filter',
           'butter_lowpass_filtfilt',]

import copy
import numpy as np
import warnings

from scipy.signal import butter, lfilter, filtfilt, firwin
from math import log10, ceil

from .core import AnalogSignalArray

def butter_bandpass(lowcut, highcut, fs, order=5):
    """Returns a bandpass butterworth filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, *, lowcut, highcut, fs, order=5):
    """Band filter data using a butterworth filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    """Returns a lowpass butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filtfilt(data, *, cutoff, fs, order=5):
    """Lowpass filter data using a zero-phase filt-filt butterworth
    filter.

    Performs zero-phase digital filtering by processing the input data
    in both the forward and reverse directions.
    """
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data, padlen=150)
    return y

def bandpass_filter(data, lowcut=None, highcut=None, *, numtaps=None,
                    fs=None):
    """Band filter data using a zero phase FIR filter (filtfilt).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 1 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 600 Hz)
        Upper cut-off frequency
    numtaps : int, optional (default 25)
        Number of filter taps
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """

    if numtaps is None:
        numtaps = 25
    if lowcut is None:
        lowcut = 1
    if highcut is None:
        highcut = 600

    if isinstance(data, (np.ndarray, list)):
        if fs is None:
            raise ValueError("sampling frequency must be specified!")
        # Generate filter for detection
        b = firwin(numtaps=numtaps,
                   cutoff=[lowcut/(fs/2), highcut/(fs/2)],
                   pass_zero=False)
        # Filter raw data to get ripple data
        ripple_data = filtfilt(b, 1, data)
        return ripple_data
    elif isinstance(data, AnalogSignalArray):
        if fs is None:
            fs = data.fs
            warnings.warn("no sampling frequency provided,"
                " using fs={} Hz from AnalogSignalArray".format(fs))
        # Generate filter for detection
        b = firwin(numtaps=numtaps,
                   cutoff=[lowcut/(fs/2), highcut/(fs/2)],
                   pass_zero=False)
        # Filter raw data to get ripple data
        ripple_data = filtfilt(b,1,data.ydata)
        # Return a copy of the AnalogSignalArray with the filtered data
        filtered_analogsignalarray = data.copy()
        filtered_analogsignalarray._ydata = ripple_data
        return filtered_analogsignalarray
    else:
        raise TypeError(
          "Unknown data type {} to filter.".format(str(type(data))))

def spike_filter(data, lowcut=None, highcut=None, *, fs=None, verbose=False):
    """Filter data to the spike band (default 600--6000 Hz).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 600 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 6000 Hz)
        Upper cut-off frequency
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """

    if isinstance(data, (np.ndarray, list)):
        if fs is None:
            raise ValueError("sampling frequency must be specified!")
    elif isinstance(data, AnalogSignalArray):
        if fs is None:
            fs = data.fs

    if lowcut is None:
        lowcut = 600
    if highcut is None:
        highcut = 6000

    [b, a] = butter(2, lowcut/(fs/2), btype='highpass')
    [bhigh, ahigh] = butter(1, highcut/(fs/2))

    if isinstance(data, (np.ndarray, list)):
        # Filter raw data
        spikedata = lfilter(b, a, lfilter(bhigh, ahigh, data))
        return spikedata
    elif isinstance(data, AnalogSignalArray):
        spikedata = lfilter(b, a, lfilter(bhigh, ahigh, data.ydata))

        # Return a copy of the AnalogSignalArray with the filtered data
        out = copy.copy(data)
        out._ydata = spikedata
        return out

def spike_filtfilt(data, lowcut=None, highcut=None, *, fs=None, verbose=False):
    """Filter data to the spike band (default 600--6000 Hz).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 600 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 6000 Hz)
        Upper cut-off frequency
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """

    if isinstance(data, (np.ndarray, list)):
        if fs is None:
            raise ValueError("sampling frequency must be specified!")
    elif isinstance(data, AnalogSignalArray):
        if fs is None:
            fs = data.fs

    if lowcut is None:
        lowcut = 600
    if highcut is None:
        highcut = 6000

    [b, a] = butter(2, lowcut/(fs/2), btype='highpass')
    [bhigh, ahigh] = butter(1, highcut/(fs/2))

    if isinstance(data, (np.ndarray, list)):
        # Filter raw data
        spikedata = filtfilt(b, a,filtfilt(bhigh, ahigh, data))
        return spikedata
    elif isinstance(data, AnalogSignalArray):
        spikedata = filtfilt(b, a, filtfilt(bhigh, ahigh, data.ydata))

        # Return a copy of the AnalogSignalArray with the filtered data
        out = copy.copy(data)
        out._ydata = spikedata
        return out

def ripple_band_filter(data, lowcut=None, highcut=None, *, numtaps=None,
                       fs=None, verbose=False):
    """Filter data to the ripple band (default 150--250 Hz).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 150 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 250 Hz)
        Upper cut-off frequency
    numtaps : int, optional (default automatically determined)
        Number of filter taps
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """
    if numtaps is None:
        if isinstance(data, (np.ndarray, list)):
            if fs is None:
                raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, AnalogSignalArray):
            if fs is None:
                fs = data.fs
        numtaps = approx_number_of_taps(fs=fs,
                                        delta_f=20,
                                        delta1=10e-2,
                                        delta2=10e-2)
        if verbose:
            print("Filtering with {} taps.".format(numtaps))
    if lowcut is None:
        lowcut = 150
    if highcut is None:
        highcut = 250
    return bandpass_filter(data,
                           lowcut=lowcut,
                           highcut=highcut,
                           numtaps=numtaps,
                           fs=fs)

def approx_number_of_taps(fs, delta_f, delta1=None, delta2=None):
    """Docstring goes here.
    http://dsp.stackexchange.com/questions/31066/how-many-taps-does-an-fir-filter-need

    Parameters
    ----------
    fs : float
        Sampling frequency (Hz)
    delta_f : float
        transition width; difference between end of pass band and start
        of stop band, in Hz
    delta1 : float, optional (default is 1% ==> 0.01 ==> 10e-3)
        ripple in passband
    delta2 : float, optional (default is -30 dB ==> 10e-3)
        suppression in the stopband

    Returns
    -------
    numtaps : int
        number of FIR filter taps
    """
    if delta1 is None:
        delta1 = 10e-3
    if delta2 is None:
        delta2 = 10e-3

    numtaps = ceil(2*log10(1/(10*delta1*delta2))*fs/delta_f/3)
    return numtaps

def delta_band_filter(data, lowcut=None, highcut=None, *, numtaps=None,
                       fs=None, verbose=False):
    """Filter data to the rodent delta band (default 1--4 Hz).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 1 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 4 Hz)
        Upper cut-off frequency
    numtaps : int, optional (default determined automatically)
        Number of filter taps
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """

    if numtaps is None:
        if isinstance(data, (np.ndarray, list)):
            if fs is None:
                raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, AnalogSignalArray):
            if fs is None:
                fs = data.fs
        numtaps = approx_number_of_taps(fs=fs,
                                        delta_f=1,
                                        delta1=10e-3,
                                        delta2=10e-3)
        if verbose:
            print("Filtering with {} taps.".format(numtaps))

    if lowcut is None:
        lowcut = 1
    if highcut is None:
        highcut = 4
    return bandpass_filter(data,
                           lowcut=lowcut,
                           highcut=highcut,
                           numtaps=numtaps,
                           fs=fs)

def theta_band_filter(data, lowcut=None, highcut=None, *, numtaps=None,
                       fs=None, verbose=False):
    """Filter data to the rodent theta band (default 6--12 Hz).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 6 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 12 Hz)
        Upper cut-off frequency
    numtaps : int, optional (default determined automatically)
        Number of filter taps
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """

    if numtaps is None:
        if isinstance(data, (np.ndarray, list)):
            if fs is None:
                raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, AnalogSignalArray):
            if fs is None:
                fs = data.fs
        numtaps = approx_number_of_taps(fs=fs,
                                        delta_f=1,
                                        delta1=10e-3,
                                        delta2=10e-3)
        if verbose:
            print("Filtering with {} taps.".format(numtaps))

    if lowcut is None:
        lowcut = 6
    if highcut is None:
        highcut = 12
    return bandpass_filter(data,
                           lowcut=lowcut,
                           highcut=highcut,
                           numtaps=numtaps,
                           fs=fs)

def gamma_band_filter(data, lowcut=None, highcut=None, *, numtaps=None,
                       fs=None, verbose=False):
    """Filter data to the rodent gamma band (default 32--100 Hz).

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    lowcut : float, optional (default 32 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 100 Hz)
        Upper cut-off frequency
    numtaps : int, optional (default determined automatically)
        Number of filter taps
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)

    Returns
    -------
    filtered : same type as data
    """

    if numtaps is None:
        if isinstance(data, (np.ndarray, list)):
            if fs is None:
                raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, AnalogSignalArray):
            if fs is None:
                fs = data.fs
        numtaps = approx_number_of_taps(fs=fs,
                                        delta_f=1,
                                        delta1=10e-3,
                                        delta2=10e-3)
        if verbose:
            print("Filtering with {} taps.".format(numtaps))

    if lowcut is None:
        lowcut = 32
    if highcut is None:
        highcut = 100
    return bandpass_filter(data,
                           lowcut=lowcut,
                           highcut=highcut,
                           numtaps=numtaps,
                           fs=fs)

def filter_lfp(data, band=None, *, lowcut=None, highcut=None,
               numtaps=None, fs=None, verbose=False):
    """Filter data with a zero phase FIR filtfilt filter.

    This is a convenience wrapper function for
        ripple_band_filter()
        theta_band_filter()
        delta_band_filter()
        ...

    Parameters
    ----------
    data : AnalogSignalArray, ndarray, or list
    band : string, optional
        One of ['ripple', 'theta', 'delta', ...]
        Defaults to 'ripple'.
    lowcut : float, optional (default 6 Hz)
        Lower cut-off frequency
    highcut : float, optional (default 12 Hz)
        Upper cut-off frequency
    numtaps : int, optional (default determined automatically)
        Number of filter taps
    fs : float, optional if AnalogSignalArray is passed
        Sampling frequency (Hz)
    verbose : bool, optional

    Returns
    -------
    filtered : same type as data.
    """
    supported_bands = ['ripple', 'delta', 'theta', 'gamma']


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

    if band is None:
        band = 'ripple'
    else:
        band = band.strip().lower()

    if band not in supported_bands:
        raise NotImplementedError("filter_lfp not supported or not yet implemented for band '{}'".format(str(band)))

    kwargs = {'data' : data,
              'lowcut' : lowcut,
              'highcut' : highcut,
              'numtaps' : numtaps,
              'fs' : fs,
              'verbose' : verbose}

    if band == 'ripple':
        return ripple_band_filter(**kwargs)
    if band == 'theta':
        return theta_band_filter(**kwargs)
    if band == 'delta':
        return delta_band_filter(**kwargs)
    if band == 'gamma':
        return gamma_band_filter(**kwargs)

    return 0

########################################################################
# uncurated below this line!
########################################################################

# taken from https://github.com/kghose/neurapy/blob/master/neurapy/signal/continuous.py

"""Some methods for dealing with continuous data. We assume that the original data is in files and that they are
annoyingly large. So all the methods here work on buffered input, using memory maps.
"""
import pylab
from scipy.signal import filtfilt, iirdesign

#Some useful presets for loading continuous data dumped from the Neuralynx system
lynxlfp = {
    'fmt': 'i',
    'fs' : 32556,
    'fl' : 5,
    'fh' : 100,
    'gpass' : 0.1,
    'gstop' : 15,
    'buffer_len' : 100000,
    'overlap_len': 100,
    'max_len': -1
}

lynxspike = {
    'fmt': 'i',
    'fs' : 32556,
    'fl' : 500,
    'fh' : 8000,
    'gpass' : 0.1,
    'gstop' : 15,
    'buffer_len' : 100000,
    'overlap_len': 100,
    'max_len': -1
}
"""Use these presets as follows
from neurapy.utility import continuous as cc
y,b,a = cc.butterfilt('chan_000.raw', 'test.raw', **cc.lynxlfp)"""


def butterfilt(finname, foutname, fmt, fs, fl=5.0, fh=100.0, gpass=1.0, gstop=30.0, ftype='butter', buffer_len=100000, overlap_len=100, max_len=-1):
    """Given sampling frequency, low and high pass frequencies design a butterworth filter and filter our data with it."""
    fso2 = fs/2.0
    wp = [fl/fso2, fh/fso2]
    ws = [0.8*fl/fso2,1.4*fh/fso2]
    import pdb; pdb.set_trace()
    b, a = iirdesign(wp, ws, gpass=gpass, gstop=gstop, ftype=ftype, output='ba')
    y = filtfiltlong(finname, foutname, fmt, b, a, buffer_len, overlap_len, max_len)
    return y, b, a

def filtfiltlong(finname, foutname, fmt, b, a, buffer_len=100000, overlap_len=100, max_len=-1):
  """Use memmap and chunking to filter continuous data.
  Inputs:
    finname -
    foutname    -
    fmt         - data format eg 'i'
    b,a         - filter coefficients
    buffer_len  - how much data to process at a time
    overlap_len - how much data do we add to the end of each chunk to smooth out filter transients
    max_len     - how many samples to process. If set to -1, processes the whole file
  Outputs:
    y           - The memmapped array pointing to the written file
  Notes on algorithm:
    1. The arrays are memmapped, so we let pylab (numpy) take care of handling large arrays
    2. The filtering is done in chunks:
    Chunking details:
                |<------- b1 ------->||<------- b2 ------->|
    -----[------*--------------{-----*------]--------------*------}----------
         |<-------------- c1 -------------->|
                               |<-------------- c2 -------------->|
    From the array of data we cut out contiguous buffers (b1,b2,...) and to each buffer we add some extra overlap to
    make chunks (c1,c2). The overlap helps to remove the transients from the filtering which would otherwise appear at
    each buffer boundary.
  """
  x = pylab.memmap(finname, dtype=fmt, mode='r')
  if max_len == -1:
      max_len = x.size
  y = pylab.memmap(foutname, dtype=fmt, mode='w+', shape=max_len)

  for buff_st_idx in xrange(0, max_len, buffer_len):
      chk_st_idx = max(0, buff_st_idx - overlap_len)
      buff_nd_idx = min(max_len, buff_st_idx + buffer_len)
      chk_nd_idx = min(x.size, buff_nd_idx + overlap_len)
      rel_st_idx = buff_st_idx - chk_st_idx
      rel_nd_idx = buff_nd_idx - chk_st_idx
      this_y_chk = filtfilt(b, a, x[chk_st_idx:chk_nd_idx])
      y[buff_st_idx:buff_nd_idx] = this_y_chk[rel_st_idx:rel_nd_idx]

  return y