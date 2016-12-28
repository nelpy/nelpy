"""safemath.py -- helper functions for math-related code to handle
overflow and underflow

This software is licensed under the terms of the MIT License as
follows:

Copyright (c) 2013 Jessica B. Hamrick

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import numpy as np
import sys

from numpy import log, exp

# constants for largest/smallest log values we can handle
MIN_LOG = log(sys.float_info.min)
MAX_LOG = log(sys.float_info.max)
EPS = np.finfo(float).eps


def normalize(logarr, axis=-1):
    """Normalize an array of log-values.

    This function is very useful if you have an array of log
    probabilities that need to be normalized, but some of the
    probabilies might be extremely small (i.e., underflow will occur if
    you try to exponentiate them). This function computes the
    normalization constants in log space, thus avoiding the need to
    exponentiate the values.

    Parameters
    ----------
    logarr: numpy.ndarray
        Array of log values
    axis: integer (default=-1)
        Axis over which to normalize

    Returns
    -------
    out: (numpy.ndarray, numpy.ndarray)
        2-tuple consisting of the log normalization constants used to
        normalize the array, and the normalized array of log values

    """

    # shape for the normalization constants (that would otherwise be
    # missing axis)
    shape = list(logarr.shape)
    shape[axis] = 1
    # get maximum value of array
    maxlogarr = logarr.max(axis=axis).reshape(shape)
    # calculate how much to shift the array up by
    shift = MAX_LOG - maxlogarr - 2 - logarr.shape[axis]
    shift[shift < 0] = 0
    # shift the array
    unnormed = logarr + shift
    # convert from logspace
    arr = exp(unnormed)
    # calculate shifted log normalization constants
    _lognormconsts = log(arr.sum(axis=axis)).reshape(shape)
    # calculate normalized array
    lognormarr = unnormed - _lognormconsts
    # unshift normalization constants
    _lognormconsts -= shift
    # get rid of the dimension we normalized over
    lognormconsts = _lognormconsts.sum(axis=axis)

    return lognormconsts, lognormarr


def log_clip(arr):
    """Clip an array of log values at MIN_LOG (lower bound) and
    MAX_LOG (upper bound).

    Parameters
    ----------
    arr : numpy.ndarray
        Array of log values

    Returns
    -------
    out : numpy.ndarray
        Input array, with values clipped at MIN_LOG and MAX_LOG

    """

    carr = np.clip(arr, MIN_LOG, MAX_LOG)
    return carr


def safe_multiply(*arrs):
    """Multiply several arrays, but perform the multiplication safely
    to avoid overflow and underflow. Values that are too small are
    clipped to MIN_LOG, and values that are too large are clipped to
    MAX_LOG.

    Parameters
    ----------
    *arrs : list of numpy.ndarray
        Input arrays to be multiplied

    Returns
    -------
    out : numpy.ndarray
        The product of the input arrays.

    """

    sign = np.prod([np.sign(arr) for arr in arrs], axis=0)
    log_arr = np.sum([safe_log(np.abs(arr)) for arr in arrs], axis=0)
    clipped = exp(log_clip(log_arr))
    return clipped * sign


def safe_log(arr):
    """Take the log of an array, explicitly setting zero values tp
    -infinity.

    Parameters
    ----------
    arr : numpy.ndarray
        Input array

    Returns
    -------
    out : numpy.ndarray
        The log of the input array

    """

    log_arr = np.empty(arr.shape)
    mask = arr != 0
    log_arr[~mask] = -np.inf
    log_arr[mask] = log(arr[mask])
    return log_arr
