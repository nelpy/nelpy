"""stats.py -- helper functions for working with probabilities/statistics

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
import circstats as circ


def xcorr(x, y, circular=False, deg=False, nanrobust=False):
    """Returns matrix of correlations between x and y

    Parameters
    ----------
    x : np.ndarray
        Columns are different parameters, rows are sets of observations
    y : np.ndarray
        Columns are different parameters, rows are sets of observations
    circular : bool (default=False)
        Whether or not the data is circular
    deg : bool (default=False)
        Whether or not the data is in degrees (if circular)

    Returns
    -------
    out : np.ndarray
        Matrix of correlations between rows of x and y
    """

    # Inputs' shapes
    xshape = x.shape
    yshape = y.shape

    # Store original (output) shapes
    corrshape = xshape[:-1] + yshape[:-1]

    # Prepares inputs' shapes for computations
    if len(x.shape) > 2:
        x = x.reshape((np.prod(xshape[:-1]), xshape[-1]), order='C')
    if len(y.shape) > 2:
        y = y.reshape((np.prod(yshape[:-1]), yshape[-1]), order='C')

    if x.ndim == 1:
        x = x[None, :]
    if y.ndim == 1:
        y = y[None, :]

    if circular:
        if deg:
            x = np.radians(x)
            y = np.radians(y)

        # if nanrobust:
        #     corr = circ.nancorrcc(x, y, axis=1)
        # else:
        #     corr = circ.corrcc(x, y, axis=1, nancheck=False)
        if nanrobust:
            corr = circ.nancorrcc(x[:, :, None], y.T[None, :, :], axis=1)
        else:
            corr = circ.corrcc(x[:, :, None], y.T[None, :, :], axis=1)

    else:
        avgfn = np.mean
        stdfn = np.std

        # numerator factors (centered means)
        nx = (x.T - avgfn(x, axis=1)).T
        ny = (y.T - avgfn(y, axis=1)).T

        # denominator factors (std devs)
        sx = stdfn(x, axis=1)
        sy = stdfn(y, axis=1)

        # numerator
        num = np.dot(nx, ny.T) / x.shape[1]

        # correlation
        corr = num / np.outer(sx, sy)

    # reshape to take original
    corr = corr.reshape(corrshape, order='F')

    return corr
