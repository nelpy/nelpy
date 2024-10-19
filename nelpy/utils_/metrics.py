"""
:mod:`metrics` --- metrics and measures
=============================================================
"""

import numpy as np

__all__ = [
    "gini",  # Gini coefficient of inequality
]


def gini(arr, mode="all"):
    """Calculate the Gini coefficient(s) of a matrix or vector.

    Parameters
    ----------
    arr : array-like
        Array or matrix on which to compute the Gini coefficient(s).
    mode : string, optional
        One of ['row-wise', 'col-wise', 'all']. Default is 'all'.

    Returns
    -------
    coeffs : array-like
        Array of Gini coefficients.

    Note
    ----
    If arr is a transition matrix A, such that Aij = P(S_k=j|S_{k-1}=i),
    then 'row-wise' is equivalent to 'tmat_departure' and 'col-wise' is
    equivalent to 'tmat_arrival'.

    Similarly, if arr is the observation (lambda) matrix of an HMM such that
    lambda \in \mathcal{C}^{n_states \times n_units}, then 'row-wise' is
    equivalent to 'lambda_across_units' and 'col-wise' is equivalent to
    'lambda_across_units'.

    If mode = 'all', then the matrix is unwrapped into a numel-dimensional
    array before computing the Gini coefficient.

    """
    if mode is None:
        mode = "row-wise"

    if mode not in ["row-wise", "col-wise", "all"]:
        raise ValueError("mode '{}' not supported!".format(mode))

    gini_coeffs = None

    if mode == "all":
        arr = np.atleast_1d(arr).astype(float)
        gini_coeffs = _gini(arr)

    elif mode == "row-wise":
        arr = np.atleast_2d(arr).astype(float)
        gini_coeffs = []
        for row in arr:
            gini_coeffs.append(_gini(row))

    elif mode == "col-wise":
        arr = np.atleast_2d(arr).astype(float)
        gini_coeffs = []
        for row in arr.T:
            gini_coeffs.append(_gini(row))

    return gini_coeffs


def _gini(arr):
    """Calculate the Gini coefficient of inequality of a numpy array.

    Parameters
    ----------
    arr : array-like

    Returns
    -------
    gini : float
        Gini coefficient of inequality.

    Additional information
    ----------------------

    This function was written by Olivia Guest (https://github.com/oliviaguest/gini).

    See https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm

    The classical definition of G appears in the notation of the theory of
    relative mean difference:

    .. math::
        G = \dfrac{\sum_{i=1}^n \sum_{j=1}^n |x_i - x_j|}{2n^2\bar{x}}
    where x is an observed value, n is the number of values observed and
    x bar is the mean value.

    If the x values are first placed in ascending order, such that each x
    has rank i, the some of the comparisons above can be avoided and
    computation is quicker:

    .. math::
        G = \dfrac{\sum_{i=1}^n(2i-n-1)x_i}{n \sum_{i=1}^n x_i}

    where x is an observed value, n is the number of values observed and i
    is the rank of values in ascending order.

    Note that only positive non-zero values are used.

    Examples
    --------
    For a very unequal sample, 999 zeros and a single one,

    .. code-block:: python
        a = np.zeros((1000))
        a[0] = 1.0
    the Gini coefficient is very close to 1.0:
    .. code-block:: python
        gini(a)
        0.99890010998900103

    For uniformly distributed random numbers, it will be low, around 0.33:

    .. code-block:: python
        s = np.random.uniform(-1,0,1000)
        gini(s)
    0.3295183767105907

    For a homogeneous sample, the Gini coefficient is 0.0:
    .. code-block:: python
        b = np.ones((1000))
        gini(b)
    0.0
    """
    # All values are treated equally, arrays must be 1d:
    arr = arr.flatten()
    if np.amin(arr) < 0:
        # Values cannot be negative:
        arr -= np.amin(arr)
    # Values cannot be 0:
    arr += 0.0000001
    # Values must be sorted:
    arr = np.sort(arr)
    # Index per array element:
    index = np.arange(1, arr.shape[0] + 1)
    # Number of array elements:
    n = arr.shape[0]
    # Gini coefficient:
    return (np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr))
