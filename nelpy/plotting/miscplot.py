"""Miscellaneous support plots for nelpy

'palplot' Copyright (c) 2012-2016, Michael L. Waskom
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .. import core
from . import utils  # import plotting/utils

__all__ = ["palplot", "stripplot", "veva_scatter"]


def palplot(pal, size=1):
    """
    Plot the values in a color palette as a horizontal array.

    Parameters
    ----------
    pal : sequence of matplotlib colors
        Colors, i.e. as returned by nelpy.color_palette().
    size : float, optional
        Scaling factor for size of plot. Default is 1.

    Examples
    --------
    >>> from nelpy.plotting.miscplot import palplot
    >>> pal = ["#FF0000", "#00FF00", "#0000FF"]
    >>> palplot(pal)
    """
    n = len(pal)
    f, ax = plt.subplots(1, 1, figsize=(n * size, size))
    ax.imshow(
        np.arange(n).reshape(1, n),
        cmap=mpl.colors.ListedColormap(list(pal)),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks(np.arange(n) - 0.5)
    ax.set_yticks([-0.5, 0.5])
    ax.set_xticklabels([])
    ax.set_yticklabels([])


def stripplot(*eps, voffset=None, lw=None, labels=None):
    """
    Plot epochs as segments on a line.

    Parameters
    ----------
    *eps : nelpy.EpochArray
        One or more EpochArray objects to plot.
    voffset : float, optional
        Vertical offset between lines.
    lw : float, optional
        Line width.
    labels : array-like of str, optional
        Labels for each EpochArray.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the strip plot.

    Examples
    --------
    >>> from nelpy import EpochArray
    >>> ep1 = EpochArray([[0, 1], [2, 3]])
    >>> ep2 = EpochArray([[4, 5], [6, 7]])
    >>> stripplot(ep1, ep2, labels=["A", "B"])
    """

    # TODO: this plot is in alpha mode; i.e., needs lots of work...
    # TODO: list unpacking if eps is a list of EpochArrays...

    fig = plt.figure(figsize=(10, 2))
    ax0 = fig.add_subplot(111)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    epmin = np.inf
    epmax = -np.inf

    for ii, epa in enumerate(eps):
        epmin = np.min((epa.start, epmin))
        epmax = np.max((epa.stop, epmax))

    # WARNING TODO: this does not yet wrap the color cycler, but it's easy to do with mod arith
    y = 0.2
    for ii, epa in enumerate(eps):
        ax0.hlines(y, epmin, epmax, "0.7")
        for ep in epa:
            ax0.plot(
                [ep.start, ep.stop],
                [y, y],
                lw=6,
                color=colors[ii],
                solid_capstyle="round",
            )
        y += 0.2

    utils.clear_top(ax0)
    #     npl.utils.clear_bottom(ax0)

    if labels is None:
        # try to get labels from epoch arrays
        labels = [""]
        labels.extend([epa.label for epa in eps])
    else:
        labels.insert(0, "")

    ax0.set_yticklabels(labels)

    ax0.set_xlim(epmin - 10, epmax + 10)
    ax0.set_ylim(0, 0.2 * (ii + 2))

    utils.no_yticks(ax0)
    utils.clear_left(ax0)
    utils.clear_right(ax0)

    return ax0


def veva_scatter(data, *, cmap=None, color=None, ax=None, lw=None, lh=None, **kwargs):
    """
    Scatter plot for ValueEventArray objects, colored by value.

    Parameters
    ----------
    data : nelpy.ValueEventArray
        The value event data to plot.
    cmap : matplotlib colormap, optional
        Colormap to use for the event values.
    color : matplotlib color, optional
        Color for the events if cmap is not specified.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis.
    lw : float, optional
        Line width for the event markers.
    lh : float, optional
        Line height for the event markers.
    **kwargs : dict
        Additional keyword arguments passed to vlines.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the scatter plot.

    Examples
    --------
    >>> from nelpy.core import ValueEventArray
    >>> vea = ValueEventArray(
    ...     [[1, 2, 3], [4, 5, 6]], values=[[10, 20, 30], [40, 50, 60]]
    ... )
    >>> veva_scatter(vea)
    """
    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()
    if cmap is None and color is None:
        color = "0.25"
    if lw is None:
        lw = 1.5
    if lh is None:
        lh = 0.95

    hh = lh / 2.0  # half the line height

    # Handle different types of input data
    if isinstance(data, core.ValueEventArray):
        vmin = (
            np.min([np.min(x) for x in data.values]) - 1
        )  # TODO: -1 because white is invisible... fix this properly
        vmax = np.max([np.max(x) for x in data.values])
        norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

        for ii, (events, values) in enumerate(zip(data.events, data.values)):
            if cmap is not None:
                colors = cmap(norm(values))
            else:
                colors = color
            ax.vlines(events, ii - hh, ii + hh, colors=colors, lw=lw, **kwargs)

    else:
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data)))
        )
    return ax
