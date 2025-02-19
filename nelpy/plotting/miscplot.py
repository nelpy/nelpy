"""Miscellaneous support plots for nelpy

'palplot' Copyright (c) 2012-2016, Michael L. Waskom
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from . import utils  # import plotting/utils
from .. import core

__all__ = ["palplot", "stripplot", "veva_scatter"]


def palplot(pal, size=1):
    """Plot the values in a color palette as a horizontal array.
    Parameters
    ----------
    pal : sequence of matplotlib colors
        colors, i.e. as returned by nelpy.color_palette()
    size :
        scaling factor for size of plot
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
    """Plot epochs as segments on a line.
    Parameters
    ----------
    *eps : EpochArrays
    voffset : float
    lw : float
    labels : array-like of str
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
