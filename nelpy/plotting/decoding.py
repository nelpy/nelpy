__all__ = ['plot_cum_error_dist']

import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import itertools
from . import palettes
# colors = itertools.cycle(npl.palettes.color_palette(palette="sweet", n_colors=15))

# from ..core import *
# from ..auxiliary import *
from .. import decoding
# from . import utils  # import plotting/utils

def plot_cum_error_dist(*, cumhist=None, bincenters=None,
                        bst=None, extern=None, decodefunc=None,
                        k=None, transfunc=None, n_extern=None,
                        n_bins = None, extmin=None, extmax=None,
                        sigma=None, lw=None, ax=None, inset=True,
                        inset_ax=None, color=None, **kwargs):
    """Plot (and optionally compute) the cumulative distribution of
    decoding errors, evaluated using a cross-validation procedure.

    See Fig 3.(b) of "Analysis of Hippocampal Memory Replay Using Neural
        Population Decoding", Fabian Kloosterman, 2012.

    Parameters
    ----------


    Returns
    -------
    """

    if ax is None:
        ax = plt.gca()
    if lw is None:
        lw=1.5
    if decodefunc is None:
        decodefunc = decoding.decode1D
    if k is None:
        k=5
    if n_extern is None:
        n_extern=100
    if n_bins is None:
        n_bins = 200
    if extmin is None:
        extmin=0
    if extmax is None:
        extmax=100
    if sigma is None:
        sigma = 3

    # Get the color from the current color cycle
    if color is None:
        line, = ax.plot(0, 0.5)
        color = line.get_color()
        line.remove()

    # if cumhist or bincenters are NOT provided, then compute them
    if cumhist is None or bincenters is None:
        assert bst is not None, "if cumhist and bincenters are not given, then bst must be provided to recompute them!"
        assert extern is not None, "if cumhist and bincenters are not given, then extern must be provided to recompute them!"
        cumhist, bincenters = \
        decoding.cumulative_dist_decoding_error_using_xval(
            bst=bst,
            extern=extern,
            decodefunc=decoding.decode1D,
            k=k,
            transfunc=transfunc,
            n_extern=n_extern,
            extmin=extmin,
            extmax=extmax,
            sigma=sigma,
            n_bins=n_bins)
    # now plot results
    ax.plot(bincenters, cumhist, lw=lw, color=color, **kwargs)
    ax.set_xlim(bincenters[0], bincenters[-1])
    ax.set_xlabel('error [cm]')
    ax.set_ylabel('cumulative probability')

    ax.set_ylim(0)

    if inset:
        if inset_ax is None:
            inset_ax = inset_axes(parent_axes=ax,
                                  width="60%",
                                  height="50%",
                                  loc=4,
                                  borderpad=2)

        inset_ax.plot(bincenters, cumhist, lw=lw, color=color, **kwargs)

        # annotate inset
        thresh1 = 0.7
        bcidx = np.asscalar(np.argwhere(cumhist>thresh1)[0]-1)
        inset_ax.hlines(thresh1, 0, bincenters[bcidx], color=color, alpha=0.9, linestyle='--')
        inset_ax.vlines(bincenters[bcidx], 0, thresh1, color=color, alpha=0.9, linestyle='--')

        inset_ax.set_xlim(0,12*np.ceil(bincenters[bcidx]/10))

        thresh2 = 0.5
        bcidx = np.asscalar(np.argwhere(cumhist>thresh2)[0]-1)

        inset_ax.hlines(thresh2, 0, bincenters[bcidx], color=color, alpha=0.6, linestyle='--')
        inset_ax.vlines(bincenters[bcidx], 0, thresh2, color=color, alpha=0.6, linestyle='--')

        inset_ax.set_yticks((0,thresh1, thresh2, 1))
        inset_ax.set_ylim(0)

        return ax, inset_ax

    return ax