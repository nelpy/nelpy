__all__ = ['decode_and_plot_events1D',
           'plot_cum_error_dist']

import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import itertools

from . import palettes
# colors = itertools.cycle(npl.palettes.color_palette(palette="sweet", n_colors=15))

from .. import decoding
from . import utils as plotutils
from ..utils import is_sorted, collapse_time
from .core import rasterplot, imagesc

def decode_and_plot_events1D(*, bst, tc, raster=True, st=None, st_order='track',
                             evt_subset=None, **kwargs):
    """
    bst : BinnedSpikeTrainArray
    tc : TuningCurve1D
    raster : bool
    st : SpikeTrainArray, optional
    st_order : string, optional
        = ['track', 'first', 'random']
    evt_subset : list, optional
        List of integer indices. If the list is not sorted, it will be sorted first.
    """

    #TODO: add **kwargs
    #   fig size, cmap, raster lw, raster color, other axes props, ...

    from mpl_toolkits.axes_grid1 import make_axes_locatable

    unit_ids = set(bst.unit_ids)
    unit_ids = unit_ids.intersection(st.unit_ids)
    unit_ids = unit_ids.intersection(tc.unit_ids)

    bst = bst._unit_subset(unit_ids)
    st = st._unit_subset(unit_ids)
    tc = tc._unit_subset(unit_ids)

    if evt_subset is None:
        evt_subset = np.arange(bst.n_epochs)
    evt_subset = list(evt_subset)
    if not is_sorted(evt_subset):
        evt_subset.sort()
    bst = bst[evt_subset]

    # now that the bst has potentially been restricted by evt_subset, we trim down the spike train as well:
    st = st[bst.support]
    st = collapse_time(st)

    if st_order == 'track':
        new_order = tc.get_peak_firing_order_ids()
    elif st_order == 'first':
        new_order = st.get_spike_firing_order()
    elif st_order == 'random':
        new_order = np.random.permutation(st.unit_ids)
    else:
        new_order = st_order
    st.reorder_units_by_ids(new_order, inplace=True)

    # now decode events in bst:
    posterior, bdries, mode_pth, mean_pth = decoding.decode1D(bst=bst, ratemap=tc, xmax=tc.bins[-1])

    fig, ax = plt.subplots(figsize=(bst.n_bins/5, 4))

    pixel_width = 0.5

    imagesc(x=np.arange(bst.n_bins), y=np.arange(311), data=posterior, cmap=plt.cm.Spectral_r, ax=ax)
    plotutils.yticks_interval(310)
    plotutils.no_yticks(ax)

    ax.vlines(np.arange(bst.lengths.sum())-pixel_width, *ax.get_ylim(), lw=1, linestyle=':', color='0.8')
    ax.vlines(np.cumsum(bst.lengths)-pixel_width, *ax.get_ylim(), lw=1)

    ax.set_xlim(-pixel_width, bst.lengths.sum()-pixel_width)

    event_centers = np.insert(np.cumsum(bst.lengths),0,0)
    event_centers = event_centers[:-1] + bst.lengths/2 - 0.5

#     ax.set_xticks([0, bst.n_bins-1])
#     ax.set_xticklabels([1, bst.n_bins])

    ax.set_xticks(event_centers)
    ax.set_xticklabels(evt_subset)
#     ax.xaxis.tick_top()
#     ax.xaxis.set_label_position('top')

    plotutils.no_xticks(ax)

    divider = make_axes_locatable(ax)
    axRaster = divider.append_axes("top", size=1.5, pad=0)

    rasterplot(st, vertstack=True, ax=axRaster, lh=1.25, lw=2.5, color='0.1')

    axRaster.set_xlim(st.support.time.squeeze())
    bin_edges = np.linspace(st.support.time[0,0],st.support.time[0,1], bst.n_bins+1)
    axRaster.vlines(bin_edges, *axRaster.get_ylim(), lw=1, linestyle=':', color='0.8')
    axRaster.vlines(bin_edges[np.cumsum(bst.lengths)], *axRaster.get_ylim(), lw=1, color='0.2')
    plotutils.no_xticks(axRaster)
    plotutils.no_xticklabels(axRaster)
    plotutils.no_yticklabels(axRaster)
    plotutils.no_yticks(axRaster)
    ax.set_ylabel('position')
    axRaster.set_ylabel('units')
    ax.set_xlabel('time bins')
    plotutils.clear_left_right(axRaster)
    plotutils.clear_top_bottom(axRaster)

    plotutils.align_ylabels(0, ax, axRaster)
    return fig


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