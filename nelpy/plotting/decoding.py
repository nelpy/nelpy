__all__ = ["decode_and_plot_events1D", "plot_cum_error_dist", "plot_posteriors"]

# import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

try:
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except ImportError:
    from mpl_toolkits.axes_grid.inset_locator import inset_axes


# colors = itertools.cycle(npl.palettes.color_palette(palette="sweet", n_colors=15))

from .. import decoding
from ..utils import collapse_time, is_sorted
from . import utils as plotutils
from .core import imagesc, rasterplot


def plot_posteriors(bst, tuningcurve, idx=None, w=1, bin_px_size=0.08):
    """
    Plot posterior probabilities for decoded neural activity.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        The binned spike train array to decode.
    tuningcurve : TuningCurve1D
        The tuning curve used for decoding.
    idx : array-like, optional
        Indices of events to plot. If None, all events are plotted.
    w : int, optional
        Window size for decoding (default is 1).
    bin_px_size : float, optional
        Size of each bin in pixels for the plot (default is 0.08).

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the posterior plot.

    Examples
    --------
    >>> ax = plot_posteriors(bst, tc)
    >>> plt.show()
    """
    if idx is not None:
        bst = bst[idx]
    tc = tuningcurve

    # decode neural activity
    posterior, bdries, mode_pth, mean_pth = decoding.decode1D(
        bst=bst, ratemap=tc, xmin=tc.bins[0], xmax=tc.bins[-1], w=w
    )

    pixel_width = 0.5

    n_ext, n_bins = posterior.shape
    lengths = np.diff(bdries)

    plt.figure(figsize=(bin_px_size * n_bins, 2))
    ax = plt.gca()

    imagesc(
        x=np.arange(n_bins),
        y=np.arange(int(tc.bins[-1] + 1)),
        data=posterior,
        cmap=plt.cm.Spectral_r,
        ax=ax,
    )
    plotutils.yticks_interval(tc.bins[-1])
    plotutils.no_yticks(ax)
    # plt.imshow(posterior, cmap=plt.cm.Spectral_r, interpolation='none', aspect='auto')
    ax.vlines(
        np.arange(lengths.sum()) - pixel_width,
        *ax.get_ylim(),
        lw=1,
        linestyle=":",
        color="0.8",
    )
    ax.vlines(np.cumsum(lengths) - pixel_width, *ax.get_ylim(), lw=1)

    ax.set_xlim(-pixel_width, lengths.sum() - pixel_width)

    event_centers = np.insert(np.cumsum(lengths), 0, 0)
    event_centers = event_centers[:-1] + lengths / 2 - 0.5

    ax.set_xticks(event_centers)
    if idx is not None:
        ax.set_xticklabels(idx)
    else:
        ax.set_xticklabels(np.arange(bst.n_intervals))

    plotutils.no_xticks(ax)

    return ax


def decode_and_plot_events1D(
    *, bst, tc, raster=True, st=None, st_order="track", evt_subset=None, **kwargs
):
    """
    Decode and plot 1D events with optional raster plot overlay.

    Parameters
    ----------
    bst : BinnedSpikeTrainArray
        The binned spike train array to decode.
    tc : TuningCurve1D
        The tuning curve used for decoding.
    raster : bool, optional
        Whether to include a raster plot (default is True).
    st : SpikeTrainArray, optional
        The spike train array for raster plotting.
    st_order : str or array-like, optional
        Order of units for raster plot. Options: 'track', 'first', 'random', or array of unit ids.
    evt_subset : list, optional
        List of integer indices for event selection. If not sorted, will be sorted.
    **kwargs
        Additional keyword arguments for plotting.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure containing the plot.

    Examples
    --------
    >>> fig = decode_and_plot_events1D(bst=bst, tc=tc, st=st)
    >>> plt.show()
    """

    # TODO: add **kwargs
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

    if st_order == "track":
        new_order = tc.get_peak_firing_order_ids()
    elif st_order == "first":
        new_order = st.get_spike_firing_order()
    elif st_order == "random":
        new_order = np.random.permutation(st.unit_ids)
    else:
        new_order = st_order
    st.reorder_units_by_ids(new_order, inplace=True)

    # now decode events in bst:
    posterior, bdries, mode_pth, mean_pth = decoding.decode1D(
        bst=bst, ratemap=tc, xmax=tc.bins[-1]
    )

    fig, ax = plt.subplots(figsize=(bst.n_bins / 5, 4))

    pixel_width = 0.5

    imagesc(
        x=np.arange(bst.n_bins),
        y=np.arange(311),
        data=posterior,
        cmap=plt.cm.Spectral_r,
        ax=ax,
    )
    plotutils.yticks_interval(310)
    plotutils.no_yticks(ax)

    ax.vlines(
        np.arange(bst.lengths.sum()) - pixel_width,
        *ax.get_ylim(),
        lw=1,
        linestyle=":",
        color="0.8",
    )
    ax.vlines(np.cumsum(bst.lengths) - pixel_width, *ax.get_ylim(), lw=1)

    ax.set_xlim(-pixel_width, bst.lengths.sum() - pixel_width)

    event_centers = np.insert(np.cumsum(bst.lengths), 0, 0)
    event_centers = event_centers[:-1] + bst.lengths / 2 - 0.5

    #     ax.set_xticks([0, bst.n_bins-1])
    #     ax.set_xticklabels([1, bst.n_bins])

    ax.set_xticks(event_centers)
    ax.set_xticklabels(evt_subset)
    #     ax.xaxis.tick_top()
    #     ax.xaxis.set_label_position('top')

    plotutils.no_xticks(ax)

    divider = make_axes_locatable(ax)
    axRaster = divider.append_axes("top", size=1.5, pad=0)

    rasterplot(st, vertstack=True, ax=axRaster, lh=1.25, lw=2.5, color="0.1")

    axRaster.set_xlim(st.support.time.squeeze())
    bin_edges = np.linspace(
        st.support.time[0, 0], st.support.time[0, 1], bst.n_bins + 1
    )
    axRaster.vlines(bin_edges, *axRaster.get_ylim(), lw=1, linestyle=":", color="0.8")
    axRaster.vlines(
        bin_edges[np.cumsum(bst.lengths)], *axRaster.get_ylim(), lw=1, color="0.2"
    )
    plotutils.no_xticks(axRaster)
    plotutils.no_xticklabels(axRaster)
    plotutils.no_yticklabels(axRaster)
    plotutils.no_yticks(axRaster)
    ax.set_ylabel("position")
    axRaster.set_ylabel("units")
    ax.set_xlabel("time bins")
    plotutils.clear_left_right(axRaster)
    plotutils.clear_top_bottom(axRaster)

    plotutils.align_ylabels(0, ax, axRaster)
    return fig


def plot_cum_error_dist(
    *,
    cumhist=None,
    bincenters=None,
    bst=None,
    extern=None,
    decodefunc=None,
    k=None,
    transfunc=None,
    n_extern=None,
    n_bins=None,
    extmin=None,
    extmax=None,
    sigma=None,
    lw=None,
    ax=None,
    inset=True,
    inset_ax=None,
    color=None,
    **kwargs,
):
    """
    Plot (and optionally compute) the cumulative distribution of decoding errors.

    Evaluated using a cross-validation procedure. See Fig 3.(b) of "Analysis of Hippocampal Memory Replay Using Neural Population Decoding", Fabian Kloosterman, 2012.

    Parameters
    ----------
    cumhist : array-like, optional
        Precomputed cumulative histogram of errors. If None, will be computed.
    bincenters : array-like, optional
        Bin centers for the cumulative histogram. If None, will be computed.
    bst : BinnedSpikeTrainArray, optional
        Required if cumhist and bincenters are not provided. Used for error computation.
    extern : array-like, optional
        External variable (e.g., position) for decoding. Required if cumhist and bincenters are not provided.
    decodefunc : callable, optional
        Decoding function to use. Defaults to decoding.decode1D.
    k : int, optional
        Number of cross-validation folds. Default is 5.
    transfunc : callable, optional
        Optional transformation function for the external variable.
    n_extern : int, optional
        Number of external variable samples. Default is 100.
    n_bins : int, optional
        Number of bins for the error histogram. Default is 200.
    extmin : float, optional
        Minimum value of the external variable. Default is 0.
    extmax : float, optional
        Maximum value of the external variable. Default is 100.
    sigma : float, optional
        Smoothing parameter. Default is 3.
    lw : float, optional
        Line width for the plot. Default is 1.5.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, uses current axis.
    inset : bool, optional
        Whether to include an inset plot. Default is True.
    inset_ax : matplotlib.axes.Axes, optional
        Axis for the inset plot. If None, one will be created.
    color : color, optional
        Line color. If None, uses next color in cycle.
    **kwargs
        Additional keyword arguments for plotting.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis with the cumulative error plot.
    inset_ax : matplotlib.axes.Axes, optional
        The axis with the inset plot (if inset=True).

    Examples
    --------
    >>> ax, inset_ax = plot_cum_error_dist(bst=bst, extern=pos)
    >>> plt.show()
    """

    if ax is None:
        ax = plt.gca()
    if lw is None:
        lw = 1.5
    if decodefunc is None:
        decodefunc = decoding.decode1D
    if k is None:
        k = 5
    if n_extern is None:
        n_extern = 100
    if n_bins is None:
        n_bins = 200
    if extmin is None:
        extmin = 0
    if extmax is None:
        extmax = 100
    if sigma is None:
        sigma = 3

    # Get the color from the current color cycle
    if color is None:
        (line,) = ax.plot(0, 0.5)
        color = line.get_color()
        line.remove()

    # if cumhist or bincenters are NOT provided, then compute them
    if cumhist is None or bincenters is None:
        assert bst is not None, (
            "if cumhist and bincenters are not given, then bst must be provided to recompute them!"
        )
        assert extern is not None, (
            "if cumhist and bincenters are not given, then extern must be provided to recompute them!"
        )
        cumhist, bincenters = decoding.cumulative_dist_decoding_error_using_xval(
            bst=bst,
            extern=extern,
            decodefunc=decoding.decode1D,
            k=k,
            transfunc=transfunc,
            n_extern=n_extern,
            extmin=extmin,
            extmax=extmax,
            sigma=sigma,
            n_bins=n_bins,
        )
    # now plot results
    ax.plot(bincenters, cumhist, lw=lw, color=color, **kwargs)
    ax.set_xlim(bincenters[0], bincenters[-1])
    ax.set_xlabel("error [cm]")
    ax.set_ylabel("cumulative probability")

    ax.set_ylim(0)

    if inset:
        if inset_ax is None:
            inset_ax = inset_axes(
                parent_axes=ax, width="60%", height="50%", loc=4, borderpad=2
            )

        inset_ax.plot(bincenters, cumhist, lw=lw, color=color, **kwargs)

        # annotate inset
        thresh1 = 0.7
        inset_ax.hlines(
            thresh1, 0, cumhist(thresh1), color=color, alpha=0.9, lw=lw, linestyle="--"
        )
        inset_ax.vlines(
            cumhist(thresh1), 0, thresh1, color=color, alpha=0.9, lw=lw, linestyle="--"
        )
        inset_ax.set_xlim(0, 12 * np.ceil(cumhist(thresh1) / 10))

        thresh2 = 0.5
        inset_ax.hlines(
            thresh2, 0, cumhist(thresh2), color=color, alpha=0.6, lw=lw, linestyle="--"
        )
        inset_ax.vlines(
            cumhist(thresh2), 0, thresh2, color=color, alpha=0.6, lw=lw, linestyle="--"
        )

        inset_ax.set_yticks((0, thresh1, thresh2, 1))
        inset_ax.set_ylim(0)

        return ax, inset_ax

    return ax
