import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from .helpers import RasterLabelData
from ..objects import *

__all__ = ['plot',
           'raster',
           'epoch_plot']

def plot(epocharray, data, *, ax=None, lw=None, mew=None, color=None,
         mec=None, **kwargs):
    """Plot an array-like object on an EpochArray.

    Parameters
    ----------
    epocharray : nelpy.EpochArray
        EpochArray on which the data is defined.
    data : array-like
        Data to plot on y axis; must be of size (epocharray.n_epochs,).
    ax : axis object, optional
        Plot in given axis; if None creates a new figure
    lw : float, optional
        Linewidth, default value of lw=1.5.
    mew : float, optional
        Marker edge width, default is equal to lw.
    color : matplotlib color, optional
        Plot color; default is '0.5' (gray).
    mec : matplotlib color, optional
        Marker edge color, default is equal to color.
    kwargs :
        Other keyword arguments are passed to main plot() call

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot data.

    Examples
    --------
    Plot a simple 5-element list on an EpochArray:

        >>> ep = EpochArray([[3, 4], [5, 8], [10, 12], [16, 20], [22, 23]])
        >>> data = [3, 4, 2, 5, 2]
        >>> npl.plot(ep, data)

    Hide the markers and change the linewidth:

        >>> ep = EpochArray([[3, 4], [5, 8], [10, 12], [16, 20], [22, 23]])
        >>> data = [3, 4, 2, 5, 2]
        >>> npl.plot(ep, data, ms=0, lw=3)
    """

    if ax is None:
        ax = plt.gca()
    if lw is None:
        lw = 1.5
    if mew is None:
        mew = lw
    if color is None:
        color = '0.3'
    if mec is None:
        mec = color

    if epocharray.n_epochs != len(data):
        raise ValueError("epocharray and data musthave the same length")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epoch, val in zip(epocharray, data):
            ax.plot(
                [epoch.start, epoch.stop],
                [val, val],
                '-o',
                color=color,
                mec=mec,
                markerfacecolor='w',
                lw=lw,
                mew=mew,
                **kwargs)

    ax.set_ylim([np.array(data).min()-0.5, np.array(data).max()+0.5])

    return ax

def imshow(data, *, ax=None, interpolation=None, **kwargs):
    """Docstring goes here."""

    # set default interpolation mode to 'none'
    if interpolation is None:
        interpolation = 'none'

def matshow(data, *, ax=None, **kwargs):
    """Docstring goes here."""

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    # Handle different types of input data
    if isinstance(data, BinnedSpikeTrainArray):
        # TODO: split by epoch, and plot matshows in same row, but with
        # a small gap to indicate discontinuities. How about slicing
        # then? Or slicing within an epoch?
        ax.matshow(data.data, **kwargs)
        ax.set_xlabel('time')
        ax.set_ylabel('unit')
        warnings.warn("Automatic x-axis formatting not yet implemented")
    else:
        raise NotImplementedError(
            "matshow({}) not yet supported".format(str(type(data))))

    return ax

def comboplot(*, ax=None, raster=None, analog=None, events=None):
    """Combo plot (consider better name) showing spike / state raster
    with additional analog signals, such as LFP or velocity, and also
    possibly with events. Here, the benefit is to have the figure and
    axes created automatically, in addition to prettification, as well
    as axis-linking. I don't know if we will really call this plot often
    though, so may be more of a gimmick?
    """

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    raise NotImplementedError("comboplot() not implemented yet")

    return ax

def occupancy():
    """Docstring goes here. TODO: complete me."""
    raise NotImplementedError("occupancy() not implemented yet")

def overviewstrip():
    """Plot an epoch array similar to vs scrollbar, to show gaps in e.g.
    matshow plots. TODO: complete me.
    """
    raise NotImplementedError("overviewstripplot() not implemented yet")

def rasterc(spiketrain, nbins=25, **kwargs):
    fig = plt.figure(figsize=(12, 4))
    gs = gridspec.GridSpec(2, 1, hspace=0.01, height_ratios=[0.2,0.8])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    ds = (spiketrain.support.stop - spiketrain.support.start)/nbins
    steps = np.squeeze(spiketrain.bin(ds=ds).flatten().data)
    stepsx = np.linspace(spiketrain.support.start, spiketrain.support.stop, num=nbins)
#     ax1.plot(stepsx, steps, drawstyle='steps-mid', color='none');
    ax1.set_ylim([-0.5, np.max(steps)+1])
    npl.raster(spiketrain, ax=ax2, **kwargs)

    npl.utils.clear_left_right(ax=ax1)
    npl.utils.clear_top_bottom(ax=ax1)
    npl.utils.clear_top(ax=ax2)

    ax1.fill_between(stepsx, steps, step='mid', color='0.4')

    npl.utils.sync_xlims(ax1, ax2)

    return ax1, ax2

def raster(data, *, cmap=None, color=None, ax=None, lw=None, lh=None,
           vertstack=None, labels=None, **kwargs):
    """Make a raster plot from a SpikeTrainArray object.

    Parameters
    ----------
    data : nelpy.SpikeTrainArray object
    cmap : matplotlib colormap, optional
    color: matplotlib color, optional
        Plot color; default is '0.25'
    ax : axis object, optional
        Plot in given axis. If None, plots on current axes
    lw : float, optional
        Linewidth, default value of lw=1.5.
    lh : float, optional
        Line height, default value of 0.95
    vertstack : Stack units in vertically adjacent positions, optional
        Default is to plot units in absolute positions according
        to their unit_ids
    labels : Labels for input data units, optional
        If not specified, default is to use the unit_labels from the
        SpikeTrainArray input. See SpikeTrainArray docstring for
        default behavior of unit_labels
    kwargs :
        Other keyword arguments are passed to main vlines() call

    Returns
    -------
    ax : matplotlib axis
        Axis object with plot data.

    Examples
    --------
    Instantiate a SpikeTrainArray and create a raster plot
        >>> stdata1 = [1,2,4,5,6,10,20]
        >>> stdata2 = [3,4,4.5,5,5.5,19]
        >>> stdata3 = [5,12,14,15,16,18,22,23,24]
        >>> stdata4 = [5,12,14,15,16,18,23,25,32]

        >>> sta1 = nelpy.SpikeTrainArray([stdata1, stdata2, stdata3,
                                          stdata4, stdata1+stdata4],
                                          fs=5, unit_ids=[1,2,3,4,6])
        >>> ax = raster(sta1, color='cyan', lw=2, lh=2)

    Instantiate another SpikeTrain Array, stack units, and specify labels.
    Note that the user-specified labels in the call to raster() will be
    shown instead of the unit_labels associated with the input data
        >>> sta3 = nelpy.SpikeTrainArray([stdata1, stdata4, stdata2+stdata3],
                                         support=ep1, fs=5, unit_ids=[10,5,12],
                                         unit_labels=['some', 'more', 'cells'])
        >>> raster(sta3, color=plt.cm.Blues, lw=2, lh=2, vertstack=True,
                   labels=['units', 'of', 'interest'])
    """

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()
    if cmap is None and color is None:
        color = '0.25'
    if lw is None:
        lw = 1.5
    if lh is None:
        lh = 0.95
    if vertstack is None:
        vertstack = False

    firstplot = False
    if not ax.findobj(match=RasterLabelData):
        firstplot = True
        ax.add_artist(RasterLabelData())

    # override labels
    if labels is not None:
        unit_labels = labels
    else:
        unit_labels = []

    hh = lh/2.0  # half the line height

    # Handle different types of input data
    if isinstance(data, SpikeTrainArray):

        label_data = ax.findobj(match=RasterLabelData)[0].label_data
        unitlist = [np.NINF for element in data.unit_ids]
        # no override labels so use unit_labels from input
        if not unit_labels:
            unit_labels = data.unit_labels

        if firstplot:
            if vertstack:
                minunit = 1
                maxunit = data.n_units
                unitlist = range(1, data.n_units + 1)
            else:
                minunit = np.array(data.unit_ids).min()
                maxunit = np.array(data.unit_ids).max()
                unitlist = data.unit_ids
        # see if any of the unit_ids has already been plotted. If so,
        # then merge
        else:
            for idx, unit_id in enumerate(data.unit_ids):
                if unit_id in label_data.keys():
                    position, _ = label_data[unit_id]
                    unitlist[idx] = position
                else:  # unit not yet plotted
                    if vertstack:
                        unitlist[idx] = 1 + max(int(ax.get_yticks()[-1]),
                                                max(unitlist))
                    else:
                        warnings.warn("Spike trains may be plotted in "
                                      "the same vertical position as "
                                      "another unit")
                        unitlist[idx] = data.unit_ids[idx]

        if firstplot:
            minunit = int(minunit)
            maxunit = int(maxunit)
        else:
            (prev_ymin, prev_ymax) = ax.findobj(match=RasterLabelData)[0].yrange
            minunit = int(np.min([np.ceil(prev_ymin), np.min(unitlist)]))
            maxunit = int(np.max([np.floor(prev_ymax), np.max(unitlist)]))

        yrange = (minunit - 0.5, maxunit + 0.5)

        if cmap is not None:
            color_range = range(data.n_units)
            # TODO: if we go from 0 then most colormaps are invisible at one end of the spectrum
            colors = cmap(np.linspace(0.25, 0.75, data.n_units))
            for unit, spiketrain, color_idx in zip(unitlist, data.time, color_range):
                ax.vlines(spiketrain, unit - hh, unit + hh, colors=colors[color_idx], lw=lw, **kwargs)
        else:  # use a constant color:
            for unit, spiketrain in zip(unitlist, data.time):
                ax.vlines(spiketrain, unit - hh, unit + hh, colors=color, lw=lw, **kwargs)

        # get existing label data so we can set some attributes
        rld = ax.findobj(match=RasterLabelData)[0]

        ax.set_ylim(yrange)
        rld.yrange = yrange

        for unit_id, loc, label in zip(data.unit_ids, unitlist, unit_labels):
            rld.label_data[unit_id] = (loc, label)
        unitlocs = []
        unitlabels = []
        for loc, label in label_data.values():
            unitlocs.append(loc)
            unitlabels.append(label)
        ax.set_yticks(unitlocs)
        ax.set_yticklabels(unitlabels)

    else:
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data))))
    return ax

def epoch_plot(ax, epochs, height=None, fc='0.5', ec=None,
                      alpha=0.5, hatch='////'):
    """Docstring goes here.
    """

    import matplotlib.patches as patches
    for start, stop in zip(epochs.starts, epochs.stops):
        ax.add_patch(
            patches.Rectangle(
                (start, 0),   # (x,y)
                stop - start ,          # width
                height,          # height
                hatch=hatch,
                facecolor=fc,
                edgecolor=ec,
                alpha=alpha
            )
        )
    # ax.set_xlim([epochs.start, epochs.stop])