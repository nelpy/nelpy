from itertools import tee
from scipy import signal
from matplotlib.offsetbox import AnchoredOffsetbox

import numpy as np

def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def is_sorted(iterable, key=lambda a, b: a <= b):
    """Check to see if iterable is monotonic increasing (sorted)."""
    return all(key(a, b) for a, b in pairwise(iterable))

def find_nearest_idx(array, val):
    """Finds nearest index in array to value.

    Parameters
    ----------
    array : np.array
    val : float

    Returns
    -------
    Index into array that is closest to val

    """
    return (np.abs(array-val)).argmin()


def find_nearest_indices(array, vals):
    """Finds nearest index in array to value.

    Parameters
    ----------
    array : np.array
        This is the array you wish to index into.
    vals : np.array
        This is the array that you are getting your indices from.

    Returns
    -------
    Indices into array that is closest to vals.

    Notes
    -----
    Wrapper around find_nearest_idx().

    """
    return np.array([find_nearest_idx(array, val) for val in vals], dtype=int)


def get_sort_idx(tuning_curves):
    """Finds indices to sort neurons by max firing in tuning curve.

    Parameters
    ----------
    tuning_curves : list of lists
        Where each inner list is the tuning curves for an individual
        neuron.

    Returns
    -------
    sorted_idx : list
        List of integers that correspond to the neuron in sorted order.

    """
    tc_max_loc = []
    for i, neuron_tc in enumerate(tuning_curves):
        tc_max_loc.append((i, np.where(neuron_tc == np.max(neuron_tc))[0][0]))
    sorted_by_tc = sorted(tc_max_loc, key=lambda x: x[1])

    sorted_idx = []
    for idx in sorted_by_tc:
        sorted_idx.append(idx[0])

    return sorted_idx


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None,
                 loc=4, pad=0.1, borderpad=0.1, sep=2, prop=None, fontsize='medium', **kwargs):
        """
        Modified, draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        Parameters
        ----------
        transform : the coordinate frame (typically axes.transData)
        sizex, sizey : width of x,y bar, in data units. 0 to omit
        labelx, labely : labels for x,y bars; None to omit
        loc : position in containing axes
        pad, borderpad : padding, in fraction of the legend font size (or prop)
        sep : separation between labels and bars in points.
        **kwargs : additional arguments passed to base class constructor

        Notes
        -----
        Adapted from mpl_toolkits.axes_grid2

        """
        from matplotlib.lines import Line2D
        from matplotlib.text import Text
        from matplotlib.offsetbox import AuxTransformBox
        bars = AuxTransformBox(transform)
        inv = transform.inverted()
        pixelxy = inv.transform((1, 1)) - inv.transform((0, 0))

        if sizex:
            barx = Line2D([sizex, 0], [0, 0], transform=transform, color='k')
            bars.add_artist(barx)

        if sizey:
            bary = Line2D([0, 0], [0, sizey], transform=transform, color='k')
            bars.add_artist(bary)

        if sizex and labelx:
            textx = Text(text=labelx, x=sizex/2.0, y=-5*pixelxy[1], ha='center', va='top', size=fontsize)
            bars.add_artist(textx)

        if sizey and labely:
            texty = Text(text=labely, rotation='vertical', y=sizey/2.0, x=-2*pixelxy[0],
                         va='center', ha='right', size=fontsize)
            bars.add_artist(texty)

        AnchoredOffsetbox.__init__(self, loc=loc, pad=pad, borderpad=borderpad,
                                       child=bars, prop=prop, frameon=False, **kwargs)

def add_scalebar(ax, matchx=True, matchy=True, hidex=True, hidey=True, fontsize='medium', units='ms', **kwargs):
    """Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the
    plot and optionally hiding the x and y axes

    Parameters
    ----------
    ax :
        The axis to attach ticks to
    matchx, matchy : boolean
        If True (default), set size of scale bars to spacing between
        ticks
        If False, size should be set using sizex and sizey params
    hidex, hidey : boolean
        If True, hide x-axis and y-axis of parent
    **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    from matplotlib.ticker import AutoLocator
    locator = AutoLocator()

    def find_loc(vmin, vmax):
        loc = locator.tick_values(vmin, vmax)
        return len(loc)>1 and (loc[1] - loc[0])

    if matchx:
        kwargs['sizex'] = find_loc(*ax.get_xlim())
#         kwargs['labelx'] = str(kwargs['sizex'])
        if units == 'ms':
            kwargs['labelx'] = str(int(round(kwargs['sizex']*1000, 2))) + ' ms'
        elif units == 's':
            kwargs['labelx'] = str(int(round(kwargs['sizex'], 2))) + ' s'

    if matchy:
        kwargs['sizey'] = find_loc(*ax.get_ylim())
        kwargs['labely'] = str(kwargs['sizey'])

    scalebar = AnchoredScaleBar(ax.transData, fontsize=fontsize, **kwargs)
    ax.add_artist(scalebar)

    return scalebar


def get_counts(spikes, edges, gaussian_std=None, n_gaussian_std=5):
    """Finds the number of spikes in each bin.

    Parameters
    ----------
    spikes : list
        Contains vdmlan.SpikeTrain for each neuron
    edges : np.array
        Bin edges for computing spike counts.
    gaussian_std : float
        Standard deviation for gaussian filter. Default is None.

    Returns
    -------
    counts : np.array
        Where each inner array is the number of spikes (int) in each bin for an individual neuron.

    """
    dt = np.median(np.diff(edges))

    if gaussian_std is not None:
        n_points = n_gaussian_std * gaussian_std * 2 / dt
        n_points = max(n_points, 1.0)
        if n_points % 2 == 0:
            n_points += 1
        if n_points > len(edges):
            raise ValueError("gaussian_std is too large for these times")
        gaussian_filter = signal.gaussian(n_points, gaussian_std/dt)
        gaussian_filter /= np.sum(gaussian_filter)

    counts = np.zeros((len(spikes), len(edges)-1))
    for idx, spiketrain in enumerate(spikes):
        counts[idx] = np.histogram(spiketrain.time, bins=edges)[0]
        if gaussian_std is not None and gaussian_std > dt:
            counts[idx] = np.convolve(counts[idx], gaussian_filter, mode='same')
    return counts


def cartesian(xcenters, ycenters):
    """Finds every combination of elements in two arrays.

    Parameters
    ----------
    xcenters : np.array
    ycenters : np.array

    Returns
    -------
    cartesian : np.array
        With shape(n_sample, 2).

    """
    return np.transpose([np.tile(xcenters, len(ycenters)), np.repeat(ycenters, len(xcenters))])


def epoch_position(position, epoch):
    """Finds positions associated with epoch times

    Parameters
    ----------
    position : vdmlab.Position
    epoch : vdmlab.Epoch

    Returns
    -------
    epoch_position : vdmlab.Position

    """
    return position.time_slices(epoch.starts, epoch.stops)
