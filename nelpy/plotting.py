#encoding : utf-8
"""This file contains the nelpy plotting functions and utilities.
 * 
"""

# TODO: see https://gist.github.com/arnaldorusso/6611ff6c05e1efc2fb72

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .objects import *

from matplotlib import cm
from matplotlib import colors as mplcolors

def matshow(data, *, ax=None, **kwargs):

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    # Handle different types of input data
    if isinstance(data, BinnedSpikeTrainArray):
        # TODO: split by epoch, and plot matshows in same row, but with a small gap to indicate discontinuities. How about slicing then? Or slicing within an epoch?
        ax.matshow(data.data, **kwargs)
        ax.set_xlabel('time')
        ax.set_ylabel('unit')
        warnings.warn("Automatic x-axis formatting not yet implemented")
    else:
        raise NotImplementedError(
            "matshow({}) not yet supported".format(str(type(data))))

    return ax

def comboplot(*, ax=None, raster=None, analog=None, events=None):
    """Combo plot (consider better name) showing spike / state raster with
    additional analog signals, such as LFP or velocity, and also possibly with 
    events. Here, the benefit is to have the figure and axes created automatically,
    in addition to prettification, as well as axis-linking. I don't know if we will 
    really call this plot often though, so may be more of a gimmick?
    """

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    raise NotImplementedError("comboplot() not implemented yet")

    return ax

def occupancy():
    pass

def overviewstrip():
    """Plot an epoch array similar to vs scrollbar, to show gaps in e.g.
    matshow plots.
    """
    pass


def plot(data, *, cmap=plt.cm.Accent, color=None, legend=True, ax=None, plot_support=True, **kwargs):
    """Plot one or more timeseries with flexible representation of uncertainty.

    This function is intended to be used with data where observations are
    nested within sampling units that were measured at multiple timepoints.

    It can take data specified either as a long-form (tidy) DataFrame or as an
    ndarray with dimensions (unit, time) The interpretation of some of the
    other parameters changes depending on the type of object passed as data.

    Parameters
    ----------
    data : DataFrame or ndarray
        Data for the plot. Should either be a "long form" dataframe or an
        array with dimensions (unit, time, condition). In both cases, the
        condition field/dimension is optional. The type of this argument
        determines the interpretation of the next few parameters. When
        using a DataFrame, the index has to be sequential.
    time : string or series-like
        Either the name of the field corresponding to time in the data
        DataFrame or x values for a plot when data is an array. If a Series,
        the name will be used to label the x axis.
    unit : string
        Field in the data DataFrame identifying the sampling unit (e.g.
        subject, neuron, etc.). The error representation will collapse over
        units at each time/condition observation. This has no role when data
        is an array.
    value : string
        Either the name of the field corresponding to the data values in
        the data DataFrame (i.e. the y coordinate) or a string that forms
        the y axis label when data is an array.
    condition : string or Series-like
        Either the name of the field identifying the condition an observation
        falls under in the data DataFrame, or a sequence of names with a length
        equal to the size of the third dimension of data. There will be a
        separate trace plotted for each condition. If condition is a Series
        with a name attribute, the name will form the title for the plot
        legend (unless legend is set to False).
    err_style : string or list of strings or None
        Names of ways to plot uncertainty across units from set of
        {ci_band, ci_bars, boot_traces, boot_kde, unit_traces, unit_points}.
        Can use one or more than one method.
    ci : float or list of floats in [0, 100]
        Confidence interval size(s). If a list, it will stack the error
        plots for each confidence interval. Only relevant for error styles
        with "ci" in the name.
    interpolate : boolean
        Whether to do a linear interpolation between each timepoint when
        plotting. The value of this parameter also determines the marker
        used for the main plot traces, unless marker is specified as a keyword
        argument.
    color : seaborn palette or matplotlib color name or dictionary
        Palette or color for the main plots and error representation (unless
        plotting by unit, which can be separately controlled with err_palette).
        If a dictionary, should map condition name to color spec.
    estimator : callable
        Function to determine central tendency and to pass to bootstrap
        must take an ``axis`` argument.
    n_boot : int
        Number of bootstrap iterations.
    err_palette : seaborn palette
        Palette name or list of colors used when plotting data for each unit.
    err_kws : dict, optional
        Keyword argument dictionary passed through to matplotlib function
        generating the error plot,
    legend : bool, optional
        If ``True`` and there is a ``condition`` variable, add a legend to
        the plot.
    ax : axis object, optional
        Plot in given axis; if None creates a new figure
    kwargs :
        Other keyword arguments are passed to main plot() call

    Returns
    -------
    ax : matplotlib axis
        axis with plot data

    Examples
    --------

    Plot a trace with translucent confidence bands:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(22)
        >>> import seaborn as sns; sns.set(color_codes=True)
        >>> x = np.linspace(0, 15, 31)
        >>> data = np.sin(x) + np.random.rand(10, 31) + np.random.randn(10, 1)
        >>> ax = sns.tsplot(data=data)

    Plot a long-form dataframe with several conditions:

    .. plot::
        :context: close-figs

        >>> gammas = sns.load_dataset("gammas")
        >>> ax = sns.tsplot(time="timepoint", value="BOLD signal",
        ...                 unit="subject", condition="ROI",
        ...                 data=gammas)

    Use error bars at the positions of the observations:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, err_style="ci_bars", color="g")

    Don't interpolate between the observations:

    .. plot::
        :context: close-figs

        >>> import matplotlib.pyplot as plt
        >>> ax = sns.tsplot(data=data, err_style="ci_bars", interpolate=False)

    Show multiple confidence bands:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, ci=[68, 95], color="m")

    Use a different estimator:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, estimator=np.median)

    Show each bootstrap resample:

    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, err_style="boot_traces", n_boot=500)

    Show the trace from each sampling unit:


    .. plot::
        :context: close-figs

        >>> ax = sns.tsplot(data=data, err_style="unit_traces")

    """
    # TODO: change y-axis formatter to be only integers, and add default
    # labels

    # Sort out default values for the parameters
    if ax is None:
        ax = plt.gca()

    # Handle different types of input data
    if isinstance(data, SpikeTrain):
        print("plotting SpikeTrain")
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data))))
    elif isinstance(data, SpikeTrainArray):
        colors = cmap(np.linspace(0.25, 0.75, data.n_units)) # TODO: if we go from 0 then most colormaps are invisible at one enf of the spectrum
        
        yrange = [.5, data.n_units + .5]
        
        for unit, spiketrain in enumerate(data.time):
            ax.vlines(spiketrain, unit + .55, unit + 1.45, colors=colors[unit], **kwargs)

        # change y-axis labels to integers
        yint = range(1, data.n_units+1)
        ax.set_yticks(yint)

        ax.set_ylim(yrange)

        # # plot the support epochs:
        # if plot_support:
        #     plot_epochs_hatch(ax=ax, epochs=data.support, height=data.n_units+1, fc='0.5', ec=None, alpha=0.2)

    elif isinstance(data, EpochArray):
        # TODO: how can I figure out an appropriate height when plotting a spiketrain support?
        plot_epochs_hatch(ax=ax, epochs=data, height=1e3, fc='0.2', ec=None, alpha=0.2, hatch=None)
    else:
        raise NotImplementedError(
            "plotting {} not yet supported".format(str(type(data))))

    # ax.plot(x, central_data, color=color, label=label, **kwargs)

    # # Pad the sides of the plot only when not interpolating
    # ax.set_xlim(x.min(), x.max())
    # x_diff = x[1] - x[0]
    # if not interpolate:
    #     ax.set_xlim(x.min() - x_diff, x.max() + x_diff)

    # # Add the plot labels
    # if xlabel is not None:
    #     ax.set_xlabel(xlabel)
    # if ylabel is not None:
    #     ax.set_ylabel(ylabel)
    # if legend:
    #     ax.legend(loc=0, title=legend_name)

    return ax

def plot_epochs_hatch(ax, epochs, height=None, fc='0.5', ec=None,
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

__all__ = ["annotate", "figure_grid", "savefig"]

# add ax2.yaxis.set_major_formatter(FixedOrderFormatter(-3))
# add scalebar
# add scale_grid
# add spike raster plot
# add 

def annotate(ax, text, xy=(0.5, 0.5), rotation=0, va=None, **kwargs):
    if rotation == 'vert' or rotation == 'v':
        rotation = 90
    if rotation == 'horz' or rotation == 'h':
        rotation = 0
    if va is None:
        if rotation == 90:
            va = 'bottom'
        else:
            va = 'baseline'

    ax.annotate(text, xy=xy, rotation=rotation, va=va, **kwargs)

def figure_grid(b=True, fig=None ):
    """draw a figure grid over an entore figure to facilitate annotation placement"""

    if fig is None:
        fig = plt.gcf()

    if b:
        # new clear axis overlay with 0-1 limits
        ax = fig.add_axes([0,0,1,1], axisbg=(1,1,1,0.7))
        ax.minorticks_on()
        ax.grid(b=True, which='major', color='k')
        ax.grid(b=True, which='minor', color='0.4', linestyle=':')
    else:
        pass

def get_extension_from_filename(name):
    """Extracts an extension from a filename string"""
    name = name.strip()
    ext = ((name.split('\\')[-1]).split('/')[-1]).split('.')
    if len(ext) > 1 and ext[-1] is not '':
        nameOnly = '.'.join(name.split('.')[:-1])
        ext = ext[-1]
    else:
        nameOnly = name
        ext = None
    return nameOnly, ext

def savefig(name, fig=None, formats=None, dpi=300, verbose=True, overwrite=False):
    """Saves a figure in one or multiple formats.

    Parameters
    ----------
    name : string
        Filename without an extension. If an extension is present, 
        AND if formats is empty, then the filename extension will be used.
    fig : matplotlib figure, optional
        Figure to save, default uses current figure.
    formats: list
        List of formats to export. Defaults to ['pdf', 'png']
    dpi: float
        Resolution of the figure in dots per inch (DPI).
    verbose: bool, optional
        If true, print additional output to screen.
    overwrite: bool, optional
        If true, file will be overwritten.

    Returns
    -------
    none
    
    """
    # Check inputs
    # if not 0 <= prop <= 1:
    #     raise ValueError("prop must be between 0 and 1")

    supportedFormats = ['eps', 'jpeg', 'jpg', 'pdf', 'pgf', 'png', 'ps', 'raw', 'rgba', 'svg', 'svgz', 'tif', 'tiff']

    name, ext = get_extension_from_filename(name)

   # if no list of formats is given, use defaults
    if formats is None and ext is None:
        formats = ['pdf','png']
    # if the filename has an extension, AND a list of extensions is given, then use only the list
    elif formats is not None and ext is not None:
        if not isinstance(formats, list):
            formats = [formats]
        print("WARNING! Extension in filename ignored in favor of formats list.")
    # if no list of extensions is given, use the extension from the filename
    elif formats is None and ext is not None:
        formats = [ext]
    else:
        print('WARNING! Unhandled format.')

    if fig is None:
        fig = plt.gcf()

    for extension in formats:
        if extension not in supportedFormats:
            print("WARNING! Format '{}' not supported. Aborting...".format(extension))
        else:
            my_file = 'figures/{}.{}'.format(name, extension)
            
            if os.path.isfile(my_file):
                # file exists
                print('{} already exists!'.format(my_file))
                
                if overwrite:
                    fig.savefig(my_file, dpi=dpi, bbox_inches='tight')
                    
                    if verbose:
                        print('{} saved successfully... [using overwrite]'.format(extension))
            else:
                fig.savefig(my_file, dpi=dpi, bbox_inches='tight')
                
                if verbose:
                    print('{} saved successfully...'.format(extension))

from matplotlib.ticker import ScalarFormatter
class FixedOrderFormatter(ScalarFormatter):
    """Formats axis ticks using scientific notation with a constant
    order of magnitude.

    Parameters
    ----------
    order_of_mag : int
        order of magnitude for the exponent
    useOffset : bool, optional
        If True includes an offset. Default is True.
    useMathText : bool, optional
        If True use 1x10^exp; otherwise use 1e-exp. Default is True.

    Example
    -------
    ax.yaxis.set_major_formatter(npl.FixedOrderFormatter(+2))

    See http://stackoverflow.com/questions/3677368/\
matplotlib-format-axis-offset-values-to-whole-numbers-\
or-specific-number
    """
    def __init__(self, order_of_mag=0, *, useOffset=None,
                 useMathText=None):
        # set parameter defaults:
        if useOffset is None:
            useOffset = True
        if useMathText is None:
            useMathText = True

        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset,
                                 useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        """Override to prevent order_of_mag being reset elsewhere."""
        self.orderOfMagnitude = self._order_of_mag

from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, *, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.5, borderpad=0.1, sep=2, prop=None, ec='k', fc='k', fontsize=None, lw=1.5, capstyle='round', xfirst=True, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - ec : edgecolor of scalebar
        - lw : linewidth of scalebar
        - fontsize : font size of labels
        - fc : font color / face color of labels
        - capstyle : capstyle of bars ['round', 'butt', 'projecting'] # TODO: NO LONGER USED
        - **kwargs : additional arguments passed to base class constructor
        
        adapted from https://gist.github.com/dmeliza/3251476
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        import matplotlib.patches as mpatches
        
        if fontsize is None:
            fontsize = mpl.rcParams['font.size']

        bars = AuxTransformBox(transform)
        
        if sizex and sizey:  # both horizontal and vertical scalebar
            endpt = (sizex, 0)
            art = mpatches.FancyArrowPatch((0, 0), endpt, color=ec, linewidth=lw,
                                        arrowstyle = mpatches.ArrowStyle.BarAB(widthA=0, widthB=lw*2))
            barsx = bars
            barsx.add_artist(art)
            endpt = (0, sizey)
            art = mpatches.FancyArrowPatch((0, 0), endpt, color=ec, linewidth=lw,
                                        arrowstyle = mpatches.ArrowStyle.BarAB(widthA=0, widthB=lw*2))
            barsy = bars
            barsy.add_artist(art)
        else:
            if sizex:
                endpt = (sizex, 0)
                art = mpatches.FancyArrowPatch((0, 0), endpt, color=ec, linewidth=lw,
                                            arrowstyle = mpatches.ArrowStyle.BarAB(widthA=lw*2, widthB=lw*2))
                bars.add_artist(art)
                
            if sizey:
                endpt = (0, sizey)
                art = mpatches.FancyArrowPatch((0, 0), endpt, color=ec, linewidth=lw,
                                            arrowstyle = mpatches.ArrowStyle.BarAB(widthA=lw*2, widthB=lw*2))
                bars.add_artist(art)

        if xfirst:
            if sizex and labelx:
                bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False, textprops=dict(color=fc, size=fontsize))],
                               align="center", pad=pad, sep=sep)
            if sizey and labely:
                bars = HPacker(children=[TextArea(labely, textprops=dict(color=fc, size=fontsize)), bars],
                                align="center", pad=pad, sep=sep)
        else:
            if sizey and labely:
                bars = HPacker(children=[TextArea(labely, textprops=dict(color=fc, size=fontsize)), bars],
                                align="center", pad=pad, sep=sep)
            if sizex and labelx:
                bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False, textprops=dict(color=fc, size=fontsize))],
                               align="center", pad=pad, sep=sep)
            
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)

def add_scalebar(ax, *, matchx=False, matchy=False, sizex=None, sizey=None, labelx=None, labely=None, hidex=True, hidey=True, verbose=False, **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    if sizex is None and not matchx:
        warnings.warn("either sizex or matchx must be set; assuming matchx = True")
        matchx = True

    if sizey is None and not matchy:
        warnings.warn("either sizey or matchy must be set; assuming matchy = True")
        matchy = True

    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    else:
        kwargs['sizex'] = sizex
        if labelx is not None:
            kwargs['labelx'] = str(labelx)
        else:
            kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
    else:
        kwargs['sizey'] = sizey
        if labely is not None:
            kwargs['labely'] = str(labely)
        else:
            kwargs['labely'] = str(kwargs['sizey'])
            
    # if x and y, create phantom objects to get centering done
    if sizex > 0 and sizey > 0:
        labely = kwargs['labely']
        kwargs['labely'] = ' '
        sb = AnchoredScaleBar(ax.transData, xfirst=True, **kwargs)
        ax.add_artist(sb)
        kwargs['labelx'] = ' '
        kwargs['labely'] = labely
        sb = AnchoredScaleBar(ax.transData, xfirst=False, **kwargs)
        ax.add_artist(sb)
    else:
        sb = AnchoredScaleBar(ax.transData, **kwargs)
        ax.add_artist(sb)

    if hidex: ax.xaxis.set_visible(False)
    if hidey: ax.yaxis.set_visible(False)

    return sb