"""Small plotting-related utility functions."""
# Some small functions Copyright (c) 2013 Jessica B. Hamrick
import warnings
import os.path

import matplotlib as mpl
import matplotlib.pyplot as plt

from distutils.version import LooseVersion
mpl_ge_150 = LooseVersion(mpl.__version__) >= "1.5.0"


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
    """Formats axis ticks using scientific notation with a constant order of 
    magnitude
    
    Example
    ===========
    ax.yaxis.set_major_formatter(FixedOrderFormatter(+2))"""
    def __init__(self, order_of_mag=0, useOffset=True, useMathText=True):
        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset, 
                                 useMathText=useMathText)
    def _set_orderOfMagnitude(self, range):
        """Over-riding this to avoid having orderOfMagnitude reset elsewhere"""
        self.orderOfMagnitude = self._order_of_mag

def set_major_formatter(ax, order_of_mag=0, useOffset=True, useMathText=True):
    """Wrapper function to use with eplotlib."""
    ax.set_major_formatter(FixedOrderFormatter(order_of_mag=order_of_mag,
                            useOffset=useOffset, 
                            useMathText=useMathText))

from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, *, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, ec='k', lw=2, capstyle='round', **kwargs):
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
        - capstyle : capstyle of bars ['round', 'butt', 'projecting']
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, 0, fc="none", edgecolor=ec, linewidth=lw, capstyle=capstyle))
        if sizey:
            bars.add_artist(Rectangle((0,0), 0, sizey, fc="none", edgecolor=ec, linewidth=lw, capstyle=capstyle))

        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)

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
        return len(l)>1 and (l[1] - l[0])
        
    if sizex is None and not matchx:
        print('warning! either sizex or matchx must be set; assuming matchx = True')
        matchx = True

    if sizey is None and not matchy:
        print('warning! either sizey or matchy must be set; assuming matchy = True')
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
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    ax.add_artist(sb)

    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)

    return sb

def clear_top(ax=None):
    """Remove the top edge of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')


def clear_bottom(ax=None):
    """Remove the bottom edge of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks_position('top')


def clear_top_bottom(ax=None):
    """Remove the top and bottom edges of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.xaxis.set_ticks([])


def clear_left(ax=None):
    """Remove the left edge of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.yaxis.set_ticks_position('right')


def clear_right(ax=None):
    """Remove the right edge of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks_position('left')


def clear_left_right(ax=None):
    """Remove the left and right edges of the axis bounding box.

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if ax is None:
        ax = plt.gca()
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.yaxis.set_ticks([])


def outward_ticks(ax=None, axis='both'):
    """Make axis ticks face outwards rather than inwards (which is the
    default).

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())
    axis : string (default='both')
        The axis (either 'x', 'y', or 'both') for which to set the tick
        direction.

    """

    if ax is None:
        ax = plt.gca()
    if axis == 'both':
        ax.tick_params(direction='out')
    else:
        ax.tick_params(axis=axis, direction='out')


def set_xlabel_coords(y, x=0.5, ax=None):
    """Set the y-coordinate (and optionally the x-coordinate) of the x-axis
    label.

    Parameters
    ----------
    y : float
        y-coordinate for the label
    x : float (default=0.5)
        x-coordinate for the label
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots

    """
    if ax is None:
        ax = plt.gca()
    ax.xaxis.set_label_coords(x, y)


def set_ylabel_coords(x, y=0.5, ax=None):
    """Set the x-coordinate (and optionally the y-coordinate) of the y-axis
    label.

    Parameters
    ----------
    x : float
        x-coordinate for the label
    y : float (default=0.5)
        y-coordinate for the label
    ax : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots

    """
    if ax is None:
        ax = plt.gca()
    ax.yaxis.set_label_coords(x, y)


def align_ylabels(xcoord, *axes):
    """Align the y-axis labels of multiple axes.

    Parameters
    ----------
    xcoord : float
        x-coordinate of the y-axis labels
    *axes : axis objects
        The matplotlib axis objects to format

    """
    for ax in axes:
        set_ylabel_coords(xcoord, ax=ax)


def align_xlabels(ycoord, *axes):
    """Align the x-axis labels of multiple axes

    Parameters
    ----------
    ycoord : float
        y-coordinate of the x-axis labels
    *axes : axis objects
        The matplotlib axis objects to format

    """
    for ax in axes:
        set_xlabel_coords(ycoord, ax=ax)


def no_xticklabels(ax=None):
    """Remove the tick labels on the x-axis (but leave the tick marks).

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    """
    if ax is None:
        ax = plt.gca()
    ax.set_xticklabels([])


def no_yticklabels(ax=None):
    """Remove the tick labels on the y-axis (but leave the tick marks).

    Parameters
    ----------
    ax : axis object (default=pyplot.gca())

    """
    if ax is None:
        ax = plt.gca()
    ax.set_yticklabels([])


def set_figsize(width, height, fig=None):
    """Set the figure width and height.

    Parameters
    ----------
    width : float
        Figure width
    height : float
        Figure height
    fig : figure object (default=pyplot.gcf())

    """

    if fig is None:
        fig = plt.gcf()
    fig.set_figwidth(width)
    fig.set_figheight(height)


def set_scientific(low, high, axis=None, ax=None):
    """Set the axes or axis specified by `axis` to use scientific notation for
    ticklabels, if the value is <10**low or >10**high.

    Parameters
    ----------
    low : int
        Lower exponent bound for non-scientific notation
    high : int
        Upper exponent bound for non-scientific notation
    axis : str (default=None)
        Which axis to format ('x', 'y', or None for both)
    ax : axis object (default=pyplot.gca())
        The matplotlib axis object to use

    """
    # get the axis
    if ax is None:
        ax = plt.gca()
    # create the tick label formatter
    fmt = plt.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((low, high))
    # format the axis/axes
    if axis is None or axis == 'x':
        ax.get_yaxis().set_major_formatter(fmt)
    if axis is None or axis == 'y':
        ax.get_yaxis().set_major_formatter(fmt)


def sync_ylims(*axes):
    """Synchronize the y-axis data limits for multiple axes. Uses the maximum
    upper limit and minimum lower limit across all given axes.

    Parameters
    ----------
    *axes : axis objects
        List of matplotlib axis objects to format

    Returns
    -------
    out : ymin, ymax
        The computed bounds

    """
    ymins, ymaxs = zip(*[ax.get_ylim() for ax in axes])
    ymin = min(ymins)
    ymax = max(ymaxs)
    for ax in axes:
        ax.set_ylim(ymin, ymax)
    return ymin, ymax


def sync_xlims(*axes):
    """Synchronize the x-axis data limits for multiple axes. Uses the maximum
    upper limit and minimum lower limit across all given axes.

    Parameters
    ----------
    *axes : axis objects
        List of matplotlib axis objects to format

    Returns
    -------
    out : yxin, xmax
        The computed bounds

    """
    xmins, xmaxs = zip(*[ax.get_xlim() for ax in axes])
    xmin = min(xmins)
    xmax = max(xmaxs)
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        return xmin, xmax
