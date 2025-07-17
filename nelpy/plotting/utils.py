# encoding : utf-8
"""This file contains the nelpy plotting functions and utilities.

Some functions Copyright (c) 2016, Etienne R. Ackermann
Some functions are modified from Jessica B. Hamrick, Copyright (c) 2013
'get_color_cycle', 'set_palette', and 'desaturate' are Copyright (c) 2012-2016, Michael L. Waskom
'FigureManager' modified from Camille Scott, Copyright (C) 2015 <camille.scott.w@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.
"""

# TODO: see https://gist.github.com/arnaldorusso/6611ff6c05e1efc2fb72
# TODO: see https://github.com/nengo/nengo/blob/master/nengo/utils/matplotlib.py

import colorsys
import os

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook, rcParams
from matplotlib import colors as mcolors
from matplotlib import colors as mplcolors
from matplotlib.image import AxesImage
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from packaging.version import parse as Version

mpl_ge_150 = Version(mpl.__version__) >= Version("1.5.0")

__all__ = [
    "add_colorbar",
    "align_xlabels",
    "align_ylabels",
    "annotate",
    "clear_bottom",
    "clear_left",
    "clear_left_right",
    "clear_right",
    "clear_top",
    "clear_top_bottom",
    "figure_grid",
    "FixedOrderFormatter",
    "no_xticklabels",
    "no_yticklabels",
    "no_xticks",
    "no_yticks",
    "outward_ticks",
    "savefig",
    "set_figsize",
    "set_scientific",
    "set_xlabel_coords",
    "set_ylabel_coords",
    "suptitle",
    "sync_xlims",
    "sync_ylims",
    "xticks_interval",
    "yticks_interval",
    "get_color_cycle",
    "FigureManager",
]


def add_colorbar(img, ax=None):
    """
    Add a colorbar to the given axis for the provided image.

    Parameters
    ----------
    img : matplotlib.image.AxesImage
        The image to which the colorbar applies.
    ax : matplotlib.axes.Axes, optional
        The axis to which the colorbar will be added. If None, uses current axis.

    Returns
    -------
    cb : matplotlib.colorbar.Colorbar
        The colorbar object.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from nelpy.plotting.utils import add_colorbar
    >>> fig, ax = plt.subplots()
    >>> im = ax.imshow(np.random.rand(10, 10))
    >>> cb = add_colorbar(im, ax=ax)
    >>> plt.show()
    """
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cb = plt.colorbar(img, cax=cax, orientation="vertical")
    # cb.set_label('probability', labelpad=-10)
    # cb.set_ticks([0,1])
    return cb


class FigureManager(object):
    """
    Figure context manager for creating, displaying, and saving figures.

    See http://stackoverflow.com/questions/12594148/skipping-execution-of-with-block
    but I was unable to get a solution so far...

    See http://stackoverflow.com/questions/11195140/break-or-exit-out-of-with-statement
    for additional inspiration for making nested context managers...

    Parameters
    ----------
    filename : str, optional
        Filename without an extension. If an extension is present,
        AND if formats is empty, then the filename extension will be used.
    save : bool, optional
        If True, figure will be saved to disk.
    show : bool, optional
        If True, figure will be shown.
    nrows : int, optional
        Number of subplot rows.
    ncols : int, optional
        Number of subplot columns.
    figsize : tuple, optional
        Figure size in inches (width, height).
    tight_layout : bool, optional
        If True, use tight layout.
    formats : list, optional
        List of formats to export. Defaults to ['pdf', 'png']
    dpi : float, optional
        Resolution of the figure in dots per inch (DPI).
    verbose : bool, optional
        If True, print additional output to screen.
    overwrite : bool, optional
        If True, file will be overwritten.
    **kwargs : dict
        Additional keyword arguments passed to plt.figure().

    Examples
    --------
    >>> with FigureManager(filename="myfig", save=True, show=False) as (fig, ax):
    ...     ax.plot([1, 2, 3], [4, 5, 6])
    ...
    """

    class Break(Exception):
        """Exception to break out of the context manager block."""
        pass

    def __init__(
        self,
        *,
        filename=None,
        save=False,
        show=False,
        nrows=1,
        ncols=1,
        figsize=(8, 3),
        tight_layout=False,
        formats=None,
        dpi=None,
        verbose=True,
        overwrite=False,
        **kwargs,
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.tight_layout = tight_layout
        self.dpi = dpi
        self.kwargs = kwargs

        self.filename = filename
        self.show = show
        self.save = save
        self.formats = formats
        self.dpi = dpi
        self.verbose = verbose
        self.overwrite = overwrite

        if self.show or self.save:
            self.skip = False
        else:
            self.skip = True

    def __enter__(self):
        """
        Enter the context manager, creating the figure and axes.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes or numpy.ndarray
            The created axes (single or array, depending on nrows/ncols).
        """
        if not self.skip:
            self.fig = plt.figure(figsize=self.figsize, dpi=self.dpi, **self.kwargs)
            self.fig.npl_gs = gridspec.GridSpec(nrows=self.nrows, ncols=self.ncols)

            self.ax = np.array([self.fig.add_subplot(ss) for ss in self.fig.npl_gs])
            # self.fig, self.ax = plt.subplots(nrows=self.nrows,
            #                                  ncols=self.ncols,
            #                                  figsize=self.figsize,
            #                                  tight_layout=self.tight_layout,
            #                                  dpi=self.dpi,
            #                                  **self.kwargs)
            if len(self.ax) == 1:
                self.ax = self.ax[0]

            if self.tight_layout:
                self.fig.npl_gs.tight_layout(self.fig)

            # gs1.tight_layout(fig, rect=[0, 0.03, 1, 0.95])
            if self.fig != plt.gcf():
                self.clear()
                raise RuntimeError("Figure does not match active mpl figure")
            return self.fig, self.ax
        return -1, -1

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context manager, saving and/or showing the figure if requested.

        Parameters
        ----------
        exc_type : type
            Exception type, if any.
        exc_value : Exception
            Exception value, if any.
        traceback : traceback
            Traceback object, if any.
        """
        if self.skip:
            return True
        if not exc_type:
            if self.save:
                assert self.filename is not None, "filename has to be specified!"
                savefig(
                    name=self.filename,
                    fig=self.fig,
                    formats=self.formats,
                    dpi=self.dpi,
                    verbose=self.verbose,
                    overwrite=self.overwrite,
                )

            if self.show:
                plt.show(self.fig)
            self.clear()
        else:
            self.clear()
            return False

    def clear(self):
        """
        Close the figure and clean up references.
        """
        plt.close(self.fig)
        del self.ax
        del self.fig


def suptitle(t, gs=None, rect=(0, 0, 1, 0.95), **kwargs):
    """
    Add a suptitle to a figure with an embedded gridspec.

    Parameters
    ----------
    t : str
        The suptitle text.
    gs : matplotlib.gridspec.GridSpec, optional
        The gridspec to use. If None, uses fig.npl_gs.
    rect : tuple, optional
        Rectangle in figure coordinates (x1, y1, x2, y2).
    **kwargs : dict
        Additional keyword arguments passed to fig.suptitle().

    Raises
    ------
    AttributeError
        If no gridspec is found in the figure.

    See Also
    --------
    https://matplotlib.org/users/tight_layout_guide.html
    """
    fig = plt.gcf()
    if gs is None:
        try:
            gs = fig.npl_gs
        except AttributeError:
            raise AttributeError(
                "nelpy suptitle requires an embedded gridspec! Use the nelpy FigureManager."
            )

    fig.suptitle(t, **kwargs)
    gs.tight_layout(fig, rect=rect)


def skip_if_no_output(fig):
    """
    Raise FigureManager.Break if the figure is not to be output.

    Parameters
    ----------
    fig : matplotlib.figure.Figure or int
        The figure object or -1 if not outputting.

    Returns
    -------
    bool
        True if output should continue, otherwise raises exception.
    """
    if fig == -1:
        raise FigureManager.Break
    return True


def annotate(text, ax=None, xy=None, rotation=None, va=None, **kwargs):
    """
    Annotate the given axis with text at a specified location and rotation.

    Parameters
    ----------
    text : str
        The annotation text.
    ax : matplotlib.axes.Axes, optional
        The axis to annotate. If None, uses current axis.
    xy : tuple, optional
        The (x, y) location for the annotation. Defaults to (0.5, 0.5).
    rotation : float or str, optional
        The rotation angle in degrees, or 'vert'/'v' for 90, 'horz'/'h' for 0.
    va : str, optional
        Vertical alignment. If None, chosen based on rotation.
    **kwargs : dict
        Additional keyword arguments passed to ax.annotate().

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> annotate('Hello', ax=ax, xy=(0.2, 0.8), rotation=45)
    >>> plt.show()
    """

    if ax is None:
        ax = plt.gca()
    if xy is None:
        xy = (0.5, 0.5)
    if rotation is None:
        rotation = 0
    if rotation == "vert" or rotation == "v":
        rotation = 90
    if rotation == "horz" or rotation == "h":
        rotation = 0
    if va is None:
        if rotation == 90:
            va = "bottom"
        else:
            va = "baseline"

    ax.annotate(text, xy=xy, rotation=rotation, va=va, **kwargs)


def figure_grid(b=True, fig=None):
    """
    Draw a figure grid over an entire figure to facilitate annotation placement.

    Parameters
    ----------
    b : bool, optional
        Whether to draw the grid (default True).
    fig : matplotlib.figure.Figure, optional
        The figure to draw on. If None, uses current figure.
    """

    if fig is None:
        fig = plt.gcf()

    if b:
        # new clear axis overlay with 0-1 limits
        ax = fig.add_axes([0, 0, 1, 1], axisbg=(1, 1, 1, 0.7))
        ax.minorticks_on()
        ax.grid(b=True, which="major", color="k")
        ax.grid(b=True, which="minor", color="0.4", linestyle=":")
    else:
        pass


def get_extension_from_filename(name):
    """
    Extract the extension from a filename string.

    Parameters
    ----------
    name : str
        The filename string.

    Returns
    -------
    nameOnly : str
        The filename without extension.
    ext : str or None
        The extension, or None if not present.
    """
    name = name.strip()
    ext = ((name.split("\\")[-1]).split("/")[-1]).split(".")
    if len(ext) > 1 and ext[-1] != "":
        nameOnly = ".".join(name.split(".")[:-1])
        ext = ext[-1]
    else:
        nameOnly = name
        ext = None
    return nameOnly, ext


def savefig(name, fig=None, formats=None, dpi=None, verbose=True, overwrite=False):
    """
    Save a figure in one or multiple formats.

    Parameters
    ----------
    name : str
        Filename without an extension. If an extension is present,
        AND if formats is empty, then the filename extension will be used.
    fig : matplotlib.figure.Figure, optional
        Figure to save, default uses current figure.
    formats : list, optional
        List of formats to export. Defaults to ['pdf', 'png']
    dpi : float, optional
        Resolution of the figure in dots per inch (DPI).
    verbose : bool, optional
        If True, print additional output to screen.
    overwrite : bool, optional
        If True, file will be overwritten.

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6])
    >>> savefig('myplot', fig=fig, formats=['png'], overwrite=True)
    """
    # Check inputs
    # if not 0 <= prop <= 1:
    #     raise ValueError("prop must be between 0 and 1")

    if dpi is None:
        dpi = 300

    supportedFormats = [
        "eps",
        "jpeg",
        "jpg",
        "pdf",
        "pgf",
        "png",
        "ps",
        "raw",
        "rgba",
        "svg",
        "svgz",
        "tif",
        "tiff",
    ]

    name, ext = get_extension_from_filename(name)

    # if no list of formats is given, use defaults
    if formats is None and ext is None:
        formats = ["pdf", "png"]
    # if the filename has an extension, AND a list of extensions is given, then use only the list
    elif formats is not None and ext is not None:
        if not isinstance(formats, list):
            formats = [formats]
        print("WARNING! Extension in filename ignored in favor of formats list.")
    # if no list of extensions is given, use the extension from the filename
    elif formats is None and ext is not None:
        formats = [ext]
    else:
        pass

    if fig is None:
        fig = plt.gcf()

    for extension in formats:
        if extension not in supportedFormats:
            print("WARNING! Format '{}' not supported. Aborting...".format(extension))
        else:
            my_file = "figures/{}.{}".format(name, extension)

            if os.path.isfile(my_file):
                # file exists
                print("{} already exists!".format(my_file))

                if overwrite:
                    fig.savefig(my_file, dpi=dpi, bbox_inches="tight")

                    if verbose:
                        print(
                            "{} saved successfully... [using overwrite]".format(
                                extension
                            )
                        )
            else:
                fig.savefig(my_file, dpi=dpi, bbox_inches="tight")

                if verbose:
                    print("{} saved successfully...".format(extension))


class FixedOrderFormatter(ScalarFormatter):
    """
    Formats axis ticks using scientific notation with a constant order of magnitude.

    Parameters
    ----------
    order_of_mag : int
        Order of magnitude for the exponent.
    useOffset : bool, optional
        If True includes an offset. Default is True.
    useMathText : bool, optional
        If True use 1x10^exp; otherwise use 1e-exp. Default is True.

    Examples
    --------
    Force the y-axis ticks to use 1e+2 as a base exponent:
    >>> ax.yaxis.set_major_formatter(npl.FixedOrderFormatter(+2))

    Make the x-axis ticks formatted to 0 decimal places:
    >>> from matplotlib.ticker import FormatStrFormatter
    >>> ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

    Turn off offset on x-axis:
    >>> ax.xaxis.get_major_formatter().set_useOffset(False)
    """

    def __init__(self, order_of_mag=0, *, useOffset=None, useMathText=None):
        # set parameter defaults:
        if useOffset is None:
            useOffset = True
        if useMathText is None:
            useMathText = True

        self._order_of_mag = order_of_mag
        ScalarFormatter.__init__(self, useOffset=useOffset, useMathText=useMathText)

    def _set_orderOfMagnitude(self, range):
        """Override to prevent order_of_mag being reset elsewhere."""
        self.orderOfMagnitude = self._order_of_mag


def xticks_interval(step=10, *axes):
    """Set xticks interval."""
    if len(axes) == 0:
        axes = [plt.gca()]
    loc = mpl.ticker.MultipleLocator(
        base=step
    )  # this locator puts ticks at regular intervals
    for ax in axes:
        ax.xaxis.set_major_locator(loc)


def yticks_interval(step=10, *axes):
    """Set yticks interval."""
    if len(axes) == 0:
        axes = [plt.gca()]
    loc = mpl.ticker.MultipleLocator(
        base=step
    )  # this locator puts ticks at regular intervals
    for ax in axes:
        ax.yaxis.set_major_locator(loc)


def clear_top(*axes):
    """Remove the top edge of the axis bounding box.

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.spines["top"].set_color("none")
        ax.xaxis.set_ticks_position("bottom")


def clear_bottom(*axes):
    """Remove the bottom edge of the axis bounding box.

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.spines["bottom"].set_color("none")
        ax.xaxis.set_ticks_position("top")


def clear_top_bottom(*axes):
    """Remove the top and bottom edges of the axis bounding box.

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.spines["top"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.xaxis.set_ticks([])


def clear_left(*axes):
    """Remove the left edge of the axis bounding box.

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.spines["left"].set_color("none")
        ax.yaxis.set_ticks_position("right")


def clear_right(*axes):
    """Remove the right edge of the axis bounding box.

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.spines["right"].set_color("none")
        ax.yaxis.set_ticks_position("left")


def clear_left_right(*axes):
    """Remove the left and right edges of the axis bounding box.

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.spines["left"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.yaxis.set_ticks([])


def outward_ticks(*axes, axis="both"):
    """Make axis ticks face outwards rather than inwards (which is the
    default).

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())
    axis : string (default='both')
        The axis (either 'x', 'y', or 'both') for which to set the tick
        direction.

    """

    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        if axis == "both":
            ax.tick_params(direction="out")
        else:
            ax.tick_params(axis=axis, direction="out")


def set_xlabel_coords(y, *axes, x=0.5):
    """Set the y-coordinate (and optionally the x-coordinate) of the x-axis
    label.

    Parameters
    ----------
    y : float
        y-coordinate for the label
    x : float (default=0.5)
        x-coordinate for the label
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.xaxis.set_label_coords(x, y)


def set_ylabel_coords(x, *axes, y=0.5):
    """Set the x-coordinate (and optionally the y-coordinate) of the y-axis
    label.

    Parameters
    ----------
    x : float
        x-coordinate for the label
    y : float (default=0.5)
        y-coordinate for the label
    axes : axis object (default=pyplot.gca())

    References
    ----------
    http://matplotlib.org/faq/howto_faq.html#align-my-ylabels-across-multiple-subplots

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
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
    set_ylabel_coords(xcoord, *axes)
    # for ax in axes:
    #     set_ylabel_coords(xcoord, ax=ax)


def align_xlabels(ycoord, *axes):
    """Align the x-axis labels of multiple axes

    Parameters
    ----------
    ycoord : float
        y-coordinate of the x-axis labels
    *axes : axis objects
        The matplotlib axis objects to format

    """
    set_xlabel_coords(ycoord, *axes)


def no_xticks(*axes):
    """Remove the tick marks on the x-axis (but leave the labels).

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.tick_params(axis="x", which="both", length=0)


def no_yticks(*axes):
    """Remove the tick marks on the y-axis (but leave the labels).

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.tick_params(axis="y", which="both", length=0)


def no_ticks(*axes, where=None):
    """Remove the tick marks on the desired axes (but leave the labels).

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())
    where : string, optional (default 'all') or list
        Where to remove ticks ['left', 'right', 'top', 'bottom', 'all']

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    if where is None:
        where = ["all"]

    if isinstance(where, str):
        where = [where]
    for ax in axes:
        if "left" in where:
            ax.tick_params(axis="y", which="both", left=False)
        if "right" in where:
            ax.tick_params(axis="y", which="both", right=False)
        if "top" in where:
            ax.tick_params(axis="x", which="both", top=False)
        if "bottom" in where:
            ax.tick_params(axis="x", which="both", bottom=False)
        if "all" in where:
            ax.tick_params(axis="y", which="both", left=False)
            ax.tick_params(axis="y", which="both", right=False)
            ax.tick_params(axis="x", which="both", top=False)
            ax.tick_params(axis="x", which="both", bottom=False)


def no_xticklabels(*axes):
    """Remove the tick labels on the x-axis (but leave the tick marks).

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
        ax.set_xticklabels([])


def no_yticklabels(*axes):
    """Remove the tick labels on the y-axis (but leave the tick marks).

    Parameters
    ----------
    axes : axis object (default=pyplot.gca())

    """
    if len(axes) == 0:
        axes = [plt.gca()]
    for ax in axes:
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


def set_scientific(low, high, axis=None, *axes):
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
    axes : axis object (default=pyplot.gca())
        The matplotlib axis object to use

    """
    # get the axis
    if len(axes) == 0:
        axes = [plt.gca()]
    # create the tick label formatter
    fmt = plt.ScalarFormatter()
    fmt.set_scientific(True)
    fmt.set_powerlimits((low, high))
    # format the axis/axes
    for ax in axes:
        if axis is None or axis == "x":
            ax.get_yaxis().set_major_formatter(fmt)
        if axis is None or axis == "y":
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


def set_xlim(xlims, *axes):
    """Sets the xlims for all axes.

    Parameters
    ----------
    xlims : tuple? list?
    *axes : axis objects
        List of matplotlib axis objects to format

    Returns
    -------
    None

    """
    for ax in axes:
        ax.set_xlim(xlims[0], xlims[1])


def set_ylim(ylims, *axes):
    """Sets the ylims for all axes.

    Parameters
    ----------
    ylims : tuple? list?
    *axes : axis objects
        List of matplotlib axis objects to format

    Returns
    -------
    None

    """
    for ax in axes:
        ax.set_ylim(ylims[0], ylims[1])


def get_color_cycle():
    if mpl_ge_150:
        cyl = mpl.rcParams["axes.prop_cycle"]
        # matplotlib 1.5 verifies that axes.prop_cycle *is* a cycler
        # but no garuantee that there's a `color` key.
        # so users could have a custom rcParmas w/ no color...
        try:
            return [x["color"] for x in cyl]
        except KeyError:
            pass  # just return axes.color style below
    return mpl.rcParams["axes.color_cycle"]


def desaturate(color, prop):
    """Decrease the saturation channel of a color by some percent.
    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    prop : float
        saturation channel of color will be multiplied by this value
    Returns
    -------
    new_color : rgb tuple
        desaturated color code in RGB tuple representation
    """
    # Check inputs
    if not 0 <= prop <= 1:
        raise ValueError("prop must be between 0 and 1")

    # Get rgb tuple rep
    rgb = mplcolors.colorConverter.to_rgb(color)

    # Convert to hls
    h, lightness, s = colorsys.rgb_to_hls(*rgb)

    # Desaturate the saturation channel
    s *= prop

    # Convert back to rgb
    new_color = colorsys.hls_to_rgb(h, lightness, s)

    return new_color


class ModestImage(AxesImage):
    """Computationally modest image class.

    Customization of https://github.com/ChrisBeaumont/ModestImage to allow
    extent support.

    ModestImage is an extension of the Matplotlib AxesImage class
    better suited for the interactive display of larger images. Before
    drawing, ModestImage resamples the data array based on the screen
    resolution and view window. This has very little affect on the
    appearance of the image, but can substantially cut down on
    computation since calculations of unresolved or clipped pixels
    are skipped.

    The interface of ModestImage is the same as AxesImage. However, it
    does not currently support setting the 'extent' property. There
    may also be weird coordinate warping operations for images that
    I'm not aware of. Don't expect those to work either.
    """

    def __init__(self, *args, **kwargs):
        self._full_res = None
        self._sx, self._sy = None, None
        self._bounds = (None, None, None, None)
        self._origExtent = None
        super(ModestImage, self).__init__(*args, **kwargs)
        if "extent" in kwargs and kwargs["extent"] is not None:
            self.set_extent(kwargs["extent"])

    def set_extent(self, extent):
        super(ModestImage, self).set_extent(extent)
        if self._origExtent is None:
            self._origExtent = self.get_extent()

    def get_image_extent(self):
        """Returns the extent of the whole image.

        get_extent returns the extent of the drawn area and not of the full
        image.

        :return: Bounds of the image (x0, x1, y0, y1).
        :rtype: Tuple of 4 floats.
        """
        if self._origExtent is not None:
            return self._origExtent
        else:
            return self.get_extent()

    def set_data(self, A):
        """
        Set the image array

        ACCEPTS: numpy/PIL Image A
        """

        self._full_res = A
        self._A = A

        if self._A.dtype != np.uint8 and not np.can_cast(self._A.dtype, np.float):
            raise TypeError("Image data can not convert to float")

        if self._A.ndim not in (2, 3) or (
            self._A.ndim == 3 and self._A.shape[-1] not in (3, 4)
        ):
            raise TypeError("Invalid dimensions for image data")

        self._imcache = None
        self._rgbacache = None
        self._oldxslice = None
        self._oldyslice = None
        self._sx, self._sy = None, None

    def get_array(self):
        """Override to return the full-resolution array"""
        return self._full_res

    def _scale_to_res(self):
        """Change self._A and _extent to render an image whose
        resolution is matched to the eventual rendering."""
        # extent has to be set BEFORE set_data
        if self._origExtent is None:
            if self.origin == "upper":
                self._origExtent = (
                    0,
                    self._full_res.shape[1],
                    self._full_res.shape[0],
                    0,
                )
            else:
                self._origExtent = (
                    0,
                    self._full_res.shape[1],
                    0,
                    self._full_res.shape[0],
                )

        if self.origin == "upper":
            origXMin, origXMax, origYMax, origYMin = self._origExtent[0:4]
        else:
            origXMin, origXMax, origYMin, origYMax = self._origExtent[0:4]
        ax = self.axes
        ext = ax.transAxes.transform([1, 1]) - ax.transAxes.transform([0, 0])
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        xlim = max(xlim[0], origXMin), min(xlim[1], origXMax)
        if ylim[0] > ylim[1]:
            ylim = max(ylim[1], origYMin), min(ylim[0], origYMax)
        else:
            ylim = max(ylim[0], origYMin), min(ylim[1], origYMax)
        # print("THOSE LIMITS ARE TO BE COMPARED WITH THE EXTENT")
        # print("IN ORDER TO KNOW WHAT IT IS LIMITING THE DISPLAY")
        # print("IF THE AXES OR THE EXTENT")
        dx, dy = xlim[1] - xlim[0], ylim[1] - ylim[0]

        y0 = max(0, ylim[0] - 5)
        y1 = min(self._full_res.shape[0], ylim[1] + 5)
        x0 = max(0, xlim[0] - 5)
        x1 = min(self._full_res.shape[1], xlim[1] + 5)
        y0, y1, x0, x1 = [int(a) for a in [y0, y1, x0, x1]]

        sy = int(max(1, min((y1 - y0) / 5.0, np.ceil(dy / ext[1]))))
        sx = int(max(1, min((x1 - x0) / 5.0, np.ceil(dx / ext[0]))))

        # have we already calculated what we need?
        if (self._sx is not None) and (self._sy is not None):
            if (
                sx >= self._sx
                and sy >= self._sy
                and x0 >= self._bounds[0]
                and x1 <= self._bounds[1]
                and y0 >= self._bounds[2]
                and y1 <= self._bounds[3]
            ):
                return

        self._A = self._full_res[y0:y1:sy, x0:x1:sx]
        self._A = cbook.safe_masked_invalid(self._A)
        x1 = x0 + self._A.shape[1] * sx
        y1 = y0 + self._A.shape[0] * sy

        if self.origin == "upper":
            self.set_extent([x0, x1, y1, y0])
        else:
            self.set_extent([x0, x1, y0, y1])
        self._sx = sx
        self._sy = sy
        self._bounds = (x0, x1, y0, y1)
        self.changed()

    def draw(self, renderer, *args, **kwargs):
        self._scale_to_res()
        super(ModestImage, self).draw(renderer, *args, **kwargs)


def imshow(
    axes,
    X,
    cmap=None,
    norm=None,
    aspect=None,
    interpolation=None,
    alpha=None,
    vmin=None,
    vmax=None,
    origin=None,
    extent=None,
    shape=None,
    filternorm=1,
    filterrad=4.0,
    imlim=None,
    resample=None,
    url=None,
    clearaxes=True,
    **kwargs,
):
    """Similar to matplotlib's imshow command, but produces a ModestImage

    Unlike matplotlib version, must explicitly specify axes
    """

    if clearaxes:
        axes.cla()
    if norm is not None:
        assert isinstance(norm, mcolors.Normalize)
    if aspect is None:
        aspect = rcParams["image.aspect"]
    axes.set_aspect(aspect)
    im = ModestImage(
        axes,
        cmap,
        norm,
        interpolation,
        origin,
        extent,
        filternorm=filternorm,
        filterrad=filterrad,
        resample=resample,
        **kwargs,
    )

    im.set_data(X)
    im.set_alpha(alpha)
    axes._set_artist_props(im)

    if im.get_clip_path() is None:
        # image does not already have clipping set, clip to axes patch
        im.set_clip_path(axes.patch)

    # if norm is None and shape is None:
    #    im.set_clim(vmin, vmax)
    if vmin is not None or vmax is not None:
        im.set_clim(vmin, vmax)
    elif norm is None:
        im.autoscale_None()

    im.set_url(url)

    # update ax.dataLim, and, if autoscaling, set viewLim
    # to tightly fit the image, regardless of dataLim.
    im.set_extent(im.get_extent())

    axes.images.append(im)
    im._remove_method = lambda h: axes.images.remove(h)

    return im
