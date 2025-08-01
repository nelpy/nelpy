# TODO: add docstring

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox


class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(
        self,
        transform,
        *,
        sizex=0,
        sizey=0,
        labelx=None,
        labely=None,
        loc=4,
        pad=0.5,
        borderpad=0.1,
        sep=2,
        prop=None,
        ec="k",
        fc="k",
        fontsize=None,
        lw=1.5,
        capstyle="projecting",
        xfirst=True,
        **kwargs,
    ):
        """
        Create an anchored scale bar for matplotlib axes.

        Parameters
        ----------
        transform : matplotlib transform
            The coordinate frame (typically axes.transData).
        sizex : float, optional
            Width of the x bar, in data units. 0 to omit. Default is 0.
        sizey : float, optional
            Height of the y bar, in data units. 0 to omit. Default is 0.
        labelx : str, optional
            Label for the x bar. None to omit.
        labely : str, optional
            Label for the y bar. None to omit.
        loc : int, optional
            Location in containing axes (see matplotlib legend locations). Default is 4 (lower right).
        pad : float, optional
            Padding, in fraction of the legend font size. Default is 0.5.
        borderpad : float, optional
            Border padding, in fraction of the legend font size. Default is 0.1.
        sep : float, optional
            Separation between labels and bars in points. Default is 2.
        prop : font properties, optional
            Font properties for the labels.
        ec : color, optional
            Edge color of the scalebar. Default is 'k'.
        fc : color, optional
            Font color / face color of labels. Default is 'k'.
        fontsize : float, optional
            Font size of labels. If None, uses matplotlib default.
        lw : float, optional
            Line width of the scalebar. Default is 1.5.
        capstyle : {'round', 'butt', 'projecting'}, optional
            Cap style of bars. Default is 'projecting'.
        xfirst : bool, optional
            If True, draw x bar and label first. Default is True.
        **kwargs : dict
            Additional arguments passed to base constructor.

        Notes
        -----
        Adapted from https://gist.github.com/dmeliza/3251476

        Examples
        --------
        >>> from nelpy.plotting.scalebar import AnchoredScaleBar
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> scalebar = AnchoredScaleBar(ax.transData, sizex=1, labelx="1 s")
        >>> ax.add_artist(scalebar)
        """
        import matplotlib.patches as mpatches
        from matplotlib.offsetbox import AuxTransformBox, HPacker, TextArea, VPacker

        if fontsize is None:
            fontsize = mpl.rcParams["font.size"]

        bars = AuxTransformBox(transform)

        if sizex and sizey:  # both horizontal and vertical scalebar
            # hacky fix for possible misalignment errors that may occur
            #  on small figures
            if ec is None:
                lw = 0
            endpt = (sizex, 0)
            art = mpatches.FancyArrowPatch(
                (0, 0),
                endpt,
                color=ec,
                linewidth=lw,
                capstyle=capstyle,
                arrowstyle=mpatches.ArrowStyle.BarAB(widthA=0, widthB=lw * 2),
            )
            barsx = bars
            barsx.add_artist(art)
            endpt = (0, sizey)
            art = mpatches.FancyArrowPatch(
                (0, 0),
                endpt,
                color=ec,
                linewidth=lw,
                capstyle=capstyle,
                arrowstyle=mpatches.ArrowStyle.BarAB(widthA=0, widthB=lw * 2),
            )
            barsy = bars
            barsy.add_artist(art)
        else:
            if sizex:
                endpt = (sizex, 0)
                art = mpatches.FancyArrowPatch(
                    (0, 0),
                    endpt,
                    color=ec,
                    linewidth=lw,
                    arrowstyle=mpatches.ArrowStyle.BarAB(widthA=lw * 2, widthB=lw * 2),
                )
                bars.add_artist(art)

            if sizey:
                endpt = (0, sizey)
                art = mpatches.FancyArrowPatch(
                    (0, 0),
                    endpt,
                    color=ec,
                    linewidth=lw,
                    arrowstyle=mpatches.ArrowStyle.BarAB(widthA=lw * 2, widthB=lw * 2),
                )
                bars.add_artist(art)

        if xfirst:
            if sizex and labelx:
                bars = VPacker(
                    children=[
                        bars,
                        TextArea(
                            labelx,
                            minimumdescent=False,
                            textprops=dict(color=fc, size=fontsize),
                        ),
                    ],
                    align="center",
                    pad=pad,
                    sep=sep,
                )
            if sizey and labely:
                bars = HPacker(
                    children=[
                        TextArea(labely, textprops=dict(color=fc, size=fontsize)),
                        bars,
                    ],
                    align="center",
                    pad=pad,
                    sep=sep,
                )
        else:
            if sizey and labely:
                bars = HPacker(
                    children=[
                        TextArea(labely, textprops=dict(color=fc, size=fontsize)),
                        bars,
                    ],
                    align="center",
                    pad=pad,
                    sep=sep,
                )
            if sizex and labelx:
                bars = VPacker(
                    children=[
                        bars,
                        TextArea(
                            labelx,
                            minimumdescent=False,
                            textprops=dict(color=fc, size=fontsize),
                        ),
                    ],
                    align="center",
                    pad=pad,
                    sep=sep,
                )

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs,
        )


def add_simple_scalebar(
    text,
    ax=None,
    xy=None,
    length=None,
    orientation="v",
    rotation_text=None,
    xytext=None,
    **kwargs,
):
    """
    Add a simple horizontal or vertical scalebar with a label to an axis.

    Parameters
    ----------
    text : str
        The label for the scalebar.
    ax : matplotlib.axes.Axes, optional
        Axis to add the scalebar to. If None, uses current axis.
    xy : tuple of float
        Starting (x, y) position for the scalebar.
    length : float, optional
        Length of the scalebar. Default is 10.
    orientation : {'v', 'h', 'vert', 'horz'}, optional
        Orientation of the scalebar. 'v' or 'vert' for vertical, 'h' or 'horz' for horizontal. Default is 'v'.
    rotation_text : int or str, optional
        Rotation of the label text. Default is 0.
    xytext : tuple of float, optional
        Position for the label text. If None, automatically determined.
    **kwargs : dict
        Additional keyword arguments passed to matplotlib's annotate.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> add_simple_scalebar("10 s", ax=ax, xy=(0, 0), length=10, orientation="h")
    """
    if rotation_text is None:
        rotation_text = 0
    if rotation_text == "vert" or rotation_text == "v":
        rotation_text = 90
    if rotation_text == "horz" or rotation_text == "h":
        rotation_text = 0
    if orientation is None:
        orientation = 0
    if orientation == "vert" or orientation == "v":
        orientation = 90
    if orientation == "horz" or orientation == "h":
        orientation = 0

    if length is None:
        length = 10

    if ax is None:
        ax = plt.gca()

    #     if va is None:
    #         if rotation_text == 90:
    #             va = 'bottom'
    #         else:
    #             va = 'baseline'

    if orientation == 0:
        ax.hlines(xy[1], xy[0], xy[0] + length, lw=2, zorder=1000)
    else:
        ax.vlines(xy[0], xy[1], xy[1] + length, lw=2, zorder=1000)
        xytext = (xy[0] + 3, xy[1] + length / 2)
        ax.annotate(
            text, xy=xytext, rotation=rotation_text, va="center", zorder=1000, **kwargs
        )


def add_scalebar(
    ax,
    *,
    matchx=False,
    matchy=False,
    sizex=None,
    sizey=None,
    labelx=None,
    labely=None,
    hidex=True,
    hidey=True,
    ec="k",
    **kwargs,
):
    """
    Add scalebars to axes, matching the size to the ticks of the plot and optionally hiding the x and y axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to attach scalebars to.
    matchx : bool, optional
        If True, set size of x scalebar to spacing between ticks. Default is False.
    matchy : bool, optional
        If True, set size of y scalebar to spacing between ticks. Default is False.
    sizex : float, optional
        Size of x scalebar. Used if matchx is False.
    sizey : float, optional
        Size of y scalebar. Used if matchy is False.
    labelx : str, optional
        Label for x scalebar.
    labely : str, optional
        Label for y scalebar.
    hidex : bool, optional
        If True, hide x-axis of parent. Default is True.
    hidey : bool, optional
        If True, hide y-axis of parent. Default is True.
    ec : color, optional
        Edge color of the scalebar. Default is 'k'.
    **kwargs : dict
        Additional arguments passed to AnchoredScaleBar.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The axis containing the scalebar object.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> fig, ax = plt.subplots()
    >>> add_scalebar(ax, sizex=1, labelx="1 s")
    """

    # determine which type op scalebar to plot:
    # [(horizontal, vertical, both), (matchx, matchy), (labelx, labely)]
    #
    # matchx AND sizex ==> error
    # matchy AND sizey ==> error
    #
    # matchx == True ==> determine sizex
    # matchy == True ==> determine sizey
    #
    # if sizex ==> horizontal
    # if sizey ==> vertical
    # if sizex and sizey ==> both
    #
    # at this point we fully know which type the scalebar is
    #
    # labelx is None ==> determine from size
    # labely is None ==> determine from size
    #
    # NOTE: to force label empty, use labelx = ' '
    #

    # TODO: add logic for inverted axes:
    # yinverted = ax.yaxis_inverted()
    # xinverted = ax.xaxis_inverted()

    def f(axis):
        tick_locations = axis.get_majorticklocs()
        return len(tick_locations) > 1 and (tick_locations[1] - tick_locations[0])

    if matchx and sizex:
        raise ValueError("matchx and sizex cannot both be specified")
    if matchy and sizey:
        raise ValueError("matchy and sizey cannot both be specified")

    if matchx:
        sizex = f(ax.xaxis)
    if matchy:
        sizey = f(ax.yaxis)

    if not sizex and not sizey:
        raise ValueError("sizex and sizey cannot both be zero")

    kwargs["sizex"] = sizex
    kwargs["sizey"] = sizey

    if sizex:
        sbtype = "horizontal"
        if labelx is None:
            labelx = str(sizex)
    if sizey:
        sbtype = "vertical"
        if labely is None:
            labely = str(sizey)
    if sizex and sizey:
        sbtype = "both"

    kwargs["labelx"] = labelx
    kwargs["labely"] = labely
    kwargs["ec"] = ec

    if sbtype == "both":
        # draw horizontal component:
        kwargs["labely"] = " "  # necessary to correct center alignment
        kwargs["ec"] = None  # necessary to correct possible artifact
        sbx = AnchoredScaleBar(ax.transData, xfirst=True, **kwargs)

        # draw vertical component:
        kwargs["ec"] = ec
        kwargs["labelx"] = " "
        kwargs["labely"] = labely
        sby = AnchoredScaleBar(ax.transData, xfirst=False, **kwargs)
        ax.add_artist(sbx)
        ax.add_artist(sby)
    else:
        sb = AnchoredScaleBar(ax.transData, **kwargs)
        ax.add_artist(sb)

    if hidex:
        ax.xaxis.set_visible(False)
    if hidey:
        ax.yaxis.set_visible(False)

    return ax
