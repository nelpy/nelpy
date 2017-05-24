# TODO: add docstring

import matplotlib as mpl

from matplotlib.offsetbox import AnchoredOffsetbox
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, *, sizex=0, sizey=0, labelx=None,
                 labely=None, loc=4, pad=0.5, borderpad=0.1, sep=2,
                 prop=None, ec='k', fc='k', fontsize=None, lw=1.5,
                 capstyle='projecting', xfirst=True, **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data
        coordinate of the give axes. A label will be drawn underneath
        (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : (int) position in containing axes
            'upper right'  : 1,
            'upper left'   : 2,
            'lower left'   : 3,
            'lower right'  : 4, default
            'right'        : 5,
            'center left'  : 6,
            'center right' : 7,
            'lower center' : 8,
            'upper center' : 9,
            'center'       : 10,
        - pad, borderpad : padding, in fraction of the legend font size
        - sep : separation between labels and bars in points.
        - ec : edgecolor of scalebar
        - lw : linewidth of scalebar
        - fontsize : font size of labels
        - fc : font color / face color of labels
        - capstyle : capstyle of bars ['round', 'butt', 'projecting']
        - **kwargs : additional arguments passed to base constructor

        adapted from https://gist.github.com/dmeliza/3251476
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker
        from matplotlib.offsetbox import HPacker, TextArea, DrawingArea
        import matplotlib.patches as mpatches

        if fontsize is None:
            fontsize = mpl.rcParams['font.size']

        bars = AuxTransformBox(transform)

        if sizex and sizey:  # both horizontal and vertical scalebar
            # hacky fix for possible misalignment errors that may occur
            #  on small figures
            if ec is None:
                lw=0
            endpt = (sizex, 0)
            art = mpatches.FancyArrowPatch(
                (0, 0),
                endpt,
                color=ec,
                linewidth=lw,
                capstyle =capstyle,
                arrowstyle = mpatches.ArrowStyle.BarAB(
                    widthA=0,
                    widthB=lw*2))
            barsx = bars
            barsx.add_artist(art)
            endpt = (0, sizey)
            art = mpatches.FancyArrowPatch(
                (0, 0),
                endpt,
                color=ec,
                linewidth=lw,
                capstyle=capstyle,
                arrowstyle = mpatches.ArrowStyle.BarAB(
                    widthA=0,
                    widthB=lw*2))
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
                    arrowstyle = mpatches.ArrowStyle.BarAB(
                        widthA=lw*2,
                        widthB=lw*2))
                bars.add_artist(art)

            if sizey:
                endpt = (0, sizey)
                art = mpatches.FancyArrowPatch(
                    (0, 0),
                    endpt,
                    color=ec,
                    linewidth=lw,
                    arrowstyle = mpatches.ArrowStyle.BarAB(
                        widthA=lw*2,
                        widthB=lw*2))
                bars.add_artist(art)

        if xfirst:
            if sizex and labelx:
                bars = VPacker(
                    children=[
                        bars,
                        TextArea(
                            labelx,
                            minimumdescent=False,
                            textprops=dict(color=fc, size=fontsize))],
                    align="center",
                    pad=pad,
                    sep=sep)
            if sizey and labely:
                bars = HPacker(
                    children=[
                        TextArea(
                            labely,
                            textprops=dict(color=fc, size=fontsize)),
                        bars],
                    align="center",
                    pad=pad,
                    sep=sep)
        else:
            if sizey and labely:
                bars = HPacker(
                    children=[
                        TextArea(
                            labely,
                            textprops=dict(color=fc, size=fontsize)),
                        bars],
                    align="center",
                    pad=pad,
                    sep=sep)
            if sizex and labelx:
                bars = VPacker(
                    children=[
                        bars,
                        TextArea(
                            labelx,
                            minimumdescent=False,
                            textprops=dict(color=fc, size=fontsize))],
                    align="center",
                    pad=pad,
                    sep=sep)

        AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=bars,
            prop=prop,
            frameon=False,
            **kwargs)

def add_simple_scalebar(text, ax=None, xy=None, length=None, orientation='v', rotation_text=None, xytext=None, **kwargs):
    if rotation_text is None:
        rotation_text = 0
    if rotation_text == 'vert' or rotation_text == 'v':
        rotation_text = 90
    if rotation_text == 'horz' or rotation_text == 'h':
        rotation_text = 0
    if orientation is None:
        orientation = 0
    if orientation == 'vert' or orientation == 'v':
        orientation = 90
    if orientation == 'horz' or orientation == 'h':
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
        ax.hlines(xy[1], xy[0], xy[0] + length, lw=2, zorder=1000 )
    else:
        ax.vlines(xy[0], xy[1], xy[1] + length, lw=2, zorder=1000 )
        xytext = (xy[0] + 3, xy[1] + length/2)
        ax.annotate(text, xy=xytext, rotation=rotation_text, va='center', zorder=1000, **kwargs)


def add_scalebar(ax, *, matchx=False, matchy=False, sizex=None,
                 sizey=None, labelx=None, labely=None, hidex=True,
                 hidey=True, ec='k', **kwargs):
    """ Add scalebars to axes
    TODO: improve documentation and standardize docstring.
    Adds a set of scale bars to *ax*, matching the size to the ticks of
    the plot and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between
                      ticks if False, size should be set using sizex and
                      sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns axis containing scalebar object
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
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

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

    kwargs['sizex'] = sizex
    kwargs['sizey'] = sizey

    if sizex:
        sbtype = 'horizontal'
        if labelx is None:
            labelx = str(sizex)
    if sizey:
        sbtype = 'vertical'
        if labely is None:
            labely = str(sizey)
    if sizex and sizey:
        sbtype = 'both'

    kwargs['labelx'] = labelx
    kwargs['labely'] = labely
    kwargs['ec'] = ec

    if sbtype == 'both':
        # draw horizontal component:
        kwargs['labely'] = ' '  # necessary to correct center alignment
        kwargs['ec'] = None  # necessary to correct possible artifact
        sbx = AnchoredScaleBar(ax.transData, xfirst=True, **kwargs)

        # draw vertical component:
        kwargs['ec'] = ec
        kwargs['labelx'] = ' '
        kwargs['labely'] = labely
        sby = AnchoredScaleBar(ax.transData, xfirst=False, **kwargs)
        ax.add_artist(sbx)
        ax.add_artist(sby)
    else:
        sb = AnchoredScaleBar(ax.transData, **kwargs)
        ax.add_artist(sb)

    if hidex: ax.xaxis.set_visible(False)
    if hidey: ax.yaxis.set_visible(False)

    return ax