"""Color palettes for use with nelpy.

Some of these functions are Copyright (c) 2012-2016, Michael L. Waskom
"""

import colorsys
from itertools import cycle

import matplotlib as mpl
import numpy as np
from matplotlib.colors import to_rgb

from . import (
    colors,
    utils,  # import get_color_cycle, desaturate
)

NELPY_PALETTES = dict(
    sweet=["#00CF97", "#F05340", "#56B4E9", "#D3AA65", "#B47CC7", "#C44E52"],
    # sweet=["#00CF97", "#F05340", "#0098A9",
    #        "#6ACC65", "#4878CF", "#C44E52"],
    old=["#6bacd0", "#cfa255", "#58b0a6", "#e48065", "#5f486f", "#9a91c4"],
    new=["#6BACD0", "#D3AA65", "#00CF97", "#F05340", "#B47CC7", "#C44E52"],
)

# blue orange green red purple brown pink

# color_light="#5f486f",
#              color_dark="#355d7a",
#              color_extra="0.5",
#              color_contr1="#67a9cf",
#              color_contr2="#d6604d", # "#ef8a62",
#              color_pastel_green="#58b0a6",
#              color_pastel_blue="#6bacd0",
#              color_pastel_orange="#cfa255",
#              color_pastel_red="#e48065",

SEABORN_PALETTES = dict(
    deep=["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"],
    muted=["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"],
    pastel=["#92C6FF", "#97F0AA", "#FF9F9A", "#D0BBFF", "#FFFEA3", "#B0E0E6"],
    bright=["#003FFF", "#03ED3A", "#E8000B", "#8A2BE2", "#FFC400", "#00D7FF"],
    dark=["#001C7F", "#017517", "#8C0900", "#7600A1", "#B8860B", "#006374"],
    colorblind=["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442", "#56B4E9"],
)


class _ColorPalette(list):
    """
    Set the color palette in a with statement, otherwise be a list.

    This class extends the list of colors to be used as a context manager for temporarily setting the color palette.

    Examples
    --------
    >>> with _ColorPalette(["#FF0000", "#00FF00"]) as pal:
    ...     # palette is set within this block
    ...     pass
    """

    def __enter__(self):
        """
        Open the context and set the palette.

        Returns
        -------
        self : _ColorPalette
            The palette object itself.
        """
        from .rcmod import set_palette

        self._orig_palette = color_palette()
        set_palette(self)
        return self

    def __exit__(self, *args):
        """
        Close the context and restore the original palette.
        """
        from .rcmod import set_palette

        set_palette(self._orig_palette)

    def as_hex(self):
        """
        Return a color palette with hex codes instead of RGB values.

        Returns
        -------
        hex : _ColorPalette
            Palette with hex color codes.
        """
        hex = [mpl.colors.rgb2hex(rgb) for rgb in self]
        return _ColorPalette(hex)


def color_palette(palette=None, n_colors=None, desat=None):
    """
    Return a list of colors defining a color palette.

    Available seaborn palette names:
        deep, muted, bright, pastel, dark, colorblind

    Other options:
        hls, husl, any named matplotlib palette, list of colors

    Calling this function with ``palette=None`` will return the current
    matplotlib color cycle.

    Matplotlib palettes can be specified as reversed palettes by appending
    "_r" to the name or as dark palettes by appending "_d" to the name.
    (These options are mutually exclusive, but the resulting list of colors
    can also be reversed).

    This function can also be used in a ``with`` statement to temporarily
    set the color cycle for a plot or set of plots.

    Parameters
    ----------
    palette : None, str, or sequence, optional
        Name of palette or None to return current palette. If a sequence, input
        colors are used but possibly cycled and desaturated.
    n_colors : int, optional
        Number of colors in the palette. If ``None``, the default will depend
        on how ``palette`` is specified. Named palettes default to 6 colors,
        but grabbing the current palette or passing in a list of colors will
        not change the number of colors unless this is specified. Asking for
        more colors than exist in the palette will cause it to cycle.
    desat : float, optional
        Proportion to desaturate each color by.

    Returns
    -------
    palette : _ColorPalette
        Color palette. Behaves like a list, but can be used as a context
        manager and possesses an ``as_hex`` method to convert to hex color
        codes.

    See Also
    --------
    set_palette : Set the default color cycle for all plots.

    Examples
    --------
    Show one of the "seaborn palettes", which have the same basic order of hues
    as the default matplotlib color cycle but more attractive colors.

    .. plot::
        :context: close-figs

        >>> import seaborn as sns
        ...
        ... sns.set()
        >>> sns.palplot(sns.color_palette("muted"))

    Use discrete values from one of the built-in matplotlib colormaps.

    .. plot::
        :context: close-figs

        >>> sns.palplot(sns.color_palette("RdBu", n_colors=7))

    Make a "dark" matplotlib sequential palette variant. (This can be good
    when coloring multiple lines or points that correspond to an ordered
    variable, where you don't want the lightest lines to be invisible).

    .. plot::
        :context: close-figs

        >>> sns.palplot(sns.color_palette("Blues_d"))

    Use a categorical matplotlib palette, add some desaturation. (This can be
    good when making plots with large patches, which look best with dimmer
    colors).

    .. plot::
        :context: close-figs

        >>> sns.palplot(sns.color_palette("Set1", n_colors=8, desat=0.5))

    Use as a context manager:

    .. plot::
        :context: close-figs

        >>> import numpy as np, matplotlib.pyplot as plt
        >>> with sns.color_palette("husl", 8):
        ...     _ = plt.plot(np.c_[np.zeros(8), np.arange(8)].T)
    """
    if palette is None:
        palette = utils.get_color_cycle()
        if n_colors is None:
            n_colors = len(palette)
    elif isinstance(palette, colors.ColorGroup):
        palette = palette.colors
        if n_colors is None:
            n_colors = len(palette)
    elif not isinstance(palette, str):
        palette = palette
        if n_colors is None:
            n_colors = len(palette)
    else:
        if n_colors is None:
            n_colors = 6

        # if isinstance(palette, list):
        #     pass
        if palette in NELPY_PALETTES:
            palette = NELPY_PALETTES[palette]
        elif palette.lower() == "jet":
            raise ValueError("No.")
        elif palette in SEABORN_PALETTES:
            palette = SEABORN_PALETTES[palette]
        elif palette in dir(mpl.cm):
            palette = mpl_palette(palette, n_colors)
        elif palette[:-2] in dir(mpl.cm):
            palette = mpl_palette(palette, n_colors)
        else:
            raise ValueError("%s is not a valid palette name" % palette)

    if desat is not None:
        palette = [utils.desaturate(c, desat) for c in palette]

    # Always return as many colors as we asked for
    pal_cycle = cycle(palette)
    palette = [next(pal_cycle) for _ in range(n_colors)]

    # Always return in r, g, b tuple format
    try:
        palette = map(mpl.colors.colorConverter.to_rgb, palette)
        palette = _ColorPalette(palette)
    except ValueError:
        raise ValueError("Could not generate a palette for %s" % str(palette))

    return palette


def mpl_palette(name, n_colors=6):
    """
    Return discrete colors from a matplotlib palette.

    Parameters
    ----------
    name : str
        Name of the palette. This should be a named matplotlib colormap.
    n_colors : int
        Number of discrete colors in the palette.

    Returns
    -------
    palette : _ColorPalette
        List-like object of colors as RGB tuples.

    Notes
    -----
    This handles the qualitative colorbrewer palettes properly, although if you ask for more colors than a particular qualitative palette can provide you will get fewer than you are expecting. In contrast, asking for qualitative color brewer palettes using :func:`color_palette` will return the expected number of colors, but they will cycle.

    If you are using the IPython notebook, you can also use the function :func:`choose_colorbrewer_palette` to interactively select palettes.

    Examples
    --------
    Create a qualitative colorbrewer palette with 8 colors:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns
        ...
        ... sns.set()
        >>> sns.palplot(sns.mpl_palette("Set2", 8))

    Create a sequential colorbrewer palette:

    .. plot::
        :context: close-figs

        >>> sns.palplot(sns.mpl_palette("Blues"))

    Create a diverging palette:

    .. plot::
        :context: close-figs

        >>> sns.palplot(sns.mpl_palette("seismic", 8))

    Create a "dark" sequential palette:

    .. plot::
        :context: close-figs

        >>> sns.palplot(sns.mpl_palette("GnBu_d"))
    """
    brewer_qual_pals = {
        "Accent": 8,
        "Dark2": 8,
        "Paired": 12,
        "Pastel1": 9,
        "Pastel2": 8,
        "Set1": 9,
        "Set2": 8,
        "Set3": 12,
    }

    if name.endswith("_d"):
        pal = ["#333333"]
        pal.extend(color_palette(name.replace("_d", "_r"), 2))
        cmap = blend_palette(pal, n_colors, as_cmap=True)
    else:
        cmap = getattr(mpl.cm, name)
    if name in brewer_qual_pals:
        bins = np.linspace(0, 1, brewer_qual_pals[name])[:n_colors]
    else:
        bins = np.linspace(0, 1, n_colors + 2)[1:-1]
    palette = list(map(tuple, cmap(bins)[:, :3]))

    return _ColorPalette(palette)


def _color_to_rgb(color, input):
    """
    Convert a color to an RGB tuple, supporting multiple color spaces.

    Parameters
    ----------
    color : various
        The color to convert.
    input : {'rgb', 'hls', 'husl', 'xkcd'}
        The color space of the input.

    Returns
    -------
    rgb : tuple
        The color as an RGB tuple.
    """
    if input == "hls":
        color = colorsys.hls_to_rgb(*color)
    elif input == "husl":
        # lazy import husl here to avoid a hard dependency
        import seaborn.external.husl as husl

        color = husl.husl_to_rgb(*color)
    elif input == "xkcd":
        # lazy import xkcd_rgb here to avoid a hard dependency
        import seaborn.colors.xkcd_rgb as xkcd_rgb

        color = xkcd_rgb[color]
    return color


def dark_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    """
    Make a sequential palette that blends from dark to ``color``.

    Parameters
    ----------
    color : base color for high values
        hex, rgb-tuple, or html color name
    n_colors : int, optional
        Number of colors in the palette.
    reverse : bool, optional
        If True, reverse the direction of the blend.
    as_cmap : bool, optional
        If True, return as a matplotlib colormap instead of list.
    input : {'rgb', 'hls', 'husl', 'xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette : _ColorPalette or matplotlib colormap
        List-like object of colors as RGB tuples, or colormap object that
        can map continuous values to colors, depending on the value of the
        ``as_cmap`` parameter.
    """
    color = _color_to_rgb(color, input)
    gray = "#222222"
    colors = [color, gray] if reverse else [gray, color]
    return blend_palette(colors, n_colors, as_cmap)


def set_hls_values(color, h=None, l=None, s=None):  # noqa
    """
    Independently manipulate the h, l, or s channels of a color.

    Parameters
    ----------
    color : matplotlib color
        hex, rgb-tuple, or html color name
    h, l, s : float or None
        New values for each channel in hls space (between 0 and 1).

    Returns
    -------
    new_color : tuple
        New color code in RGB tuple representation.
    """
    # Get an RGB tuple representation
    rgb = to_rgb(color)
    vals = list(colorsys.rgb_to_hls(*rgb))
    for i, val in enumerate([h, l, s]):
        if val is not None:
            vals[i] = val

    rgb = colorsys.hls_to_rgb(*vals)
    return rgb


def light_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    """
    Make a sequential palette that blends from light to ``color``.

    Parameters
    ----------
    color : base color for high values
        Hex code, html color name, or tuple in ``input`` space.
    n_colors : int, optional
        Number of colors in the palette.
    reverse : bool, optional
        If True, reverse the direction of the blend.
    as_cmap : bool, optional
        If True, return as a matplotlib colormap instead of list.
    input : {'rgb', 'hls', 'husl', 'xkcd'}
        Color space to interpret the input color. The first three options
        apply to tuple inputs and the latter applies to string inputs.

    Returns
    -------
    palette : _ColorPalette or matplotlib colormap
        List-like object of colors as RGB tuples, or colormap object that
        can map continuous values to colors, depending on the value of the
        ``as_cmap`` parameter.
    """
    color = _color_to_rgb(color, input)
    light = set_hls_values(color, l=0.95)
    colors = [color, light] if reverse else [light, color]
    return blend_palette(colors, n_colors, as_cmap)


def _flat_palette(color, n_colors=6, reverse=False, as_cmap=False, input="rgb"):
    """
    Make a sequential palette that blends from gray to ``color``.

    Parameters
    ----------
    color : matplotlib color
        Hex, rgb-tuple, or html color name.
    n_colors : int, optional
        Number of colors in the palette.
    reverse : bool, optional
        If True, reverse the direction of the blend.
    as_cmap : bool, optional
        If True, return as a matplotlib colormap instead of list.

    Returns
    -------
    palette : _ColorPalette or matplotlib colormap
        List-like object of colors as RGB tuples, or colormap object that
        can map continuous values to colors, depending on the value of the
        ``as_cmap`` parameter.
    """
    color = _color_to_rgb(color, input)
    flat = utils.desaturate(color, 0)
    colors = [color, flat] if reverse else [flat, color]
    return blend_palette(colors, n_colors, as_cmap)


def diverging_palette(
    h_neg, h_pos, s=75, lightness=50, sep=10, n=6, center="light", as_cmap=False
):
    """
    Make a diverging palette between two HUSL colors.

    Parameters
    ----------
    h_neg, h_pos : float
        Anchor hues for negative and positive extents of the map (in [0, 359]).
    s : float, optional
        Anchor saturation for both extents of the map (in [0, 100]).
    lightness : float, optional
        Anchor lightness for both extents of the map (in [0, 100]).
    n : int, optional
        Number of colors in the palette (if not returning a cmap).
    center : {"light", "dark"}, optional
        Whether the center of the palette is light or dark.
    as_cmap : bool, optional
        If True, return a matplotlib colormap object rather than a list of colors.

    Returns
    -------
    palette : _ColorPalette or matplotlib colormap
        List-like object of colors as RGB tuples, or colormap object that
        can map continuous values to colors, depending on the value of the
        ``as_cmap`` parameter.
    """
    palfunc = dark_palette if center == "dark" else light_palette
    neg = palfunc((h_neg, s, lightness), 128 - (sep / 2), reverse=True, input="husl")
    pos = palfunc((h_pos, s, lightness), 128 - (sep / 2), input="husl")
    midpoint = dict(light=[(0.95, 0.95, 0.95, 1.0)], dark=[(0.133, 0.133, 0.133, 1.0)])[
        center
    ]
    mid = midpoint * sep
    pal = blend_palette(np.concatenate([neg, mid, pos]), n, as_cmap=as_cmap)
    return pal


def blend_palette(colors, n_colors=6, as_cmap=False, input="rgb"):
    """
    Make a palette that blends between a list of colors.

    Parameters
    ----------
    colors : sequence
        Sequence of colors in various formats interpreted by ``input``.
    n_colors : int, optional
        Number of colors in the palette.
    as_cmap : bool, optional
        If True, return as a matplotlib colormap instead of list.
    input : {'rgb', 'hls', 'husl', 'xkcd'}
        Color space to interpret the input colors.

    Returns
    -------
    palette : _ColorPalette or matplotlib colormap
        List-like object of colors as RGB tuples, or colormap object that
        can map continuous values to colors, depending on the value of the
        ``as_cmap`` parameter.
    """
    colors = [_color_to_rgb(color, input) for color in colors]
    name = "blend"
    pal = mpl.colors.LinearSegmentedColormap.from_list(name, colors)
    if not as_cmap:
        pal = _ColorPalette(pal(np.linspace(0, 1, n_colors)))
    return pal
