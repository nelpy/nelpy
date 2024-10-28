"""
nelpy.plotting
=====

This is the nelpy plotting sub-package.

nelpy.plotting provides many plot types that work directly on nelpy
objects, as well as some convenience functions to make using matplotlib
more convenient.
"""

from . import colors, utils
from .core import (
    colorline,
    epochplot,
    imagesc,
    matshow,
    overviewstrip,
    plot,
    plot2d,
    plot_tuning_curves1D,
    psdplot,
    rastercountplot,
    rasterplot,
)
from .decoding import decode_and_plot_events1D, plot_cum_error_dist, plot_posteriors

from .miscplot import palplot, stripplot, veva_scatter

from .scalebar import add_scalebar, add_simple_scalebar
from .utils import FigureManager, savefig, suptitle

from .rcmod import (
    axes_style,
    plotting_context,
    reset_defaults,
    reset_orig,
    set_context,
    set_palette,
    set_style,
    setup,
)

__all__ = [
    "plot",
    "plot2d",
    "colorline",
    "plot_tuning_curves1D",
    "psdplot",
    "overviewstrip",
    "imagesc",
    "matshow",
    "epochplot",
    "rasterplot",
    "rastercountplot",
    "decode_and_plot_events1D",
    "plot_cum_error_dist",
    "plot_posteriors",
    "palplot",
    "stripplot",
    "veva_scatter",
    "setup",
    "reset_defaults",
    "reset_orig",
    "axes_style",
    "set_style",
    "plotting_context",
    "set_context",
    "set_palette",
    "add_scalebar",
    "add_simple_scalebar",
    "colors",
    "FigureManager",
    "suptitle",
    "savefig",
    "utils",
]


# Set default aesthetics
# setup()

__version__ = "0.0.2"  # should I maintain a separate version for this?
