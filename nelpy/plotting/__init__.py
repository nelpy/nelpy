"""
nelpy.plotting
==============

The nelpy.plotting sub-package provides a variety of plotting functions and tools
for visualizing data in nelpy, including raster plots, tuning curves, color utilities,
colormaps, and more. It includes convenience wrappers for matplotlib and
plotting functions that work directly with nelpy objects.

Main Features
-------------
- Plotting functions for nelpy objects (e.g., rasterplot, epochplot, imagesc)
- Color palettes and colormaps for scientific visualization
- Utilities for figure management and aesthetics
- Context and style management for publication-quality figures

Examples
--------
>>> import nelpy.plotting as npl
>>> npl.rasterplot(...)
>>> npl.plot_tuning_curves1D(...)
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
from .scalebar import add_scalebar, add_simple_scalebar
from .utils import FigureManager, savefig, suptitle

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
