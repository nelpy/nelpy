<<<<<<< HEAD
#encoding : utf-8
"""This file contains the nelpy plotting functions and utilities.
 *
"""

# TODO: see https://gist.github.com/arnaldorusso/6611ff6c05e1efc2fb72
# TODO: see https://github.com/nengo/nengo/blob/master/nengo/utils/matplotlib.py

import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors as mplcolors
import matplotlib.pyplot as plt

__all__ = ["annotate", "figure_grid", "savefig"]

def annotate(ax, text, xy=(0.5, 0.5), rotation=0, va=None, **kwargs):
    """Docstring goes here."""
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
    """Extracts an extension from a filename string."""
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

    Examples
    --------
    # Force the y-axis ticks to use 1e+2 as a base exponent :
    >>> ax.yaxis.set_major_formatter(npl.FixedOrderFormatter(+2))

    # Make the x-axis ticks formatted to 0 decimal places:
    >>> from matplotlib.ticker FormatStrFormatter
    >>> ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

    # Turn off offset on x-axis:
    >>> ax.xaxis.get_major_formatter().set_useOffset(False)

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
=======
#encoding : utf-8
"""This file contains the nelpy plotting functions and utilities.
 *
"""

# TODO: see https://gist.github.com/arnaldorusso/6611ff6c05e1efc2fb72
# TODO: see https://github.com/nengo/nengo/blob/master/nengo/utils/matplotlib.py

import numpy as np
import matplotlib as mpl
from matplotlib import cm
from matplotlib import colors as mplcolors
import matplotlib.pyplot as plt

__all__ = ["annotate", "figure_grid", "savefig"]

def annotate(ax, text, xy=(0.5, 0.5), rotation=0, va=None, **kwargs):
    """Docstring goes here."""
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
    """Extracts an extension from a filename string."""
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

    Examples
    --------
    # Force the y-axis ticks to use 1e+2 as a base exponent :
    >>> ax.yaxis.set_major_formatter(npl.FixedOrderFormatter(+2))

    # Make the x-axis ticks formatted to 0 decimal places:
    >>> from matplotlib.ticker FormatStrFormatter
    >>> ax.xaxis.set_major_formatter(FormatStrFormatter('%0.0f'))

    # Turn off offset on x-axis:
    >>> ax.xaxis.get_major_formatter().set_useOffset(False)

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
>>>>>>> feature/plotting
