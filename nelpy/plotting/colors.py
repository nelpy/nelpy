"""colors"""

__all__ = ['sweet',
           'riceuniv',
           'lake_louise']

import colorsys
import numpy as np

from .miscplot import palplot as _palplot

def _get_hsv(hexrgb):
    hexrgb = hexrgb.lstrip("#")   # in case you have Web color specs
    r, g, b = (int(hexrgb[i:i+2], 16) / 255.0 for i in range(0,5,2))
    return colorsys.rgb_to_hsv(r, g, b)

class ColorGroup():
    """An unordered, named color group."""

    def __init__(self, *args, label=None, **kwargs):

        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __repr__(self):
        self.plot(size=0.5)
        address_str = " at " + str(hex(id(self)))
        return "<ColorGroup" + address_str + " with " + str(self.n_colors) + " colors>"

    @property
    def n_colors(self):
        return len(self._colors)

    def __getattr__(self, attr):
        if attr not in self.__dict__:
            raise AttributeError("{} is not a valid attribute".format(str(attr)))
        return self.get(attr)

    def __setattr__(self, key, value):
        self.__setitem__(key, value)

    def __setitem__(self, key, value):
        self.__dict__.update({key: value})

    def __delattr__(self, item):
        self.__delitem__(item)

    def __delitem__(self, key):
        del self.__dict__[key]

    @property
    def color_names(self):
        """return the list of color names, ordered by hue"""
        hue_ordered_names = [ list(self.__dict__.keys())[i] for i in self._hue_order]
        return hue_ordered_names

    @property
    def _colors(self):
        """return the list of unordered colors"""
        return list(self.__dict__.values())

    @property
    def colors(self):
        """return the list of colors, ordered by hue"""
        hue_ordered_colors = [ list(self.__dict__.values())[i] for i in self._hue_order]
        return hue_ordered_colors

    @property
    def _hue_order(self):
        return sorted(range(self.n_colors), key=lambda k: _get_hsv(self._colors[k]))

    def plot(self, size=0.5):
        """Plot all the colors in the ColorGroup, but not necessarily
        in a particular order.
        """
        _palplot(self.colors, size=size)


_sweet = {'green': '#00CF97',
          'red': '#F05340',
          'blue': '#56B4E9',
          'gold': '#D3AA65',
          'purple': '#B47CC7',
          'maroon': '#C44E52',
          }

# see Rice University identity standards: https://staff.rice.edu/Template_RiceBrand.aspx?id=4718
_riceuniv = {'blue': '#00417B',
             'gray': '#605F64',
             }

# for Joshua Chu, who likes Lake Louise
_louise = {'tree_green': '#334433',
           'lake_green': '#0099aa',
           'lake_blue': '#33aacc',
           'stone_gray': '#bbbbbb',
           'mountain_gray': '#667788',
           'lake_cyan': '#00bbff',
           'bright_teal': '#11ddbb',
           'foliage_green': '#33aa66'
            }

_rainbow = {"c0": "#fbb735",
            "c1": "#e98931",
            "c2": "#eb403b",
            "c3": "#b32E37",
            "c4": "#6c2a6a",
            "c5": "#5c4399",
            "c6": "#274389",
            "c7": "#1f5ea8",
            "c8": "#227FB0",
            "c9": "#2ab0c5",
            "c10": "#39c0b3",
            "c11": "#33aa66",
            "c12": '#00bbff'
            }

_cows = {'hereford': '#6d3129',
         'jersey': '#d3a474',
         'brown_swiss': '#968470',
         'charolais': '#d9c0a6',
         'galloway': '#191919',
         'texas_longhorn': '#aa622d',
         'gelbvieh': '#86322b',
         'shorthorn': '#86322b',
         'holstein': '#bfbaae',
         'angus': '#2c2a38',
         'texas_aggie': '#5c0025'
         }

# instantiate ColorGroups
sweet = ColorGroup(_sweet)
riceuniv = ColorGroup(_riceuniv)
lake_louise = ColorGroup(_louise)
rainbow = ColorGroup(_rainbow)
cows = ColorGroup(_cows)