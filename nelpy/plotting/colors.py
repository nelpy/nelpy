"""colors"""

__all__ = ['sweet',
           'riceuniv',
           'lake_louise',
           'corporate',
           'cows',
           'airbnb',
           'google',
           'bcg',
           'microsoft',
           'ColorGroup']

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


_sweet = {'green':  '#00CF97',
          'red':    '#F05340',
          'blue':   '#56B4E9',
          'gold':   '#D3AA65',
          'purple': '#B47CC7',
          'maroon': '#C44E52',
          }

# see Rice University identity standards: https://staff.rice.edu/Template_RiceBrand.aspx?id=4718
_riceuniv = {'blue': '#00417B',
             'gray': '#605F64',
             }

# for Joshua Chu, who likes Lake Louise
_louise = {'tree_green':    '#334433',
           'lake_green':    '#0099aa',
           'lake_blue':     '#33aacc',
           'stone_gray':    '#bbbbbb',
           'mountain_gray': '#667788',
           'lake_cyan':     '#00bbff',
           'bright_teal':   '#11ddbb',
           'foliage_green': '#33aa66'
            }

_rainbow = {"c0":  "#fbb735",
            "c1":  "#e98931",
            "c2":  "#eb403b",
            "c3":  "#b32E37",
            "c4":  "#6c2a6a",
            "c5":  "#5c4399",
            "c6":  "#274389",
            "c7":  "#1f5ea8",
            "c8":  "#227FB0",
            "c9":  "#2ab0c5",
            "c10": "#39c0b3",
            "c11": "#33aa66",
            "c12": '#00bbff'
            }

# For Shay, of course...
_cows = {'hereford':        '#6d3129',
         'jersey':          '#d3a474',
         'brown_swiss':     '#968470',
         'charolais':       '#d9c0a6',
         'galloway':        '#191919',
         'texas_longhorn':  '#aa622d',
         'gelbvieh':        '#86322b',
         'shorthorn':       '#86322b',
         'holstein':        '#bfbaae',
         'angus':           '#2c2a38',
         'texas_aggie':     '#5c0025'
         }

# Postdocs need colors, too
_vidal = {'orange':         '#d77828',
          'red':            '#d84027',
          'dark_blue':      '#3060ad',
          'blue':           '#1c87c9',
          'teal':           '#2fbfc4'
          }

_corporate = {'facebook':   '#3b5998',
              'dropbox':    '#1081de',
              'uber1':      '#11939a',
              'uber2':      '#000000',
              'airbnb1':    '#ff5a60',
              'airbnb2':    '#d53847',
              'airbnb3':    '#484848',
              'microsoft1': '#737373',
              'microsoft2': '#f24e1f',
              'microsoft3': '#ffb901',
              'microsoft4': '#7fba00',
              'microsoft5': '#01a4ef',
              'google1':    '#4285f4',
              'google2':    '#ea4335',
              'google3':    '#fbbc05',
              'google4':    '#34a853',
              'walmart1':   '#fdbb30',
              'walmart2':   '#007bc4',
              'amazon1':    '#232f3e',
              'amazon2':    '#ff9900',
              'ibm':        '#171717',
              'merck':      '#009999',
              '_3m':        '#ff0000',
              'teradata1':  '#dfdfdf',
              'teradata2':  '#3b3b3b',
              'teradata3':  '#ec881d',
              'oracle1':    '#f0f3f5',
              'oracle2':    '#393939',
              'oracle3':    '#f20000',
              'bcg1':       '#177b57',
              'bcg2':       '#32c77f',
              'bcg3':       '#333333'}

_microsoft = dict([(key, value) for key, value in _corporate.items() if key.startswith("microsoft")])
_google = dict([(key, value) for key, value in _corporate.items() if key.startswith("google")])
_airbnb = dict([(key, value) for key, value in _corporate.items() if key.startswith("airbnb")])
_bcg = dict([(key, value) for key, value in _corporate.items() if key.startswith("bcg")])

# instantiate ColorGroups
sweet = ColorGroup(_sweet)
vidal = ColorGroup(_vidal)
riceuniv = ColorGroup(_riceuniv)
lake_louise = ColorGroup(_louise)
rainbow = ColorGroup(_rainbow)
cows = ColorGroup(_cows)
corporate = ColorGroup(_corporate)
google = ColorGroup(_google)
microsoft = ColorGroup(_microsoft)
airbnb = ColorGroup(_airbnb)
bcg = ColorGroup(_bcg)
