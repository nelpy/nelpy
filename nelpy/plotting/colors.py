"""colors"""

__all__ = ['sweet']

class ColorGroup():

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v

        if kwargs:
            for k, v in kwargs.items():
                self[k] = v
        for k,v in _sweet.items():
            self[k] = v

    def __getattr__(self, attr):
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
        """return the list of color names"""
        return list(self.__dict__.keys())

    @property
    def colors(self):
        """return the list of colors"""
        return list(self.__dict__.values())

    def plot(self):
        """Plot all the colors in the ColorGroup, but not necessarily in an ordered sequence"""
        npl.palplot(self.colors)

_sweet = {'green': '#00CF97',
          'red': '#F05340',
          'blue': '#56B4E9',
          'gold': '#D3AA65',
          'purple': '#B47CC7',
          'maroon': '#C44E52',
          }

sweet = ColorGroup(_sweet)
