"""This module contains string formatters for nelpy."""

import numpy as np

from collections import namedtuple
from math import floor

__all__ = ['BaseFormatter',
           'ArbitraryFormatter',
           'PrettyBytes',
           'PrettyInt',
           'PrettyDuration',
           'PrettySpace'
           ]

class BaseFormatter():
    """Base formatter."""

    base_unit = 'base units'

    def __init__(self, val):
        self.val = val
        self.base_unit = type(self).base_unit
        raise NotImplementedError

class ArbitraryFormatter(float):
    """Formatter for arbitrary units."""

    base_unit = 'a.u.'

    def __init__(self, val):
        self.val = val
        self.base_unit = type(self).base_unit

    def __str__(self):
        return '{:g}'.format(self.val)

    def __repr__(self):
        return '{:g}'.format(self.val)

class PrettyBytes(int):
    """Prints number of bytes in a more readable format"""

    base_unit = 'bytes'

    def __init__(self, val):
        self.val = val
        self.base_unit =type(self).base_unit

    def __str__(self):
        if self.val < 1024:
            return '{} bytes'.format(self.val)
        elif self.val < 1024**2:
            return '{:.3f} kilobytes'.format(self.val/1024)
        elif self.val < 1024**3:
            return '{:.3f} megabytes'.format(self.val/1024**2)
        elif self.val < 1024**4:
            return '{:.3f} gigabytes'.format(self.val/1024**3)

    def __repr__(self):
        return self.__str__()

class PrettyInt(int):
    """Prints integers in a more readable format"""

    base_unit = 'int'

    def __init__(self, val):
        self.val = val
        self.base_unit = type(self).base_unit

    def __str__(self):
        return '{:,}'.format(self.val)

    def __repr__(self):
        return '{:,}'.format(self.val)

class PrettyDuration(float):
    """Time duration with pretty print.

    Behaves like a float, and can always be cast to a float.
    """

    base_unit = 's'

    def __init__(self, seconds):
        self.duration = seconds
        self.base_unit = type(self).base_unit

    def __str__(self):
        return self.time_string(self.duration)

    def __repr__(self):
        return self.time_string(self.duration)

    @staticmethod
    def to_dhms(seconds):
        """convert seconds into hh:mm:ss:ms"""
        pos = seconds >= 0
        if not pos:
            seconds = -seconds
        ms = seconds % 1; ms = round(ms*10000)/10
        seconds = floor(seconds)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        d, h = divmod(h, 24)
        Time = namedtuple('Time', 'pos dd hh mm ss ms')
        time = Time(pos=pos, dd=d, hh=h, mm=m, ss=s, ms=ms)
        return time

    @staticmethod
    def time_string(seconds):
        """returns a formatted time string."""
        if np.isinf(seconds):
            return 'inf'
        pos, dd, hh, mm, ss, s = PrettyDuration.to_dhms(seconds)
        if s > 0:
            if mm == 0:
                # in this case, represent milliseconds in terms of
                # seconds (i.e. a decimal)
                sstr = str(s/1000).lstrip('0')
                if s >= 999.5:
                    ss += 1
                    s = 0
                    sstr = ""
                    # now propagate the carry:
                    if ss == 60:
                        mm += 1
                        ss = 0
                    if mm == 60:
                        hh +=1
                        mm = 0
                    if hh == 24:
                        dd += 1
                        hh = 0
            else:
                # for all other cases, milliseconds will be represented
                # as an integer
                if s >= 999.5:
                    ss += 1
                    s = 0
                    sstr = ""
                    # now propagate the carry:
                    if ss == 60:
                        mm += 1
                        ss = 0
                    if mm == 60:
                        hh +=1
                        mm = 0
                    if hh == 24:
                        dd += 1
                        hh = 0
                else:
                    sstr = ":{:03d}".format(int(s))
        else:
            sstr = ""
        if dd > 0:
            daystr = "{:01d} days ".format(dd)
        else:
            daystr = ""
        if hh > 0:
            timestr = daystr + "{:01d}:{:02d}:{:02d}{} hours".format(hh, mm, ss, sstr)
        elif mm > 0:
            timestr = daystr + "{:01d}:{:02d}{} minutes".format(mm, ss, sstr)
        elif ss > 0:
            timestr = daystr + "{:01d}{} seconds".format(ss, sstr)
        else:
            timestr = daystr +"{} milliseconds".format(s)
        if not pos:
            timestr = "-" + timestr
        return timestr

    def __add__(self, other):
        """a + b"""
        return PrettyDuration(self.duration + other)

    def __radd__(self, other):
        """b + a"""
        return self.__add__(other)

    def __sub__(self, other):
        """a - b"""
        return PrettyDuration(self.duration - other)

    def __rsub__(self, other):
        """b - a"""
        return other - self.duration

    def __mul__(self, other):
        """a * b"""
        return PrettyDuration(self.duration * other)

    def __rmul__(self, other):
        """b * a"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """a / b"""
        return PrettyDuration(self.duration / other)


class PrettySpace(float):
    """Spatial distance/position with pretty print.

    Behaves like a float, and can always be cast to a float.
    """

    base_unit = 'cm'

    def __init__(self, centimeters):
        self.value = centimeters
        self.base_unit = type(self).base_unit

    def __str__(self):
        return self.space_string(self.value)

    def __repr__(self):
        return self.space_string(self.value)

    @staticmethod
    def decompose_cm(centimeters):
        """decompose space into (km, m, cm, mm, um).
        This function is needlessly complicated, but the template
        can be useful if we wanted to work in inches, etc.
        """
        pos = centimeters >= 0
        if not pos:
            centimeters = -centimeters
        um = round(10000*((centimeters*1000)%1))/10
        mm = floor(centimeters%1*1000)
        cm = floor(centimeters)
        m, cm = divmod(cm, 100)
        km, m = divmod(m, 1000)
        Space = namedtuple('Space', 'pos km m cm mm um')
        space = Space(pos=pos, km=km, m=m, cm=cm, mm=mm, um=um)
        return space

    @staticmethod
    def decompose_cm2(centimeters):
        """decompose space into (km, m, cm, mm, um).
        This function is needlessly complicated, but the template
        can be useful if we wanted to work in inches, etc.
        """
        pos = centimeters >= 0
        if not pos:
            centimeters = -centimeters
        um = round(10000*((centimeters*1000)%1))/10
        mm = floor(centimeters%1*1000)
        cm = floor(centimeters)
        m, cm = divmod(cm, 100)
        km, m = divmod(m, 1000)
        Space = namedtuple('Space', 'pos km m cm mm um')
        space = Space(pos=pos, km=km, m=m, cm=cm, mm=mm, um=um)
        return space

    @staticmethod
    def space_string(centimeters):
        """returns a formatted space string."""
        sstr = str(centimeters)
        if np.isinf(centimeters):
            return 'inf'
#         pos, km, m, cm, mm, um = PrettySpace.decompose(centimeters)
        pos = centimeters >= 0
        if not pos:
            centimeters = -centimeters
        if centimeters > 100000:
            sstr = "{:g} km".format(centimeters/100000)
        elif centimeters > 100:
            sstr = "{:g} m".format(centimeters/100)
        elif centimeters > 1:
            sstr = "{:g} cm".format(centimeters)
        elif centimeters < 0.001:
            sstr = "{:g} um".format(centimeters*1000000)
        else:
            sstr = "{:g} mm".format(centimeters*1000)
        if not pos:
            sstr = "-" + sstr
        return sstr

    def __add__(self, other):
        """a + b"""
        return PrettySpace(self.value + other)

    def __radd__(self, other):
        """b + a"""
        return self.__add__(other)

    def __sub__(self, other):
        """a - b"""
        return PrettySpace(self.value - other)

    def __rsub__(self, other):
        """b - a"""
        return other - self.value

    def __mul__(self, other):
        """a * b"""
        return PrettySpace(self.value * other)

    def __rmul__(self, other):
        """b * a"""
        return self.__mul__(other)

    def __truediv__(self, other):
        """a / b"""
        return PrettySpace(self.value / other)
