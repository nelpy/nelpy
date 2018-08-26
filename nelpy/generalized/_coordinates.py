"""This module contains abscissa and ordinate objects for core nelpy objects."""
__all__ = ['Abscissa', 'Ordinate', 'AnalogSignalArrayAbscissa', 'AnalogSignalArrayOrdinate']

import numpy as np

from .. import generalized
from .. import formatters

class Abscissa():
    """An abscissa (x-axis) object for core nelpy data containers.

    Parameters
    ----------
    data : np.array
        The

    Attributes
    ----------
    data : np.array
        The
    extent : tuple
        The extent [start, stop] of the abscissa (the domain). Default is None
        or [-inf, inf]. This needs to be specified / populated when wrapping is
        requested.
    """

    def __init__(self, support=None, is_wrapping=False, labelstring=None):

        # TODO: add label support
        if support is None:
            support = generalized.IntervalArray(empty=True)
        if labelstring is None:
            labelstring = '{}'

        self.formatter = formatters.ArbitraryFormatter
        self.support = support
        self.base_unit = self.support.base_unit
        self._labelstring = labelstring
        self.is_wrapping = is_wrapping

    @property
    def label(self):
        """Abscissa label."""
        return self._labelstring.format(self.base_unit)

    @label.setter
    def label(self, val):
        if val is None:
            val = '{}'
        try:  # cast to str:
            labelstring = str(val)
        except TypeError:
            raise TypeError("cannot convert label to string")
        else:
            labelstring = val
        self._labelstring = labelstring

    def __repr__(self):
        return "Abscissa(base_unit={}, is_wrapping={}) on domain [{}, {})".format(self.base_unit, self.is_wrapping, self.domain.start, self.domain.stop)

    @property
    def domain(self):
        """Domain (in base units) on which abscissa is defined."""
        return self.support.domain

    @domain.setter
    def domain(self, val):
        """Domain (in base units) on which abscissa is defined."""
        # val can be an IntervalArray type, or (start, stop)
        self.support.domain = val
        self.support = self.support[self.support.domain]

class Ordinate():
    """An ordinate (y-axis) object for core nelpy data containers.

    Parameters
    ----------
    data : np.array
        The

    Attributes
    ----------
    data : np.array
        The
    """

    def __init__(self, base_unit=None, is_linking=False, is_wrapping=False, labelstring=None, _range=None):

        # TODO: add label support

        if base_unit is None:
            base_unit = ''
        if labelstring is None:
            labelstring = '{}'

        if _range is None:
            _range = generalized.IntervalArray([-np.inf, np.inf])

        self.base_unit = base_unit
        self._labelstring = labelstring
        self.is_linking = is_linking
        self.is_wrapping = is_wrapping
        self._is_wrapped = None # intialize to unknown (None) state
        self._range = _range

    @property
    def label(self):
        """Ordinate label."""
        return self._labelstring.format(self.base_unit)

    @label.setter
    def label(self, val):
        if val is None:
            val = '{}'
        try:  # cast to str:
            labelstring = str(val)
        except TypeError:
            raise TypeError("cannot convert label to string")
        else:
            labelstring = val
        self._labelstring = labelstring

    def __repr__(self):
        return "Ordinate(base_unit={}, is_linking={}, is_wrapping={})".format(self.base_unit, self.is_linking, self.is_wrapping)

    @property
    def range(self):
        """Range (in ordinate base units) on which ordinate is defined."""
        return self._range

    @range.setter
    def range(self, val):
        """Range (in ordinate base units) on which ordinate is defined."""
        # val can be an IntervalArray type, or (start, stop)
        if isinstance(val, type(self.range)):
                self._range = val
        elif isinstance(val, (tuple, list)):
            prev_domain = self.range.domain
            self._range = type(self.range)([val[0], val[1]])
            self._range.domain = prev_domain
        else:
            raise TypeError('range must be of type {}'.format(str(type(self.range))))

        self._range = self.range[self.range.domain]


class AnalogSignalArrayAbscissa(Abscissa):
    """Abscissa for AnalogSignalArray."""
    def __init__(self, *args, **kwargs):

        support = kwargs.get('support', generalized.EpochArray(empty=True))
        labelstring = kwargs.get('labelstring', 'time ({})') # TODO FIXME after unit inheritance; inherit from formatter?

        kwargs['support'] = support
        kwargs['labelstring'] = labelstring

        super().__init__(*args, **kwargs)

        self.formatter = self.support.formatter


class AnalogSignalArrayOrdinate(Ordinate):
    """Ordinate for AnalogSignalArray.

    Example
    -------
    ng.AnalogSignalArrayOrdinate(base_unit='uV')
    """
    def __init__(self, *args, **kwargs):

        base_unit = kwargs.get('base_unit', 'V')
        labelstring = kwargs.get('labelstring', 'voltage ({})')

        kwargs['base_unit'] = base_unit
        kwargs['labelstring'] = labelstring

        super().__init__(*args, **kwargs)
