"""This module contains abscissa and ordinate objects for core nelpy objects."""
__all__ = ['Abscissa', 'Ordinate', 'AnalogSignalArrayAbscissa', 'AnalogSignalArrayOrdinate']

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
        return "Abscissa(base_unit={}, is_wrapping={})".format(self.base_unit, self.is_wrapping)


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

    def __init__(self, base_unit=None, is_linking=False, is_wrapping=False, labelstring=None):

        # TODO: add label support

        if base_unit is None:
            base_unit = ''
        if labelstring is None:
            labelstring = '{}'

        self.base_unit = base_unit
        self._labelstring = labelstring
        self.is_linking = is_linking
        self.is_wrapping = is_wrapping

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


class AnalogSignalArrayAbscissa(Abscissa):
    """Abscissa for AnalogSignalArray."""
    def __init__(self, *args, **kwargs):

        support = kwargs.get('support', generalized.EpochArray(empty=True))
        labelstring = kwargs.get('labelstring', 'time ({})')

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
