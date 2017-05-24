"""A nelpy session is roughly equivalent to a NEO segment.
It has a common clock.

This object is very much a work-in-progress!!!"""

__all__ = ['Session']

import copy
import warnings

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')

########################################################################
# class Session
########################################################################
class Session:
    """Nelpy session with common clock."""

    __attributes__ = ["_animal", "_label", "_st", "_extern", "_mua"]

    def __init__(self, animal=None, st=None, extern=None, mua=None, label=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        self._animal = animal
        self._extern = extern
        self._st = st
        self._mua = mua
        self.label = label

    @property
    def animal(self):
        return self._animal

    @property
    def extern(self):
        return self._extern

    @property
    def st(self):
        return self._st

    @property
    def mua(self):
        return self._mua

    @property
    def label(self):
        """Label pertaining to the source of the spike train."""
        if self._label is None:
            warnings.warn("label has not yet been specified")
        return self._label

    @label.setter
    def label(self, val):
        if val is not None:
            try:  # cast to str:
                label = str(val)
            except TypeError:
                raise TypeError("cannot convert label to string")
        else:
            label = val
        self._label = label

#----------------------------------------------------------------------#
#======================================================================#