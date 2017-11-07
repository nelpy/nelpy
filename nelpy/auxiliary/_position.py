# position data classes and methods
import copy

from ..core import _analogsignalarray, _epocharray #core._analogsignalarray import AnalogSignalArray
from .. import utils

class PositionArray(_analogsignalarray.AnalogSignalArray):

    __attributes__ = [] # PositionArray-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)
    def __init__(self, ydata=[], *, timestamps=None, fs=None,
                 step=None, merge_sample_gap=0, support=None,
                 in_memory=True, labels=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        if isinstance(ydata, _analogsignalarray.AnalogSignalArray):
            self.__dict__ = copy.deepcopy(ydata.__dict__)
        else:
            kwargs = {"ydata": ydata,
                    "timestamps": timestamps,
                    "fs": fs,
                    "step": step,
                    "merge_sample_gap": merge_sample_gap,
                    "support": support,
                    "in_memory": in_memory,
                    "labels": labels}

            # initialize super:
            super().__init__(**kwargs)

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty PositionArray" + address_str + ">"
        if self.n_epochs > 1:
            epstr = " ({} segments)".format(self.n_epochs)
        else:
            epstr = ""
        try:
            if(self.n_signals > 0):
                nstr = " %s signals%s" % (self.n_signals, epstr)
        except IndexError:
            nstr = " 1 signal%s" % epstr
        dstr = " for a total of {}".format(utils.PrettyDuration(self.support.duration))
        return "<PositionArray%s:%s>%s" % (address_str, nstr, dstr)

    def only_pos_has_this_function(self):
        print("oh yeah!")

