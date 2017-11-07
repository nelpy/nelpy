# IMU (inertial motion unit) sensor data classes and methods
import copy

from ..core import _analogsignalarray, _epocharray #core._analogsignalarray import AnalogSignalArray
from .. import utils

class IMUSensorArray(_analogsignalarray.AnalogSignalArray):

    __attributes__ = [] # IMUSensorArray-specific attributes
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
            return "<empty IMUSensorArray" + address_str + ">"
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
        return "<IMUSensorArray%s:%s>%s" % (address_str, nstr, dstr)

    def only_imu_has_this_function(self):
        print("oh yeah!")


"""
Notes:
------
The IMUSensorArray class is meant to make using the IMU data easier.
It is fundamentally still an AnalogSignalArray (hence the inheritance), but
it should have all the specialized convenience functions built in, such as
complementary filtering, orientation, unit-awareness, position estimation, and
whatnot. The labels should also be set in a meaningful way.

I think it is permissible to include a kind==['magnetometer', 'accelerometer', 'gyroscope']
attribute, and to do specialized things depending on the type. However, for some
filtering, we actually want to combine data from all three sensors, and so it
might make more sense to have flexible definitions for a 3dof, 6dof 1dof, or 9dof
sensor with understood types?
"""