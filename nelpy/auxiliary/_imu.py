# IMU (inertial motion unit) sensor data classes and methods
import copy

from .. import utils
from ..core import (
    _analogsignalarray,
    _epocharray,
)  # core._analogsignalarray import AnalogSignalArray


class IMUSensorArray(_analogsignalarray.AnalogSignalArray):
    """
    IMUSensorArray for storing and processing IMU (inertial motion unit) sensor data.

    This class extends AnalogSignalArray to provide specialized convenience functions for IMU data,
    such as complementary filtering, orientation, SI unit-awareness, and position estimation.

    Parameters
    ----------
    data : array-like, optional
        IMU sensor data.
    timestamps : array-like, optional
        Timestamps for the data samples.
    fs : float, optional
        Sampling frequency in Hz.
    step : float, optional
        Step size between samples.
    merge_sample_gap : int, optional
        Maximum gap to merge samples.
    support : IntervalArray, optional
        Support intervals for the data.
    in_memory : bool, optional
        Whether to store data in memory. Default is True.
    labels : list of str, optional
        Labels for the signals.
    empty : bool, optional
        If True, create an empty IMUSensorArray.

    Attributes
    ----------
    All attributes of AnalogSignalArray, plus any IMU-specific attributes.

    Notes
    -----
    The IMUSensorArray class is meant to make using the IMU data easier.
    It is fundamentally still an AnalogSignalArray (hence the inheritance), but
    it should have all the specialized convenience functions built in, such as
    complementary filtering, orientation, SI unit-awareness, position estimation,
    and whatnot. The labels should also be set in a meaningful way.

    It is permissible to include a kind==['magnetometer', 'accelerometer', 'gyroscope']
    attribute, and to do specialized things depending on the type. However, for some
    filtering, we actually want to combine data from all three sensors, and so it
    might make more sense to have flexible definitions for a 3dof, 6dof, 1dof, or 9dof
    sensor with understood types.
    """

    __attributes__ = []  # IMUSensorArray-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)

    def __init__(
        self,
        data=[],
        *,
        timestamps=None,
        fs=None,
        step=None,
        merge_sample_gap=0,
        support=None,
        in_memory=True,
        labels=None,
        empty=False,
    ):
        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        if isinstance(data, _analogsignalarray.AnalogSignalArray):
            self.__dict__ = copy.deepcopy(data.__dict__)
            self.__renew__()
        else:
            kwargs = {
                "data": data,
                "timestamps": timestamps,
                "fs": fs,
                "step": step,
                "merge_sample_gap": merge_sample_gap,
                "support": support,
                "in_memory": in_memory,
                "labels": labels,
            }

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
            if self.n_signals > 0:
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
complementary filtering, orientation, SI unit-awareness, position estimation,
and whatnot. The labels should also be set in a meaningful way.

I think it is permissible to include a kind==['magnetometer', 'accelerometer', 'gyroscope']
attribute, and to do specialized things depending on the type. However, for some
filtering, we actually want to combine data from all three sensors, and so it
might make more sense to have flexible definitions for a 3dof, 6dof 1dof, or 9dof
sensor with understood types?
"""
