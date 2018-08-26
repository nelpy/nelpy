import nelpy as nel
from nelpy.generalized import *
import numpy as np
from math import pi

class TestHalfOpenIntervals:

    def test_AnalogSignalArray_halfopen_1(self):
        asa = AnalogSignalArray([0,1,2,3,4,5,6])
        assert asa.n_samples == 7
        assert asa.support.duration == 7

    def test_AnalogSignalArray_halfopen_2(self):
        asa = AnalogSignalArray([0,0,0,1,1,1,2,2,2])
        epochs = nel.utils.get_run_epochs(asa, v1=2, v2=2)
        assert np.allclose(epochs.time, np.array([6, 9]))

    def test_AnalogSignalArray_halfopen_3(self):
        asa = AnalogSignalArray([0,0,0,1,1,1,2,2,2])
        epochs = nel.utils.get_run_epochs(asa, v1=1, v2=1)
        assert np.allclose(epochs.time, np.array([3, 9]))

    def test_AnalogSignalArray_halfopen_4(self):
        asa = AnalogSignalArray([0,0,0,1,1,1,2,2,2])
        epochs = nel.utils.get_inactive_epochs(asa, v1=1, v2=1)
        assert np.allclose(epochs.time, np.array([0, 6]))
