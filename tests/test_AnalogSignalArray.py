from nelpy.core import *
import numpy as np

class TestRegularlySampledAnalogSignalArray:

    def test_copy(self):
        data = np.arange(100)
        fs = 10
        rsasa = RegularlySampledAnalogSignalArray(data, abscissa_vals=data/fs, fs=fs)
        copied_rsasa = rsasa.copy()

        assert hex(id(copied_rsasa)) == hex(id(copied_rsasa._intervalsignalslicer.obj))