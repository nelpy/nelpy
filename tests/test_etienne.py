from nelpy.utils import *
from nelpy.objects import *
import numpy as np

class TestEvenMore:

    def test_add_signal1D_1(self):
        """Add a signal to an 1D AnalogSignalArray"""
        asa = AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert asa.n_signals == 2

    def test_add_signal1D_2(self):
        """Add a signal to a 1D AnalogSignalArray
        Note: should pass on column-wise signals"""
        asa = AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert np.array(asa._ydata == np.array([[1,2,4],[3,4,5]]).T).all()

    def test_add_signal1D_3(self):
        """Add a signal to a 1D AnalogSignalArray
        Note: should pass on row-wise signals"""
        asa = AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert np.array(asa._ydata == np.array([[1,2,4],[3,4,5]])).all()

    def test_add_signal1D_4(self):
        """Add a signal to an 2D AnalogSignalArray
        Note: should pass on column-wise signals"""
        asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._ydata == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]]).T).all()

    def test_add_signal1D_5(self):
        """Add a signal to an 2D AnalogSignalArray
        Note: should pass on row-wise signals"""
        asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._ydata == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_PrettyTimePrint1(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(62.3)) == '1:02:300 minutes'

    def test_PrettyTimePrint1(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(62.03)) == '1:02:030 minutes'

    def test_PrettyTimePrint2(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(nel.utils.PrettyDuration(4393.5)) == "1:13:13:500 hours"

    def test_PrettyTimePrint3(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(nel.utils.PrettyDuration(4393)) == "1:13:13 hours"

    def test_PrettyTimePrint4(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(nel.utils.PrettyDuration(3.5)) == "3.5 seconds"

    def test_PrettyTimePrint5(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(nel.utils.PrettyDuration(3.05)) == "3.05 seconds"

    def test_PrettyTimePrint6(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(nel.utils.PrettyDuration(3.0)) == "3 seconds"

    #TODO: add tests for adding empty signals, and adding to empty signals