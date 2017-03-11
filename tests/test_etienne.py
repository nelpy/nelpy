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

    #TODO: add tests for adding empty signals, and adding to empty signals