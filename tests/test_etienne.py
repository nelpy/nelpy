from nelpy.utils import *
from nelpy.core import *
import numpy as np

class TestEvenMore:

    def test_EpochArray_merge(self):
        times = np.array([[1.0, 3.0],
                  [4.0, 8.0],
                  [12.0, 13.0],
                  [20.0, 25.0],
                  [1.0, 5.0],
                  [6.0, 7.0],
                  [15.0, 18.0],
                  [30.0, 35.0]])

        epoch = EpochArray(times)
        merged = epoch.merge()
        assert np.allclose(merged.starts, np.array([1.0, 12.0, 15.0, 20.0, 30.0]))
        assert np.allclose(merged.stops, np.array([8.0, 13.0, 18.0, 25.0, 35.0]))

    def test_add_signal1D_1(self):
        """Add a signal to an 1D AnalogSignalArray"""
        asa = AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert asa.n_signals == 2

    # def test_add_signal1D_2(self):
    #     """Add a signal to a 1D AnalogSignalArray
    #     Note: should pass on column-wise signals"""
    #     asa = AnalogSignalArray([1,2,4])
    #     asa.add_signal([3,4,5])
    #     assert np.array(asa._ydata == np.array([[1,2,4],[3,4,5]]).T).all()

    def test_add_signal1D_3(self):
        """Add a signal to a 1D AnalogSignalArray
        Note: should pass on row-wise signals"""
        asa = AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert np.array(asa._ydata == np.array([[1,2,4],[3,4,5]])).all()

    # def test_add_signal1D_4(self):
    #     """Add a signal to an 2D AnalogSignalArray
    #     Note: should pass on column-wise signals"""
    #     asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
    #     asa.add_signal([3, 4, 5])
    #     assert np.array(asa._ydata == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]]).T).all()

    def test_add_signal1D_5(self):
        """Add a signal to an 2D AnalogSignalArray
        Note: should pass on row-wise signals"""
        asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._ydata == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_AnalogSignalArray_ydata_format1(self):
        asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._ydata_rowsig == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_AnalogSignalArray_ydata_format2(self):
        asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._ydata_colsig == np.array([[1, 7, 3], [2, 8, 4], [4, 9, 5]])).all()

    def test_AnalogSignal_n_signals(self):
        asa = AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        assert asa.n_signals == 3

    def test_AnalogSignal_n_samples(self):
        asa = AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        assert asa.n_samples == 4

    def test_AnalogSignalArray_asarray(self):
        asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa.asarray().yvals == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_AnalogSignalArray_mean1(self):
        asa = AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        assert np.array(asa.mean() == np.array([ 3. ,  8.5,  4.5])).all()

    def test_AnalogSignalArray_mean2(self):
        asa = AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        asa = asa[EpochArray([[0,1.1], [1.9,3.1]])]
        assert np.array(asa.mean() == np.array([ 3. ,  8.5,  4.5])).all()

    def test_AnalogSignalArray_mean3(self):
        asa = AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        asa = asa[EpochArray([[0,1.1], [1.9,3.1]])]
        means = [seg.mean() for seg in asa]
        assert np.array(means == np.array([np.array([ 1.5,  7.5,  3.5]), np.array([ 4.5,  9.5,  5.5])])).all()

    def test_PrettyTimePrint1(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(62.3)) == '1:02:300 minutes'

    def test_PrettyTimePrint1(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(62.03)) == '1:02:030 minutes'

    def test_PrettyTimePrint2(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(4393.5)) == "1:13:13:500 hours"

    def test_PrettyTimePrint3(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(4393)) == "1:13:13 hours"

    def test_PrettyTimePrint4(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(3.5)) == "3.5 seconds"

    def test_PrettyTimePrint5(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(3.05)) == "3.05 seconds"

    def test_PrettyTimePrint6(self):
        """make sure that PrettyTimePrint prints correctly"""
        assert str(PrettyDuration(3.0)) == "3 seconds"

    def test_PrettyTimePrint7(self):
        assert str(PrettyDuration(2586.3)) == "43:06:300 minutes"

    def test_PrettyTimePrint8(self):
        assert str(PrettyDuration(2580)) == "43:00 minutes"

    def test_PrettyTimePrint9(self):
        assert str(PrettyDuration(18.4)) == "18.4 seconds"

    def test_PrettyTimePrint10(self):
        assert str(PrettyDuration(0.340)) == "340 milliseconds"

    def test_PrettyTimePrint11(self):
        assert str(PrettyDuration(.340)) == "340 milliseconds"

    def test_PrettyTimePrint12(self):
        assert str(PrettyDuration(0.027)) == "27 milliseconds"

    #TODO: add tests for adding empty signals, and adding to empty signals
