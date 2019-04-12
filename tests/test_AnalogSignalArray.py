import nelpy as nel
import numpy as np
from math import pi

class TestRegularlySampledAnalogSignalArray:

    def test_copy(self):
        data = np.arange(100)
        fs = 10
        rsasa = nel.RegularlySampledAnalogSignalArray(data, abscissa_vals=data/fs, fs=fs)
        copied_rsasa = rsasa.copy()

        assert hex(id(copied_rsasa)) == hex(id(copied_rsasa._intervalsignalslicer.obj))

    def test_add_signal1D_1(self):
        """Add a signal to an 1D AnalogSignalArray"""
        asa = nel.AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert asa.n_signals == 2

    # def test_add_signal1D_2(self):
    #     """Add a signal to a 1D AnalogSignalArray
    #     Note: should pass on column-wise signals"""
    #     asa = AnalogSignalArray([1,2,4])
    #     asa.add_signal([3,4,5])
    #     assert np.array(asa.data == np.array([[1,2,4],[3,4,5]]).T).all()

    def test_add_signal1D_3(self):
        """Add a signal to a 1D AnalogSignalArray
        Note: should pass on row-wise signals"""
        asa = nel.AnalogSignalArray([1,2,4])
        asa.add_signal([3,4,5])
        assert np.array(asa.data == np.array([[1,2,4],[3,4,5]])).all()

    # def test_add_signal1D_4(self):
    #     """Add a signal to an 2D AnalogSignalArray
    #     Note: should pass on column-wise signals"""
    #     asa = AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
    #     asa.add_signal([3, 4, 5])
    #     assert np.array(asa.data == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]]).T).all()

    def test_add_signal1D_5(self):
        """Add a signal to an 2D AnalogSignalArray
        Note: should pass on row-wise signals"""
        asa = nel.AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa.data == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_complex_asa_1(self):
        N = 128
        theta = np.array(2*pi/N*np.arange(N))
        exp_theta = np.exp(np.array(theta)*1j)
        casa = nel.AnalogSignalArray(exp_theta)
        assert np.all(np.isclose(casa.abs.data, 1))

    def test_complex_asa_2(self):
        N = 6
        theta = np.array(2*pi/N*np.arange(N))
        exp_theta = np.exp(np.array(theta)*1j)
        casa = nel.AnalogSignalArray(exp_theta)
        expected = np.array([[ 1.0 +0.00000000e+00j, 0.5 +8.66025404e-01j,
            -0.5 +8.66025404e-01j, -1.0 +1.22464680e-16j,
            -0.5 -8.66025404e-01j,  0.5 -8.66025404e-01j]])
        assert np.all(np.isclose(casa.data, expected))

    def test_complex_asa_3(self):
        N = 6
        theta = np.array(2*pi/N*np.arange(N))
        exp_theta = np.exp(np.array(theta)*1j)
        casa = nel.AnalogSignalArray(exp_theta)
        expected = np.array([[ 0. ,  1.04719755,  2.0943951 ,  3.14159265, -2.0943951 ,
        -1.04719755]])
        assert np.all(np.isclose(casa.angle.data, expected))

    def test_asa_data_format1(self):
        asa = nel.AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._data_rowsig == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_asa_data_format2(self):
        asa = nel.AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa._data_colsig == np.array([[1, 7, 3], [2, 8, 4], [4, 9, 5]])).all()

    def test_asa_n_signals(self):
        asa = nel.AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        assert asa.n_signals == 3

    def test_asa_n_samples(self):
        asa = nel.AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        assert asa.n_samples == 4

    def test_asa_asarray(self):
        asa = nel.AnalogSignalArray([[1, 2, 4], [7, 8, 9]])
        asa.add_signal([3, 4, 5])
        assert np.array(asa.asarray().yvals == np.array([[1, 2, 4], [7, 8, 9], [3, 4, 5]])).all()

    def test_asa_mean1(self):
        asa = nel.AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        assert np.array(asa.mean() == np.array([ 3. ,  8.5,  4.5])).all()

    def test_asa_mean2(self):
        asa = nel.AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        asa = asa[nel.EpochArray([[0,1.1], [1.9,3.1]])]
        assert np.array(asa.mean() == np.array([ 3. ,  8.5,  4.5])).all()

    def test_asa_mean3(self):
        asa = nel.AnalogSignalArray([[1, 2, 4, 5], [7, 8, 9, 10]])
        asa.add_signal([3, 4, 5, 6])
        asa = asa[nel.EpochArray([[0,1.1], [1.9,3.1]])]
        means = [seg.mean() for seg in asa]
        assert np.array(means == np.array([np.array([ 1.5,  7.5,  3.5]), np.array([ 4.5,  9.5,  5.5])])).all()

class TestHalfOpenIntervals:

    def test_asa_halfopen_1(self):
        asa = nel.AnalogSignalArray([0,1,2,3,4,5,6])
        assert asa.n_samples == 7
        assert asa.support.duration == 7

    def test_asa_halfopen_2(self):
        asa = nel.AnalogSignalArray([0,0,0,1,1,1,2,2,2])
        epochs = nel.utils.get_run_epochs(asa, v1=2, v2=2)
        assert np.allclose(epochs.time, np.array([6, 9]))

    def test_asa_halfopen_3(self):
        asa = nel.AnalogSignalArray([0,0,0,1,1,1,2,2,2])
        epochs = nel.utils.get_run_epochs(asa, v1=1, v2=1)
        assert np.allclose(epochs.time, np.array([3, 9]))

    def test_asa_halfopen_4(self):
        asa = nel.AnalogSignalArray([0,0,0,1,1,1,2,2,2])
        epochs = nel.utils.get_inactive_epochs(asa, v1=1, v2=1)
        assert np.allclose(epochs.time, np.array([0, 6]))
