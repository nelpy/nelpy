from nelpy.generalized import *
import numpy as np

class TestSpikeTrainArrayEtienne:

    def test_1(self):
        sta = SpikeTrainArray([[],[],[]])
        assert sta.n_units == 3  # failed before updates

    def test_2(self):
        sta = SpikeTrainArray([[],[],[3]])
        assert sta.n_units == 3

    def test_3(self):
        sta = SpikeTrainArray([[1],[2],[3]])
        assert sta.n_units == 3  # failed before updates

    def test_4(self):
        sta = SpikeTrainArray([1])
        assert sta.n_units == 1

    def test_5(self):
        sta = SpikeTrainArray([])
        assert sta.n_units == 1  # failed before updates

    def test_6(self):
        sta = SpikeTrainArray([[]])
        assert sta.n_units == 1  # failed before updates

    def test_7(self):
        sta = SpikeTrainArray(1)
        assert sta.n_units == 1

    def test_8(self):
        sta = SpikeTrainArray([[1,2],[3,4]])
        assert sta.n_units == 2

    def test_9(self):
        sta = SpikeTrainArray([[1,2,3]])
        assert sta.n_units == 1

    def test_10(self):
        sta = SpikeTrainArray([1,2,3])
        assert sta.n_units == 1

    def test_11(self):
        sta = SpikeTrainArray([[1,2,3],[]])
        assert sta.n_units == 2

    def test_12(self):
        sta = SpikeTrainArray(empty=True)
        assert sta.n_units == 0

    def test_13(self):
        sta = SpikeTrainArray([[[3,4],[4],[2]]])  # failed before updates
        assert sta.n_units == 3

    def test_14(self):
        sta = SpikeTrainArray([[3,4],[4],[2]])
        assert sta.n_units == 3

    def test_15(self):
        sta = SpikeTrainArray([[1,2,3,5,10,11,12,15], [1,2,3,5,10,11,12,15]], fs=5)
        sta = sta.partition(n_epochs=5)
        assert np.allclose(np.array([[1, 2, 3], [1, 2, 3]]), sta.iloc[0].time)

    def test_16(self):
        sta = SpikeTrainArray([[1,2,3,5,10,11,12,15], [1,2,3,5,10,11,12,15]], fs=5)
        sta = sta.partition(n_epochs=5)
        assert np.allclose(np.array([[5, 15], [5, 15]]), sta.iloc[[1,4],:].time)
