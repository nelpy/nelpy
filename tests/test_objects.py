import numpy as np
import nelpy as nel  # recommended import for nelpy
import nelpy.plotting as npl  # recommended import for the nelpy plotting library
from nelpy import EpochArray, SpikeTrainArray, BinnedSpikeTrainArray

# myFile = np.load('/Users/calebkemere/Documents/LFP.npz')
# lfp = myFile['dataT2C1']*0.195
# onlineRippleDetect = myFile['onlineRippleDetect']
# offlineRippleDetect = myFile['offlineRippleDetections']
# timeStampsSeconds = myFile['timeStampsSeconds']
# timeStamps = myFile['timeStamps']


class TestAnalogSignal:

    # my_lfp = AnalogSignal(lfp)
    my_lfp = [5,4,3]

    def test_analogsignalmean(self):
        """lfdglsgjlkjsdg"""
        assert self.my_lfp.mean() == np.mean(lfp)

    def test_analogsignalstd(self):
        """hdhdfh
        sdgsdg
        sdgsdg
        sdgsdg.
        """
        #my_lfp = AnalogSignal(lfp)
        assert self.my_lfp.std() == np.std(lfp)

    def test_analogsignalmax(self):
        #my_lfp = AnalogSignal(lfp)
        assert self.my_lfp.max() == np.max(lfp)

    def test_analogsignalmin(self):
        #my_lfp = AnalogSignal(lfp)
        assert self.my_lfp.min() == np.min(lfp)

def EpochArrayInitialize():
    return EpochArray(empty=True)

def test_emptyEpochArray():
    assert EpochArrayInitialize().isempty