"""EpochArray tests"""
import nelpy as nel
from nelpy.core import *

class TestEpochArray:

    def test_partition(self):
        ep = EpochArray([0,10])
        partitioned = ep.partition(n_epochs=5)
        assert ep.n_epochs == 1
        assert partitioned.n_epochs == 5
