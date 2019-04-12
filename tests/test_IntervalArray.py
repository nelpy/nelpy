"""EpochArray tests"""
import nelpy as nel

class TestEpochArray:

    def test_partition(self):
        ep = nel.EpochArray([0,10])
        partitioned = ep.partition(n_epochs=5)
        assert ep.n_epochs == 1
        assert partitioned.n_epochs == 5


# epochs_a = nel.EpochArray([[0, 5], [5,10], [10,12], [12,16], [14,18]])
# epochs_b = nel.EpochArray([[3, 12], [15,20], [15,18]])
# epochs_c = nel.EpochArray([[3,21]])

# epochs_a[epochs_b][epochs_c].time

# array([[ 3,  5],
#        [ 5, 10],
#        [10, 12],
#        [15, 16],
#        [15, 16],
#        [15, 18],
#        [15, 18]])