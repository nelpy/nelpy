"""EpochArray tests"""

import nelpy as nel
import numpy as np


class TestEpochArray:

    def test_partition(self):
        ep = nel.EpochArray([0, 10])
        partitioned = ep.partition(n_intervals=5)
        assert ep.n_intervals == 1
        assert partitioned.n_intervals == 5

    def test_merge(self):
        times = np.array(
            [
                [1.0, 3.0],
                [4.0, 8.0],
                [12.0, 13.0],
                [20.0, 25.0],
                [1.0, 5.0],
                [6.0, 7.0],
                [15.0, 18.0],
                [30.0, 35.0],
            ]
        )

        epoch = nel.EpochArray(times)
        merged = epoch.merge()
        assert np.allclose(merged.starts, np.array([1.0, 12.0, 15.0, 20.0, 30.0]))
        assert np.allclose(merged.stops, np.array([8.0, 13.0, 18.0, 25.0, 35.0]))

    def test_intersection_of_contiguous_epochs(self):
        """We want contiguous intervals to stay contiguous, even if intersecting"""
        x = nel.EpochArray([[2, 3], [3, 4], [5, 7]])
        y = nel.EpochArray([2, 8])

        assert x[y].n_intervals == 3
        assert y[x].n_intervals == 3


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
