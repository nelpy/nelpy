"""EpochArray tests"""

import numpy as np

import nelpy as nel


def _reference_merge(data, *, gap=0.0, overlap=0.0):
    data = np.asarray(data, dtype=float)
    if data.size == 0:
        return np.empty((0, 2))

    data = data[np.argsort(data[:, 0])]
    merged = [data[0].copy()]
    covered_stop = data[0, 1]
    for start, stop in data[1:]:
        if start <= covered_stop + gap - overlap:
            merged[-1][1] = max(merged[-1][1], stop)
            covered_stop = max(covered_stop, stop)
        else:
            merged.append(np.array([start, stop]))
            covered_stop = stop
    return np.asarray(merged)


def _reference_intersect(data_a, data_b, *, boundaries=True):
    data_a = np.asarray(data_a, dtype=float)
    data_b = np.asarray(data_b, dtype=float)
    out = []
    for a_start, a_stop in data_a[np.argsort(data_a[:, 0])]:
        for b_start, b_stop in data_b[np.argsort(data_b[:, 0])]:
            if a_start < b_stop and b_start < a_stop:
                if boundaries:
                    out.append([max(a_start, b_start), min(a_stop, b_stop)])
                else:
                    out.append([min(a_start, b_start), max(a_stop, b_stop)])
    if not out:
        return np.empty((0, 2))
    out = np.asarray(out)
    return out[np.lexsort((out[:, 1], out[:, 0]))]


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

    def test_merge_with_gap(self):
        epoch = nel.EpochArray([[0, 1], [1.5, 2], [2.4, 3]])

        merged = epoch.merge(gap=0.6)

        assert np.allclose(merged.data, np.array([[0, 3]]))

    def test_merge_with_overlap(self):
        epoch = nel.EpochArray([[0, 1], [0.8, 2], [1.9, 3]])

        merged = epoch.merge(overlap=0.2)

        assert np.allclose(merged.data, np.array([[0, 2], [1.9, 3]]))

    def test_merge_transitive_overlap(self):
        epoch = nel.EpochArray([[0, 2], [1.5, 3], [2.8, 4]])

        merged = epoch.merge()

        assert np.allclose(merged.data, np.array([[0, 4]]))

    def test_intersection_of_contiguous_epochs(self):
        """We want contiguous intervals to stay contiguous, even if intersecting"""
        x = nel.EpochArray([[2, 3], [3, 4], [5, 7]])
        y = nel.EpochArray([2, 8])

        assert x[y].n_intervals == 3
        assert y[x].n_intervals == 3

    def test_intersection_of_merged_epochs_matches_pairwise_overlap(self):
        """Fast merged-interval intersection should preserve overlap semantics."""
        x = nel.EpochArray([[0, 2], [4, 6], [8, 10]])
        y = nel.EpochArray([[1, 5], [7, 9]])

        assert np.allclose(x[y].data, np.array([[1, 2], [4, 5], [8, 9]]))
        assert np.allclose(y[x].data, np.array([[1, 2], [4, 5], [8, 9]]))

    def test_singleton_intersection_without_boundaries(self):
        x = nel.EpochArray([[0, 2], [4, 6], [8, 10]])
        y = nel.EpochArray([1, 5])

        assert np.allclose(
            x.intersect(y, boundaries=False).data,
            np.array([[0, 5], [1, 6]]),
        )
        assert np.allclose(
            y.intersect(x, boundaries=False).data,
            np.array([[0, 5], [1, 6]]),
        )

    def test_singleton_intersection_empty_overlap(self):
        x = nel.EpochArray([[0, 2], [4, 6]])
        y = nel.EpochArray([10, 12])

        assert x[y].isempty
        assert y[x].isempty

    def test_nonmerged_intersection_falls_back_to_pairwise(self):
        x = nel.EpochArray([[0, 5], [2, 7]])
        y = nel.EpochArray([[1, 3], [4, 6]])

        assert np.allclose(
            x[y].data,
            np.array([[1, 3], [2, 3], [4, 5], [4, 6]]),
        )

    def test_intersection_with_unsorted_private_data_uses_sorted_inputs(self):
        x = nel.EpochArray([[0, 2], [4, 6]])
        y = nel.EpochArray([[1, 5], [7, 8]])
        x._data = np.array([[4.0, 6.0], [0.0, 2.0]])
        original_data = x.data.copy()

        assert np.allclose(x[y].data, np.array([[1, 2], [4, 5]]))
        assert np.allclose(x.data, original_data)

    def test_intersection_sorted_inputs_skip_argsort(self, monkeypatch):
        x = nel.EpochArray([[0, 2], [4, 6]])
        y = nel.EpochArray([[1, 5], [7, 8]])

        def fail_argsort(*args, **kwargs):
            raise AssertionError("sorted intersection should not call argsort")

        monkeypatch.setattr(np, "argsort", fail_argsort)

        assert np.allclose(x[y].data, np.array([[1, 2], [4, 5]]))

    def test_merge_matches_reference_for_randomized_intervals(self):
        rng = np.random.default_rng(20260515)
        cases = [
            {"gap": 0.0, "overlap": 0.0},
            {"gap": 0.08, "overlap": 0.0},
            {"gap": 0.0, "overlap": 0.08},
        ]

        for kwargs in cases:
            for _ in range(30):
                starts = np.sort(rng.uniform(0, 10, size=24))
                lengths = rng.uniform(0.01, 0.5, size=starts.size)
                data = np.column_stack((starts, starts + lengths))
                data[1] = data[0]  # duplicate interval
                data[4] = [data[3, 1], data[3, 1] + 0.2]  # touching boundary
                data[8] = [data[7, 0] + 0.01, data[7, 1] - 0.01]  # nested
                rng.shuffle(data)

                merged = nel.EpochArray(data).merge(**kwargs)
                expected = _reference_merge(data, **kwargs)

                assert np.allclose(merged.data, expected)

    def test_intersect_matches_reference_for_randomized_intervals(self):
        rng = np.random.default_rng(20260515)

        for boundaries in (True, False):
            for _ in range(30):
                starts_a = np.sort(rng.uniform(0, 12, size=16))
                starts_b = np.sort(rng.uniform(0, 12, size=11))
                data_a = np.column_stack(
                    (starts_a, starts_a + rng.uniform(0.05, 1.2, size=starts_a.size))
                )
                data_b = np.column_stack(
                    (starts_b, starts_b + rng.uniform(0.05, 1.2, size=starts_b.size))
                )
                data_a[3] = [data_a[2, 1], data_a[2, 1] + 0.4]
                data_b[4] = [data_a[5, 0], data_a[5, 1]]

                x = nel.EpochArray(data_a).merge()
                y = nel.EpochArray(data_b).merge()
                expected = _reference_intersect(x.data, y.data, boundaries=boundaries)

                actual = x.intersect(y, boundaries=boundaries)
                reverse = y.intersect(x, boundaries=boundaries)

                assert np.allclose(actual.data, expected)
                assert np.allclose(reverse.data, expected)

    def test_intersect_singleton_and_no_overlap_match_reference(self):
        many = np.array([[0.0, 1.0], [2.0, 4.0], [4.0, 5.0], [8.0, 10.0]])
        singleton = np.array([[3.0, 8.5]])
        miss = np.array([[10.5, 11.0]])

        for boundaries in (True, False):
            expected = _reference_intersect(many, singleton, boundaries=boundaries)
            assert np.allclose(
                nel.EpochArray(many)
                .intersect(nel.EpochArray(singleton), boundaries=boundaries)
                .data,
                expected,
            )
            assert np.allclose(
                nel.EpochArray(singleton)
                .intersect(nel.EpochArray(many), boundaries=boundaries)
                .data,
                expected,
            )

        assert nel.EpochArray(many).intersect(nel.EpochArray(miss)).isempty
        assert nel.EpochArray(miss).intersect(nel.EpochArray(many)).isempty


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
