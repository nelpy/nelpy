from nelpy.utils import *


class TestUtils:

    def test_linear_merge_1(self):
        """Merge two sorted lists"""
        merged = linear_merge([1, 2, 4], [3, 5, 6])
        assert list(merged) == [1, 2, 3, 4, 5, 6]

    def test_linear_merge_2(self):
        """Merge non-empty and empty lists"""
        merged = linear_merge([1, 2, 4], [])
        assert list(merged) == [1, 2, 4]

    def test_linear_merge_3(self):
        """Merge empty and non-empty lists"""
        merged = linear_merge([], [3, 5, 6])
        assert list(merged) == [3, 5, 6]

    def test_linear_merge_4(self):
        """Merge two unsorted lists"""
        merged = linear_merge([1, 4, 2], [3, 6, 5])
        assert list(merged) == [1, 3, 4, 2, 6, 5]

    def test_linear_merge_5(self):
        """Merge two empty lists"""
        merged = linear_merge([], [])
        assert list(merged) == []
