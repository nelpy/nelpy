import numpy as np
import pytest

from nelpy.utils import (
    PrettyBytes,
    PrettyDuration,
    PrettyInt,
    find_nearest_idx,
    find_nearest_indices,
    frange,
    is_odd,
    is_sorted,
    linear_merge,
    nextfastpower,
    nextpower,
    pairwise,
    ragged_array,
    swap_cols,
    swap_rows,
)


class TestLinearMerge:
    """Test the linear_merge function."""

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

    def test_linear_merge_with_duplicates(self):
        """Merge lists with duplicate values"""
        merged = linear_merge([1, 2, 2, 3], [2, 3, 4])
        assert list(merged) == [1, 2, 2, 2, 3, 3, 4]

    def test_linear_merge_with_floats(self):
        """Merge lists with floating point numbers"""
        merged = linear_merge([1.1, 2.2, 4.4], [3.3, 5.5, 6.6])
        assert list(merged) == [1.1, 2.2, 3.3, 4.4, 5.5, 6.6]


class TestRaggedArray:
    """Test the ragged_array function."""

    def test_ragged_array_basic(self):
        """Test basic ragged array creation"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([4, 5])
        result = ragged_array([arr1, arr2])

        assert result.dtype == object
        assert len(result) == 2
        assert np.array_equal(result[0], arr1)
        assert np.array_equal(result[1], arr2)

    def test_ragged_array_empty(self):
        """Test ragged array with empty list"""
        result = ragged_array([])
        assert len(result) == 0

    def test_ragged_array_single_element(self):
        """Test ragged array with single array"""
        arr = np.array([1, 2, 3])
        result = ragged_array([arr])

        assert len(result) == 1
        assert np.array_equal(result[0], arr)

    def test_ragged_array_different_shapes(self):
        """Test ragged array with arrays of different shapes"""
        arr1 = np.array([1, 2, 3])
        arr2 = np.array([[1, 2], [3, 4]])
        result = ragged_array([arr1, arr2])

        assert len(result) == 2
        assert np.array_equal(result[0], arr1)
        assert np.array_equal(result[1], arr2)


class TestFrange:
    """Test the frange function."""

    def test_frange_basic(self):
        """Test basic frange functionality"""
        result = list(frange(0, 1, 0.2))
        # frange uses endpoint=False, so it excludes the stop value
        expected = [0.0, 0.2, 0.4, 0.6, 0.8]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10

    def test_frange_negative_step(self):
        """Test frange with negative step"""
        result = list(frange(1, 0, -0.2))
        # frange uses endpoint=False, so it excludes the stop value
        expected = [1.0, 0.8, 0.6, 0.4, 0.2]
        assert len(result) == len(expected)
        for r, e in zip(result, expected):
            assert abs(r - e) < 1e-10

    def test_frange_empty(self):
        """Test frange that produces empty result"""
        # frange with invalid step direction should raise ValueError
        with pytest.raises(ValueError):
            list(frange(1, 0, 0.1))

    def test_frange_single_value(self):
        """Test frange with start == stop"""
        result = list(frange(1, 1, 0.1))
        # When start == stop, frange returns empty array
        assert result == []


class TestSwapCols:
    """Test the swap_cols function."""

    def test_swap_cols_basic(self):
        """Test basic column swapping"""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        swap_cols(arr, 0, 2)
        expected = np.array([[3, 2, 1], [6, 5, 4], [9, 8, 7]])
        assert np.array_equal(arr, expected)

    def test_swap_cols_adjacent(self):
        """Test swapping adjacent columns"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        swap_cols(arr, 0, 1)
        expected = np.array([[2, 1, 3], [5, 4, 6]])
        assert np.array_equal(arr, expected)

    def test_swap_cols_same_column(self):
        """Test swapping a column with itself"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        original = arr.copy()
        swap_cols(arr, 1, 1)
        assert np.array_equal(arr, original)

    def test_swap_cols_1d_array(self):
        """Test swapping columns in 1D array"""
        arr = np.array([1, 2, 3])
        swap_cols(arr, 0, 1)
        expected = np.array([2, 1, 3])
        assert np.array_equal(arr, expected)


class TestSwapRows:
    """Test the swap_rows function."""

    def test_swap_rows_basic(self):
        """Test basic row swapping"""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        swap_rows(arr, 0, 2)
        expected = np.array([[7, 8, 9], [4, 5, 6], [1, 2, 3]])
        assert np.array_equal(arr, expected)

    def test_swap_rows_adjacent(self):
        """Test swapping adjacent rows"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        swap_rows(arr, 0, 1)
        expected = np.array([[4, 5, 6], [1, 2, 3]])
        assert np.array_equal(arr, expected)

    def test_swap_rows_same_row(self):
        """Test swapping a row with itself"""
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        original = arr.copy()
        swap_rows(arr, 1, 1)
        assert np.array_equal(arr, original)

    def test_swap_rows_1d_array(self):
        """Test swapping rows in 1D array"""
        arr = np.array([1, 2, 3])
        swap_rows(arr, 0, 1)
        expected = np.array([2, 1, 3])
        assert np.array_equal(arr, expected)


class TestPairwise:
    """Test the pairwise function."""

    def test_pairwise_basic(self):
        """Test basic pairwise iteration"""
        result = list(pairwise([1, 2, 3, 4]))
        expected = [(1, 2), (2, 3), (3, 4)]
        assert result == expected

    def test_pairwise_empty(self):
        """Test pairwise with empty list"""
        result = list(pairwise([]))
        assert result == []

    def test_pairwise_single_element(self):
        """Test pairwise with single element"""
        result = list(pairwise([1]))
        assert result == []

    def test_pairwise_two_elements(self):
        """Test pairwise with two elements"""
        result = list(pairwise([1, 2]))
        assert result == [(1, 2)]

    def test_pairwise_string(self):
        """Test pairwise with string"""
        result = list(pairwise("abc"))
        assert result == [("a", "b"), ("b", "c")]


class TestIsSorted:
    """Test the is_sorted function."""

    def test_is_sorted_true(self):
        """Test with sorted array"""
        arr = np.array([1, 2, 3, 4, 5])
        assert is_sorted(arr)

    def test_is_sorted_false(self):
        """Test with unsorted array"""
        arr = np.array([1, 3, 2, 4, 5])
        assert not is_sorted(arr)

    def test_is_sorted_empty(self):
        """Test with empty array"""
        arr = np.array([])
        assert is_sorted(arr)

    def test_is_sorted_single_element(self):
        """Test with single element array"""
        arr = np.array([1])
        assert is_sorted(arr)

    def test_is_sorted_duplicates(self):
        """Test with array containing duplicates"""
        arr = np.array([1, 2, 2, 3, 4])
        assert is_sorted(arr)

    def test_is_sorted_descending(self):
        """Test with descending array"""
        arr = np.array([5, 4, 3, 2, 1])
        assert not is_sorted(arr)

    def test_is_sorted_floats(self):
        """Test with floating point array"""
        arr = np.array([1.1, 2.2, 3.3, 4.4])
        assert is_sorted(arr)


class TestIsOdd:
    """Test the is_odd function."""

    def test_is_odd_true(self):
        """Test with odd numbers"""
        assert is_odd(1)
        assert is_odd(3)
        assert is_odd(-1)
        assert is_odd(-3)

    def test_is_odd_false(self):
        """Test with even numbers"""
        assert not is_odd(0)
        assert not is_odd(2)
        assert not is_odd(-2)
        assert not is_odd(4)

    def test_is_odd_float(self):
        """Test with float inputs - should raise TypeError"""
        with pytest.raises(TypeError):
            is_odd(1.0)
        with pytest.raises(TypeError):
            is_odd(2.0)
        with pytest.raises(TypeError):
            is_odd(1.5)


class TestFindNearestIdx:
    """Test the find_nearest_idx function."""

    def test_find_nearest_idx_basic(self):
        """Test basic nearest index finding"""
        arr = np.array([1, 3, 5, 7, 9])
        assert find_nearest_idx(arr, 4) == 1  # Closest to 3
        assert find_nearest_idx(arr, 6) == 2  # Closest to 5
        assert find_nearest_idx(arr, 1) == 0  # Exact match

    def test_find_nearest_idx_exact_match(self):
        """Test with exact match"""
        arr = np.array([1, 2, 3, 4, 5])
        assert find_nearest_idx(arr, 3) == 2

    def test_find_nearest_idx_boundary(self):
        """Test with values at boundaries"""
        arr = np.array([1, 2, 3, 4, 5])
        assert find_nearest_idx(arr, 0) == 0  # Closest to 1
        assert find_nearest_idx(arr, 6) == 4  # Closest to 5

    def test_find_nearest_idx_duplicates(self):
        """Test with array containing duplicates"""
        arr = np.array([1, 2, 2, 3, 4])
        # Should return the first occurrence
        assert find_nearest_idx(arr, 2) == 1

    def test_find_nearest_idx_empty(self):
        """Test with empty array"""
        arr = np.array([])
        with pytest.raises(ValueError):
            find_nearest_idx(arr, 1)


class TestFindNearestIndices:
    """Test the find_nearest_indices function."""

    def test_find_nearest_indices_basic(self):
        """Test basic nearest indices finding"""
        arr = np.array([1, 3, 5, 7, 9])
        vals = np.array([2, 4, 6, 8])
        result = find_nearest_indices(arr, vals)
        expected = np.array([0, 1, 2, 3])  # Closest indices
        assert np.array_equal(result, expected)

    def test_find_nearest_indices_single_value(self):
        """Test with single value - should convert to array"""
        arr = np.array([1, 3, 5, 7, 9])
        result = find_nearest_indices(arr, np.array([4]))
        assert result == 1

    def test_find_nearest_indices_exact_matches(self):
        """Test with exact matches"""
        arr = np.array([1, 2, 3, 4, 5])
        vals = np.array([1, 3, 5])
        result = find_nearest_indices(arr, vals)
        expected = np.array([0, 2, 4])
        assert np.array_equal(result, expected)

    def test_find_nearest_indices_empty_values(self):
        """Test with empty values array"""
        arr = np.array([1, 2, 3, 4, 5])
        result = find_nearest_indices(arr, np.array([]))
        assert len(result) == 0


class TestNextPower:
    """Test the nextpower function."""

    def test_nextpower_basic(self):
        """Test basic next power calculation"""
        assert nextpower(5, 2) == 8  # 2^3 = 8
        assert nextpower(8, 2) == 8  # Already a power of 2
        assert nextpower(9, 2) == 16  # 2^4 = 16

    def test_nextpower_different_base(self):
        """Test with different base"""
        assert nextpower(5, 3) == 9  # 3^2 = 9
        assert nextpower(10, 3) == 27  # 3^3 = 27

    def test_nextpower_exact_power(self):
        """Test with exact power"""
        assert nextpower(4, 2) == 4  # 2^2 = 4
        assert nextpower(9, 3) == 9  # 3^2 = 9

    def test_nextpower_zero(self):
        """Test with zero input"""
        with pytest.warns(RuntimeWarning, match="divide by zero encountered in log"):
            assert (
                nextpower(0, 2) == 0
            )  # log(0) = -inf, so 2^0 = 1, but implementation returns 0

    def test_nextpower_negative(self):
        """Test with negative input - should raise ValueError"""
        with pytest.warns(RuntimeWarning, match="invalid value encountered in log"):
            with pytest.raises(ValueError):
                nextpower(-5, 2)


class TestNextFastPower:
    """Test the nextfastpower function."""

    def test_nextfastpower_basic(self):
        """Test basic next fast power calculation"""
        # These should return powers of 2, 3, 5, or 7
        result = nextfastpower(100)
        assert result >= 100
        # Check that result is a fast power (factorizable by 2, 3, 5, 7)
        factors = [2, 3, 5, 7]
        temp = result
        for factor in factors:
            while temp % factor == 0:
                temp //= factor
        assert temp == 1

    def test_nextfastpower_exact(self):
        """Test with exact fast power"""
        # 128 is 2^7, which is a fast power
        assert nextfastpower(128) == 128

    def test_nextfastpower_small(self):
        """Test with small numbers"""
        assert nextfastpower(1) == 1
        assert nextfastpower(2) == 2
        assert nextfastpower(3) == 3


class TestPrettyClasses:
    """Test the Pretty classes (PrettyDuration, PrettyInt, PrettyBytes)."""

    def test_pretty_duration_basic(self):
        """Test PrettyDuration basic functionality"""
        pd = PrettyDuration(3661)  # 1 hour, 1 minute, 1 second
        assert str(pd) == "1:01:01 hours"
        assert repr(pd) == "1:01:01 hours"

    def test_pretty_duration_zero(self):
        """Test PrettyDuration with zero"""
        pd = PrettyDuration(0)
        assert str(pd) == "0.0 milliseconds"

    def test_pretty_duration_negative(self):
        """Test PrettyDuration with negative value"""
        pd = PrettyDuration(-61)
        assert str(pd) == "-1:01 minutes"

    def test_pretty_duration_arithmetic(self):
        """Test PrettyDuration arithmetic"""
        pd1 = PrettyDuration(60)
        pd2 = PrettyDuration(30)
        assert pd1 + pd2 == 90
        assert pd1 - pd2 == 30
        assert pd1 * 2 == 120
        assert pd1 / 2 == 30

    def test_pretty_int_basic(self):
        """Test PrettyInt basic functionality"""
        pi = PrettyInt(1000)
        assert str(pi) == "1,000"
        assert repr(pi) == "1,000"

    def test_pretty_int_large(self):
        """Test PrettyInt with large numbers"""
        pi = PrettyInt(1234567)
        assert str(pi) == "1,234,567"

    def test_pretty_bytes_basic(self):
        """Test PrettyBytes basic functionality"""
        pb = PrettyBytes(1024)
        assert str(pb) == "1.000 kilobytes"
        assert repr(pb) == "1.000 kilobytes"

    def test_pretty_bytes_large(self):
        """Test PrettyBytes with large numbers"""
        pb = PrettyBytes(1024**3)  # 1 GB
        assert "gigabytes" in str(pb)
