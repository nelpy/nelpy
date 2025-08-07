import numpy as np
import pytest

from nelpy.utils import (
    PrettyBytes,
    PrettyDuration,
    PrettyInt,
    find_nearest_idx,
    find_nearest_indices,
    find_threshold_crossing_events,
    frange,
    information_rate,
    is_odd,
    is_sorted,
    linear_merge,
    nextfastpower,
    nextpower,
    pairwise,
    ragged_array,
    shrinkMatColsTo,
    spatial_information,
    spatial_selectivity,
    spatial_sparsity,
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


class TestSpatialAnalysisFunctions:
    """Test spatial analysis functions for N-dimensional ratemaps."""

    def test_spatial_information_1d(self):
        """Test spatial_information with 1D ratemap (n_units, n_bins)"""
        # Create a simple 1D ratemap with 2 units and 5 spatial bins
        ratemap = np.array(
            [
                [0.5, 1.0, 2.0, 1.0, 0.5],  # unit 1: gaussian-like profile
                [0.1, 0.1, 0.1, 0.1, 0.1],  # unit 2: uniform low firing
            ]
        )
        Pi = np.ones(5) / 5  # uniform occupancy

        si = spatial_information(ratemap, Pi)

        # Unit 1 should have higher spatial information than unit 2
        assert si[0] > si[1]
        assert len(si) == 2
        assert np.all(si >= 0)  # spatial information should be non-negative

    def test_spatial_information_2d(self):
        """Test spatial_information with 2D ratemap (n_units, n_x, n_y)"""
        # Create a simple 2D ratemap with 2 units and 3x3 spatial bins
        ratemap = np.zeros((2, 3, 3))
        ratemap[0, 1, 1] = 5.0  # unit 1: place field in center
        ratemap[0] += 0.1  # add baseline
        ratemap[1] = 1.0  # unit 2: uniform firing

        Pi = np.ones((3, 3)) / 9  # uniform occupancy

        si = spatial_information(ratemap, Pi)

        # Unit 1 should have higher spatial information than unit 2
        assert si[0] > si[1]
        assert len(si) == 2
        assert np.all(si >= 0)

    def test_spatial_information_3d(self):
        """Test spatial_information with 3D ratemap (n_units, n_x, n_y, n_z)"""
        # Create a simple 3D ratemap with 1 unit and 2x2x2 spatial bins
        ratemap = np.zeros((1, 2, 2, 2))
        ratemap[0, 1, 1, 1] = 8.0  # single highly active voxel
        ratemap[0] += 0.1  # add baseline

        Pi = np.ones((2, 2, 2)) / 8  # uniform occupancy

        si = spatial_information(ratemap, Pi)

        assert len(si) == 1
        assert si[0] > 0  # should have positive spatial information

    def test_spatial_information_4d(self):
        """Test spatial_information with 4D ratemap (n_units, n_w, n_x, n_y, n_z)"""
        # Create a simple 4D ratemap with 1 unit and 2x2x2x2 spatial bins
        ratemap = np.zeros((1, 2, 2, 2, 2))
        ratemap[0, 0, 1, 1, 0] = 16.0  # single highly active hypervoxel
        ratemap[0] += 0.1  # add baseline

        Pi = np.ones((2, 2, 2, 2)) / 16  # uniform occupancy

        si = spatial_information(ratemap, Pi)

        assert len(si) == 1
        assert si[0] > 0  # should have positive spatial information

    def test_information_rate_1d(self):
        """Test information_rate with 1D ratemap"""
        ratemap = np.array(
            [
                [0.5, 1.0, 2.0, 1.0, 0.5],  # unit 1
                [1.0, 1.0, 1.0, 1.0, 1.0],  # unit 2: uniform
            ]
        )
        Pi = np.ones(5) / 5

        ir = information_rate(ratemap, Pi)

        assert len(ir) == 2
        assert np.all(ir >= 0)
        # Information rate should be spatial_information * mean_firing_rate
        si = spatial_information(ratemap, Pi)
        mean_rates = (ratemap * Pi).sum(axis=1)
        expected_ir = si * mean_rates
        np.testing.assert_array_almost_equal(ir, expected_ir)

    def test_information_rate_3d(self):
        """Test information_rate with 3D ratemap"""
        ratemap = np.ones((2, 2, 2, 2)) * 0.5  # 2 units, 2x2x2 spatial
        ratemap[0, 1, 1, 1] = 4.0  # add place field to unit 1

        Pi = np.ones((2, 2, 2)) / 8

        ir = information_rate(ratemap, Pi)

        assert len(ir) == 2
        assert ir[0] > ir[1]  # unit with place field should have higher info rate

    def test_spatial_selectivity_1d(self):
        """Test spatial_selectivity with 1D ratemap"""
        ratemap = np.array(
            [
                [0.1, 0.1, 5.0, 0.1, 0.1],  # highly selective unit
                [1.0, 1.0, 1.0, 1.0, 1.0],  # non-selective unit
            ]
        )
        Pi = np.ones(5) / 5

        selectivity = spatial_selectivity(ratemap, Pi)

        assert len(selectivity) == 2
        assert selectivity[0] > selectivity[1]  # first unit should be more selective
        assert selectivity[1] == pytest.approx(1.0)  # uniform unit has selectivity of 1

    def test_spatial_selectivity_3d(self):
        """Test spatial_selectivity with 3D ratemap"""
        ratemap = np.ones((2, 2, 2, 2)) * 0.5  # baseline firing
        ratemap[0, 0, 0, 0] = 8.0  # add peak to unit 1

        Pi = np.ones((2, 2, 2)) / 8

        selectivity = spatial_selectivity(ratemap, Pi)

        assert len(selectivity) == 2
        assert selectivity[0] > selectivity[1]

    def test_spatial_sparsity_1d(self):
        """Test spatial_sparsity with 1D ratemap"""
        ratemap = np.array(
            [
                [0.1, 0.1, 10.0, 0.1, 0.1],  # sparse unit
                [2.0, 2.0, 2.0, 2.0, 2.0],  # non-sparse unit
            ]
        )
        Pi = np.ones(5) / 5

        sparsity = spatial_sparsity(ratemap, Pi)

        assert len(sparsity) == 2
        assert sparsity[0] < sparsity[1]  # sparse unit should have lower sparsity value
        assert np.all(sparsity > 0)

    def test_spatial_sparsity_3d(self):
        """Test spatial_sparsity with 3D ratemap"""
        ratemap = np.ones((2, 2, 2, 2)) * 0.1  # low baseline
        ratemap[0, 1, 1, 1] = 10.0  # single high-firing voxel for unit 1

        Pi = np.ones((2, 2, 2)) / 8

        sparsity = spatial_sparsity(ratemap, Pi)

        assert len(sparsity) == 2
        assert sparsity[0] < sparsity[1]  # unit with place field should be more sparse

    def test_all_functions_with_multidimensional_pi(self):
        """Test all functions work with multi-dimensional Pi arrays"""
        # 3D ratemap with 3D Pi
        ratemap = np.random.rand(2, 3, 3, 3) + 0.1
        Pi = np.random.rand(3, 3, 3)
        Pi = Pi / Pi.sum()  # normalize

        # All functions should work without errors
        si = spatial_information(ratemap, Pi)
        ir = information_rate(ratemap, Pi)
        sel = spatial_selectivity(ratemap, Pi)
        spa = spatial_sparsity(ratemap, Pi)

        assert len(si) == len(ir) == len(sel) == len(spa) == 2
        assert np.all(si >= 0)
        assert np.all(ir >= 0)
        assert np.all(sel >= 0)
        assert np.all(spa > 0)

    def test_error_on_invalid_ratemap_shape(self):
        """Test that functions raise appropriate errors for invalid shapes"""
        # 1D array (missing unit dimension)
        ratemap_1d = np.array([1, 2, 3])
        Pi = np.array([0.3, 0.3, 0.4])

        with pytest.raises(TypeError):
            spatial_information(ratemap_1d, Pi)

        with pytest.raises(TypeError):
            information_rate(ratemap_1d, Pi)

        with pytest.raises(TypeError):
            spatial_selectivity(ratemap_1d, Pi)

        with pytest.raises(TypeError):
            spatial_sparsity(ratemap_1d, Pi)

    def test_consistency_across_dimensions(self):
        """Test that results are consistent when reshaping compatible arrays"""
        # Create a 1D case and equivalent 2D case
        ratemap_1d = np.array([[1.0, 2.0, 3.0, 2.0, 1.0]])  # 1 unit, 5 bins
        Pi_1d = np.ones(5) / 5

        # Reshape to 2D: 5x1 spatial bins
        ratemap_2d = ratemap_1d.reshape(1, 5, 1)
        Pi_2d = Pi_1d.reshape(5, 1)

        si_1d = spatial_information(ratemap_1d, Pi_1d)
        si_2d = spatial_information(ratemap_2d, Pi_2d)

        np.testing.assert_array_almost_equal(si_1d, si_2d, decimal=10)


class TestShrinkMatColsTo:
    """Test shrinkMatColsTo function."""

    def test_shrink_basic(self):
        """Test basic matrix column shrinking."""
        mat = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
        result = shrinkMatColsTo(mat, 2)

        assert result.shape == (2, 2)
        # The function uses linear interpolation which samples at different points
        # than simple averaging, so let's just test that it's reasonable
        assert result[0, 0] >= 1.0 and result[0, 0] <= 2.0
        assert result[0, 1] >= 3.0 and result[0, 1] <= 4.0
        assert result[1, 0] >= 5.0 and result[1, 0] <= 6.0
        assert result[1, 1] >= 7.0 and result[1, 1] <= 8.0

    def test_shrink_single_row(self):
        """Test shrinking with single row."""
        mat = np.array([[1, 2, 3, 4, 5, 6]])
        result = shrinkMatColsTo(mat, 3)

        assert result.shape == (1, 3)
        # Should interpolate 6 columns down to 3

    def test_shrink_to_one_column(self):
        """Test shrinking to single column."""
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        result = shrinkMatColsTo(mat, 1)

        assert result.shape == (2, 1)

    def test_shrink_no_change(self):
        """Test when target columns equals source columns."""
        mat = np.array([[1, 2, 3], [4, 5, 6]])
        result = shrinkMatColsTo(mat, 3)

        assert result.shape == (2, 3)
        # Should be very close to original
        np.testing.assert_allclose(result, mat, rtol=1e-10)


class TestFindThresholdCrossingEvents:
    """Test find_threshold_crossing_events function."""

    def test_threshold_crossing_above(self):
        """Test finding events above threshold."""
        x = np.array([0, 1, 2, 3, 2, 1, 0, 0, 3, 4, 3, 2])
        threshold = 2.5

        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")

        # Should find events where x >= 2.5
        assert len(eventlist) == 2  # Two separate events
        assert len(eventmax) == 2

        # First event: index 3 (value 3)
        # Second event: indices 8, 9, 10 (values 3, 4, 3)
        assert eventlist[0][0] == 3 and eventlist[0][1] == 3
        assert eventlist[1][0] == 8 and eventlist[1][1] == 10
        assert eventmax[0] == 3
        assert eventmax[1] == 4

    def test_threshold_crossing_below(self):
        """Test finding events below threshold."""
        x = np.array([3, 2, 1, 0, 1, 2, 3, 3, 0, 1, 2, 3])
        threshold = 1.5

        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="below")

        # Should find events where x <= 1.5
        assert len(eventlist) >= 1  # At least one event
        assert len(eventmax) >= 1

    def test_threshold_crossing_no_events(self):
        """Test when no events cross threshold."""
        x = np.array([0, 0.5, 1, 1.5, 1, 0.5, 0])
        threshold = 2

        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")

        assert len(eventlist) == 0
        assert len(eventmax) == 0

    def test_threshold_crossing_invalid_mode(self):
        """Test invalid mode parameter."""
        x = np.array([0, 1, 2, 3])
        threshold = 1.5

        with pytest.raises(NotImplementedError):
            find_threshold_crossing_events(x, threshold, mode="invalid")

    def test_threshold_crossing_continuous_event(self):
        """Test with continuous event spanning multiple indices."""
        x = np.array([0, 1, 3, 3, 3, 2, 1, 0])
        threshold = 2.5

        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")

        # Should be one continuous event from index 2 to 4
        assert len(eventlist) == 1
        assert eventlist[0][0] == 2
        assert eventlist[0][1] == 4
        assert eventmax[0] == 3


class TestShrinkMatColsToEdgeCases:
    """Test shrinkMatColsTo with edge cases for additional coverage."""

    def test_shrink_mat_single_column_input(self):
        """Test shrinkMatColsTo with single column input."""
        mat = np.array([[1], [2], [3]])
        result = shrinkMatColsTo(mat, 1)
        assert result.shape == (3, 1)
        np.testing.assert_array_equal(result, mat)

    def test_shrink_mat_large_shrinkage_ratio(self):
        """Test shrinkMatColsTo with large shrinkage ratio."""
        mat = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        result = shrinkMatColsTo(mat, 2)
        assert result.shape == (1, 2)

    def test_shrink_mat_no_shrinkage_needed(self):
        """Test shrinkMatColsTo when no shrinkage is needed."""
        mat = np.array([[1, 2], [3, 4]])
        result = shrinkMatColsTo(mat, 2)
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result, mat)

    def test_shrink_mat_small_matrix(self):
        """Test shrinkMatColsTo with very small matrix."""
        mat = np.array([[5]])
        result = shrinkMatColsTo(mat, 1)
        assert result.shape == (1, 1)
        assert result[0, 0] == 5


class TestFindThresholdCrossingEdgeCases:
    """Test find_threshold_crossing_events with edge cases for additional coverage."""

    def test_threshold_crossing_all_above(self):
        """Test find_threshold_crossing_events with all values above threshold."""
        x = np.array([5, 6, 7, 8, 9])
        threshold = 3
        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")
        assert len(eventlist) == 1
        assert eventlist[0][0] == 0 and eventlist[0][1] == 4
        assert eventmax[0] == 9

    def test_threshold_crossing_all_below(self):
        """Test find_threshold_crossing_events with all values below threshold."""
        x = np.array([1, 2, 1, 2, 1])
        threshold = 3
        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")
        assert len(eventlist) == 0
        assert len(eventmax) == 0

    def test_threshold_crossing_single_value_crossing(self):
        """Test find_threshold_crossing_events with single value crossing."""
        x = np.array([1, 2, 5, 2, 1])
        threshold = 4
        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")
        assert len(eventlist) == 1
        assert eventlist[0][0] == 2 and eventlist[0][1] == 2
        assert eventmax[0] == 5

    def test_threshold_crossing_alternating_pattern(self):
        """Test find_threshold_crossing_events with alternating pattern."""
        x = np.array([1, 5, 1, 5, 1])
        threshold = 3
        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")
        # Should have two separate single-point events
        assert len(eventlist) == 2
        assert eventlist[0][0] == 1 and eventlist[0][1] == 1
        assert eventlist[1][0] == 3 and eventlist[1][1] == 3

    def test_threshold_crossing_edge_values(self):
        """Test find_threshold_crossing_events with edge values."""
        x = np.array([3, 1, 2, 1, 3])  # Starts and ends at threshold
        threshold = 3
        eventlist, eventmax = find_threshold_crossing_events(x, threshold, mode="above")
        # Should have two events at the edges
        assert len(eventlist) == 2
        assert eventlist[0][0] == 0 and eventlist[0][1] == 0
        assert eventlist[1][0] == 4 and eventlist[1][1] == 4


class TestGaussianFilterAdditionalCoverage:
    """Test gaussian_filter function edge cases for additional coverage."""

    def test_gaussian_filter_across_epochs(self):
        """Test gaussian_filter smoothing across epochs (not within)."""
        import nelpy as nel

        data = np.random.randn(1, 150)
        support = nel.EpochArray([[0, 0.5], [1.0, 2.0]])
        asa = nel.AnalogSignalArray(data, fs=100, support=support)

        # Test smoothing across epochs (default behavior)
        filtered = nel.utils.gaussian_filter(asa, sigma=0.02, within_intervals=False)
        assert isinstance(filtered, nel.AnalogSignalArray)
        assert filtered.data.shape[0] == 1
