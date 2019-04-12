import nelpy as nel
from nelpy.utils import ragged_array
import numpy as np

class TestEventArray:

    def test_copy(self):

        abscissa_vals = np.array([[2, 3, 4, 5], [11, 12, 13, 14]])
        fs = 2
        series_labels = ['a', 'b']
        ea = nel.EventArray(abscissa_vals=abscissa_vals, fs=fs, series_labels=series_labels)
        copied_ea = ea.copy()

        # Ensure slicers are attached to copied object
        assert hex(id(copied_ea)) == hex(id(copied_ea.loc.obj))
        assert hex(id(copied_ea)) == hex(id(copied_ea.iloc.obj))

        # Verify all attributes are equal
        assert ea._fs == copied_ea._fs
        assert ea._series_ids == copied_ea._series_ids
        assert ea._series_labels == copied_ea._series_labels
        assert ea._series_tags == copied_ea._series_tags
        assert ea._label == copied_ea._label
        # not sure why this didn't work:
        # assert np.all(ea._data == copied_ea._data)
        # is it because it was an object (array of arrays)?
        # But everything works for the assertions below so I think it's ok?
        for series in range(ea.n_series):
            assert np.all(ea.data[series] == copied_ea.data[series])

class TestBinnedEventArray:

    def test_copy(self):
        abscissa_vals = np.array([[2, 3, 4, 5], [11, 12, 13, 14]])
        fs = 2
        series_labels = ['a', 'b']
        ea = nel.EventArray(abscissa_vals=abscissa_vals, fs=fs, series_labels=series_labels)
        
        bea = nel.BinnedEventArray(ea, ds=1)
        copied_bea = bea.copy()

        # Ensure slicers are attached to copied object
        assert hex(id(copied_bea)) == hex(id(copied_bea.loc.obj))
        assert hex(id(copied_bea)) == hex(id(copied_bea.iloc.obj))

        # Verify all attributes are equal
        assert bea._fs == copied_bea._fs
        assert bea._series_ids == copied_bea._series_ids
        assert bea._series_labels == copied_bea._series_labels
        assert bea._series_tags == copied_bea._series_tags
        assert bea._label == copied_bea._label
        assert bea._ds == copied_bea._ds
        assert np.all(bea._bins == copied_bea._bins)
        assert np.all(bea._data == copied_bea._data)
        assert np.all(bea._bin_centers == copied_bea._bin_centers)
        assert np.all(bea._binnedSupport == copied_bea._binnedSupport)
        assert bea._eventarray == bea._eventarray

class TestSpikeTrainArray:

    def test_construct(self):
    
        fs = 1
        series_ids    = [21]
        series_labels = ['pyr']
        series_tags   = ['CA1']
        label         = 'hippocampal units'
        sta = nel.SpikeTrainArray([0, 1.5, 3], fs=fs,
                                  label=label,
                                  series_ids=series_ids,
                                  series_labels=series_labels,
                                  series_tags=series_tags)

        # Verify STA's attributes are same as arguments
        # passed to the constructor
        assert sta.fs == fs
        assert sta.series_ids == series_ids
        assert sta.series_tags == series_tags
        assert sta.series_labels == series_labels
        assert sta.label == label

        # Verify other attributes
        assert sta.n_series == 1

    def test_indexing(self):

        sta = (nel.SpikeTrainArray([[1, 2, 3, 4, 5, 6, 7, 8, 9.5, 10, 
                                    10.5, 11.4, 15, 18, 19, 20, 21], [4, 8, 17]],
                                    support=nel.EpochArray([[0, 8], [12, 22]]),
                                    fs=1)
                                    .bin(ds=1))
        sta._desc = 'test case for sta'
        data = sta.data

        sta_indexed = sta[nel.EpochArray([[2, 8], [9, 14], [19.5, 25]]), 1]

        assert sta_indexed.n_series == 1

        # make sure original object's data didn't get mutated when indexing
        assert np.all(sta.data == data)

        # make sure metadata didn't get lost!
        assert sta_indexed._desc == sta._desc

    def test_empty(self):

        sta = nel.SpikeTrainArray([[3, 4, 5, 6, 7], [2, 4, 5]],
                                   support=nel.EpochArray([0, 8]),
                                   fs=1)

        desc = 'test case for sta'
        sta._desc = desc
        n_series = sta.n_series

        sta1 = sta.empty(inplace=False)
        sta.empty(inplace=True)

        assert sta.n_series == n_series
        assert sta._desc == desc    # ensure metadata preserved
        assert sta.isempty
        assert sta.support.isempty

        # Emptying should be consistent whether we do it
        # in place or not
        assert sta1.n_series == sta.n_series
        assert sta1._desc == sta._desc
        assert sta1.isempty
        assert sta1.support.isempty

    def test_copy_without_data(self):

        sta = nel.SpikeTrainArray([[3, 4, 5, 6, 7], [2, 4, 5]],
                            support=nel.EpochArray([0, 8]),
                            fs=1)

        desc = 'test case for sta'
        sta._desc = desc

        sta_copied = sta._copy_without_data()

        assert sta_copied.n_series == sta.n_series
        assert sta_copied._desc == sta._desc
        assert sta_copied.isempty

class TestBinnedSpikeTrainArray:

    def test_construct_with_sta(self):

        fs            = 1
        series_ids    = [21]
        series_labels = ['pyr']
        series_tags   = ['CA1']
        label         = 'hippocampal units'
        sta = nel.SpikeTrainArray([0, 1.5, 3], fs=fs,
                                  label=label,
                                  series_ids=series_ids,
                                  series_labels=series_labels,
                                  series_tags=series_tags)

        ds = 0.2
        bst = nel.BinnedSpikeTrainArray(sta, ds=ds)

        # Verify BST's attributes are same as those
        # passed to the constructor
        assert bst.ds == ds

        # Verify BST's attributes are inherited from STA
        assert bst.fs == sta.fs
        assert bst.series_ids == sta.series_ids
        assert bst.series_labels == sta.series_labels
        assert bst.series_tags == sta.series_tags
        assert bst.label == sta.label
        
        # Verify BST's eventarray's attributes are also
        # inherited from STA
        assert bst.eventarray.fs == sta.fs
        assert bst.eventarray.series_ids == sta.series_ids
        assert bst.eventarray.series_labels == sta.series_labels
        assert bst.eventarray.series_tags == sta.series_tags
        assert bst.eventarray.label == sta.label

        # Verify other attributes
        assert bst.n_series == 1

    def test_indexing(self):

        bst = (nel.SpikeTrainArray([[1, 2, 3, 4, 5, 6, 7, 8, 9.5, 10, 
                                    10.5, 11.4, 15, 18, 19, 20, 21], [4, 8, 17]],
                                    support=nel.EpochArray([[0, 8], [12, 22]]),
                                    fs=1)
                                    .bin(ds=1))
        data = bst.data

        bst._desc = 'test case for bst'

        expected_bins = np.array([ 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 20, 21, 22])
        expected_bin_centers = np.array([ 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 
                                          12.5, 13.5, 20.5, 21.5])
        expected_binned_support = np.array([[0, 5],
                                            [6, 7], 
                                            [8, 9]])

        bst_indexed = bst[nel.EpochArray([[2, 8], [9, 14], [19.5, 25]]), 1]
        
        assert bst_indexed.n_series == 1

        # binned support is an int array and should be exact. The others
        # are floats so we use np.allclose
        assert bst_indexed.binnedSupport.dtype.kind in ('i', 'u')
        assert np.all(bst_indexed.binnedSupport == expected_binned_support)
        assert np.allclose(bst_indexed.bins, expected_bins)
        assert np.allclose(bst_indexed.bin_centers, expected_bin_centers)
        
        # make sure original object's data didn't get mutated when indexing
        assert np.all(bst.data == data)

        # make sure metadata didn't get lost!
        assert bst_indexed._desc == bst._desc

    def test_empty(self):

        bst = (nel.SpikeTrainArray([[3, 4, 5, 6, 7], [2, 4, 5]],
                                   support=nel.EpochArray([0, 8]),
                                   fs=1)
                                   .bin(ds=1))

        desc = 'test case for bst'
        bst._desc = desc
        n_series = bst.n_series

        bst1 = bst.empty(inplace=False)
        bst.empty(inplace=True)

        assert bst.binnedSupport == None
        assert bst.bin_centers == None
        assert bst.bins == None
        assert bst.eventarray.isempty
        assert bst.n_series == n_series
        assert bst._desc == desc
        assert bst.support.isempty

        # Emptying should be consistent whether we do it
        # in place or not
        assert bst1.binnedSupport == bst.binnedSupport
        assert bst1.bin_centers == bst.bin_centers
        assert bst1.bins == bst.bins
        assert bst1.eventarray.isempty
        assert bst1._desc == bst._desc
        assert bst1.support.isempty

    def copy_without_data(self):

        bst = (nel.SpikeTrainArray([[3, 4, 5, 6, 7], [2, 4, 5]],
                            support=nel.EpochArray([0, 8]),
                            fs=1)
                            .bin(ds=1))
        
        desc = 'test case for bst'
        bst._desc = desc

        bst_copied = bst._copy_without_data()

        assert bst_copied.n_series == bst.n_series
        assert bst._desc == desc
        assert bst.isempty
        assert bst.eventarray.isempty

class TestSpikeTrainArrayEtienne:

    def test_1(self):
        sta = nel.SpikeTrainArray([[],[],[]])
        assert sta.n_units == 3  # failed before updates

    def test_2(self):
        sta = nel.SpikeTrainArray([[],[],[3]])
        assert sta.n_units == 3

    def test_3(self):
        sta = nel.SpikeTrainArray([[1],[2],[3]])
        assert sta.n_units == 3  # failed before updates

    def test_4(self):
        sta = nel.SpikeTrainArray([1])
        assert sta.n_units == 1

    def test_5(self):
        sta = nel.SpikeTrainArray([])
        assert sta.n_units == 1  # failed before updates

    def test_6(self):
        sta = nel.SpikeTrainArray([[]])
        assert sta.n_units == 1  # failed before updates

    def test_7(self):
        sta = nel.SpikeTrainArray(1)
        assert sta.n_units == 1

    def test_8(self):
        sta = nel.SpikeTrainArray([[1,2],[3,4]])
        assert sta.n_units == 2

    def test_9(self):
        sta = nel.SpikeTrainArray([[1,2,3]])
        assert sta.n_units == 1

    def test_10(self):
        sta = nel.SpikeTrainArray([1,2,3])
        assert sta.n_units == 1

    def test_11(self):
        sta = nel.SpikeTrainArray([[1,2,3],[]])
        assert sta.n_units == 2

    def test_12(self):
        sta = nel.SpikeTrainArray(empty=True)
        assert sta.n_units == 0

    def test_13(self):
        sta = nel.SpikeTrainArray([[[3,4],[4],[2]]])  # failed before updates
        assert sta.n_units == 3

    def test_14(self):
        sta = nel.SpikeTrainArray([[3,4],[4],[2]])
        assert sta.n_units == 3

    def test_15(self):
        sta = nel.SpikeTrainArray([[1,2,3,5,10,11,12,15], [1,2,3,5,10,11,12,15]], fs=5)
        sta = sta.partition(n_epochs=5)
        for aa, bb in zip(ragged_array([np.array([1, 2, 3]), np.array([1, 2, 3])]),  sta.iloc[0].data):
            assert np.allclose(aa, bb)

    def test_16(self):
        sta = nel.SpikeTrainArray([[1,2,3,5,10,11,12,15], [1,2,3,5,10,11,12,15]], fs=5)
        sta = sta.partition(n_epochs=5)
        for aa, bb in zip(ragged_array([np.array([5, 15]), np.array([5, 15])]),  sta.iloc[[1,4],:].data):
            assert np.allclose(aa, bb)