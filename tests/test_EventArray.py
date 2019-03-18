from nelpy.core import *
import numpy as np

class TestEventArray:

    def test_copy(self):

        abscissa_vals = np.array([[2, 3, 4, 5], [11, 12, 13, 14]])
        fs = 2
        series_labels = ['a', 'b']
        ea = EventArray(abscissa_vals=abscissa_vals, fs=fs, series_labels=series_labels)
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
            assert np.all(ea._data[series] == copied_ea.data[series])

class TestBinnedEventArray:

    def test_copy(self):
        abscissa_vals = np.array([[2, 3, 4, 5], [11, 12, 13, 14]])
        fs = 2
        series_labels = ['a', 'b']
        ea = EventArray(abscissa_vals=abscissa_vals, fs=fs, series_labels=series_labels)
        
        bea = BinnedEventArray(ea, ds=1)
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
        sta = SpikeTrainArray([0, 1.5, 3], fs=fs,
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

class TestBinnedSpikeTrainArray:

    def test_construct_with_sta(self):

        fs            = 1
        series_ids    = [21]
        series_labels = ['pyr']
        series_tags   = ['CA1']
        label         = 'hippocampal units'
        sta = SpikeTrainArray([0, 1.5, 3], fs=fs,
                              label=label,
                              series_ids=series_ids,
                              series_labels=series_labels,
                              series_tags=series_tags)

        ds = 0.2
        bst = BinnedSpikeTrainArray(sta, ds=ds)

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
