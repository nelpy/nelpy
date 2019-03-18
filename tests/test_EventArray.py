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
