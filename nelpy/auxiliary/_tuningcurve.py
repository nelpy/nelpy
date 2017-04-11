__all__ = ['TuningCurve1D']

import copy
import numpy as np
import scipy.ndimage.filters
import warnings

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')


########################################################################
# class TuningCurve1D
########################################################################
class TuningCurve1D:
    """Tuning curves (1-dimensional) of multiple units.

    Get in BST
    Get in queriable object for external correlates

    Get in bins, binlabels
    Get in n_bins, xmin, xmax
    Get in a transform function f

    Parameters
    ----------

    Attributes
    ----------

    """

    def __init__(self, bst, extern, *, sigma=None, bw=None, n_extern=None, transform_func=None, minbgrate=None, extmin=0, extmax=1, extlabels=None, unit_ids=None, unit_labels=None, unit_tags=None, label=None):
        """

        If sigma is nonzero, then smoothing is applied.

        We always require bst and extern, and then some combination of
            (1) bin edges, transform_func*
            (2) n_extern, transform_func*
            (3) n_extern, x_min, x_max, transform_func*

            transform_func operates on extern and returns a value that TuninCurve1D can interpret. If no transform is specified, the identity operator is assumed.
        """
        # TODO: input validation

        self._bst = bst
        self._extern = extern

        if minbgrate is None:
            minbgrate = 0.01 # Hz minimum background firing rate

        if n_extern is not None:
            if extmin is not None and extmax is not None:
                self._bins = np.linspace(extmin, extmax, n_extern+1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self._unit_ids = bst.unit_ids
        self._unit_labels = bst.unit_labels
        self._unit_tags = bst.unit_tags  # no input validation yet
        self.label = label

        if transform_func is None:
            self.trans_func = self._trans_func

        # compute occupancy
        self._occupancy = self._compute_occupancy()
        # compute ratemap (in Hz)
        self._ratemap = self._compute_ratemap()
        # normalize firing rate by occupancy
        self._ratemap = self._normalize_firing_rate_by_occupancy()
        # enforce minimum background firing rate
        self._ratemap[self._ratemap < minbgrate] = minbgrate # background firing rate of 0.01 Hz

        if sigma is not None:
            if sigma > 0:
                self.smooth(sigma=sigma, bw=bw, inplace=True)

    @property
    def ratemap(self):
        return self._ratemap

    @property
    def n_bins(self):
        """(int) Number of external correlates (bins)."""
        return len(self.bins) - 1

    @property
    def occupancy(self):
        return self._occupancy

    @property
    def bins(self):
        """External correlate bins."""
        return self._bins

    def _trans_func(self, extern, at):
        """Default transform function to map extern into numerical bins"""

        _, ext = extern.asarray(at=at)

        return ext

    def _compute_occupancy(self):

        ext = self.trans_func(self._extern, at=self._bst.bin_centers)

        xmin = self.bins[0]
        xmax = self.bins[-1]
        occupancy, _ = np.histogram(ext, bins=self.bins, range=(xmin, xmax))
        # xbins = (bins + xmax/n_xbins)[:-1] # for plotting
        return occupancy

    def _compute_ratemap(self):

        ext = self.trans_func(self._extern, at=self._bst.bin_centers)

        ext_bin_idx = np.digitize(ext, self.bins)
        ratemap = np.zeros((self.n_units, self.n_bins))

        for tt, bidx in enumerate(ext_bin_idx):
            ratemap[:,bidx-1] += self._bst.data[:,tt]

        return ratemap / self._bst.ds

    def normalize(self, inplace=False):

        if not inplace:
            out = copy.deepcopy(self)
        else:
            out = self
        if self.n_units > 1:
            per_unit_max = np.max(out.ratemap, axis=1)[..., np.newaxis]
            out._ratemap = self.ratemap / np.tile(per_unit_max, (1, out.n_bins))
        else:
            per_unit_max = np.max(out.ratemap)
            out._ratemap = self.ratemap / np.tile(per_unit_max, out.n_bins)
        return out

    def _normalize_firing_rate_by_occupancy(self):
        # normalize spike counts by occupancy:
        denom = np.tile(self.occupancy, (self.n_units,1))
        denom[denom==0] = 1
        ratemap = self.ratemap / denom
        return ratemap

    @property
    def unit_ids(self):
        """Unit IDs contained in the SpikeTrain."""
        return self._unit_ids

    @unit_ids.setter
    def unit_ids(self, val):
        if len(val) != self.n_units:
            # print(len(val))
            # print(self.n_units)
            raise TypeError("unit_ids must be of length n_units")
        elif len(set(val)) < len(val):
            raise TypeError("duplicate unit_ids are not allowed")
        else:
            try:
                # cast to int:
                unit_ids = [int(id) for id in val]
            except TypeError:
                raise TypeError("unit_ids must be int-like")
        self._unit_ids = unit_ids

    @property
    def unit_labels(self):
        """Labels corresponding to units contained in the SpikeTrain."""
        if self._unit_labels is None:
            warnings.warn("unit labels have not yet been specified")
        return self._unit_labels

    @unit_labels.setter
    def unit_labels(self, val):
        if len(val) != self.n_units:
            raise TypeError("labels must be of length n_units")
        else:
            try:
                # cast to str:
                labels = [str(label) for label in val]
            except TypeError:
                raise TypeError("labels must be string-like")
        self._unit_labels = labels

    @property
    def unit_tags(self):
        """Tags corresponding to units contained in the SpikeTrain"""
        if self._unit_tags is None:
            warnings.warn("unit tags have not yet been specified")
        return self._unit_tags

    @property
    def label(self):
        """Label pertaining to the source of the spike train."""
        if self._label is None:
            warnings.warn("label has not yet been specified")
        return self._label

    @label.setter
    def label(self, val):
        if val is not None:
            try:  # cast to str:
                label = str(val)
            except TypeError:
                raise TypeError("cannot convert label to string")
        else:
            label = val
        self._label = label

    def __add__(self, other):
        out = copy.copy(self)
        out._ratemap = out.ratemap + other
        return out

    def __sub__(self, other):
        out = copy.copy(self)
        out._ratemap = out.ratemap - other
        return out

    def __mul__(self, other):
        """overloaded * operator."""
        out = copy.copy(self)
        out._ratemap = out.ratemap * other
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        """overloaded / operator."""
        out = copy.copy(self)
        out._ratemap = out.ratemap / other
        return out

    def __len__(self):
        return self.n_units

    def smooth(self, *, sigma=None, bw=None, inplace=False):
        """Smooths the tuning curve
        """
        if sigma is None:
            sigma = 0.1 # in units of extern
        if bw is None:
            bw = 4

        ds = (self.bins[-1] - self.bins[0])/self.n_bins
        sigma = sigma / ds

        if not inplace:
            out = copy.deepcopy(self)
        else:
            out = self

        if self.n_units > 1:
            out._ratemap = scipy.ndimage.filters.gaussian_filter(self.ratemap, sigma=(0,sigma), truncate=bw)
        else:
            out._ratemap = scipy.ndimage.filters.gaussian_filter(self.ratemap, sigma=sigma, truncate=bw)

        return out

    @property
    def n_units(self):
        """(int) The number of units."""
        try:
            return len(self._unit_ids)
        except TypeError: # when unit_ids is an integer
            return 1
        except AttributeError:
            return 0

    @property
    def shape(self):
        """(tuple) The shape of the TuningCurve1D ratemap."""
        if self.isempty:
            return (self.n_units, 0)
        if len(self.ratemap.shape) ==1:
            return (1, self.ratemap.shape[0])
        return self.ratemap.shape

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty TuningCurve1D" + address_str + ">"
        shapestr = " with shape (%s, %s)" % (self.shape[0], self.shape[1])
        return "<TuningCurve1D%s>%s" % (address_str, shapestr)

    @property
    def isempty(self):
        """(bool) True if TuningCurve1D is empty"""
        try:
            return len(self.ratemap) == 0
        except TypeError: #TypeError should happen if ratemap = []
            return True

    def __iter__(self):
        """TuningCurve1D iterator initialization"""
        # initialize the internal index to zero when used as iterator
        self._index = 0
        return self

    def __next__(self):
        """TuningCurve1D iterator advancer."""
        index = self._index
        if index > self.n_units - 1:
            raise StopIteration
        out = copy.copy(self)
        out._ratemap = self.ratemap[index,:]
        out._unit_ids = self.unit_ids[index]
        out._unit_labels = self.unit_labels[index]
        self._index += 1
        return out

    def __getitem__(self, *idx):
        """TuningCurve1D index access.

        Accepts integers, slices, and lists"""

        idx = [ii for ii in idx]
        if len(idx) == 1 and not isinstance(idx[0], int):
            idx = idx[0]
        if isinstance(idx, tuple):
            idx = [ii for ii in idx]

        if self.isempty:
            return self
        try:
            out = copy.copy(self)
            out._ratemap = self.ratemap[idx,:]
            out._unit_ids = (np.asanyarray(out._unit_ids)[idx]).tolist()
            out._unit_labels = (np.asanyarray(out._unit_labels)[idx]).tolist()
            return out
        except Exception:
            raise TypeError(
                'unsupported subsctipting type {}'.format(type(idx)))

#----------------------------------------------------------------------#
#======================================================================#