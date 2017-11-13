__all__ = ['TuningCurve1D', 'TuningCurve2D', 'DirectionalTuningCurve1D']

import copy
import numpy as np
import numbers
import scipy.ndimage.filters
import warnings

from .. import utils

# TODO: TuningCurve2D
# 1. init from rate map
# 1. magic functions
# 1. iterator
# 1. mean, max, min, etc.
# 1. unit_subsets
# 1. plotting support

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, \
    line=None: formatwarning_orig(
        message, category, filename, lineno, line='')


########################################################################
# class TuningCurve2D
########################################################################
class TuningCurve2D:
    """Tuning curves (2-dimensional) of multiple units.

    Parameters
    ----------

    Attributes
    ----------

    """

    __attributes__ = ["_ratemap", "_occupancy",  "_unit_ids", "_unit_labels", "_unit_tags", "_label"]

    def __init__(self, *, bst=None, extern=None, ratemap=None, sigma=None,
                 bw=None, ext_nx=None, ext_ny=None, transform_func=None,
                 minbgrate=None, ext_xmin=0, ext_ymin=0, ext_xmax=1, ext_ymax=1,
                 extlabels=None, min_duration=None, unit_ids=None,
                 unit_labels=None, unit_tags=None, label=None, empty=False):
        """

        NOTE: tuning curves in 2D have shapes (n_units, ny, nx) so that
        we can plot them in an intuitive manner

        If sigma is nonzero, then smoothing is applied.

        We always require bst and extern, and then some combination of
            (1) bin edges, transform_func*
            (2) n_extern, transform_func*
            (3) n_extern, x_min, x_max, transform_func*

            transform_func operates on extern and returns a value that TuninCurve1D can interpret. If no transform is specified, the identity operator is assumed.
        """
        # TODO: input validation
        if not empty:
            if ratemap is None:
                assert bst is not None, "bst must be specified or ratemap must be specified!"
                assert extern is not None, "extern must be specified or ratemap must be specified!"
            else:
                assert bst is None, "ratemap and bst cannot both be specified!"
                assert extern is None, "ratemap and extern cannot both be specified!"

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        if ratemap is not None:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._init_from_ratemap(ratemap=ratemap,
                                    ext_xmin=ext_xmin,
                                    ext_xmax=ext_xmax,
                                    ext_ymin=ext_ymin,
                                    ext_ymax=ext_ymax,
                                    extlabels=extlabels,
                                    unit_ids=unit_ids,
                                    unit_labels=unit_labels,
                                    unit_tags=unit_tags,
                                    label=label)
            return

        self._bst = bst
        self._extern = extern

        if minbgrate is None:
            minbgrate = 0.01 # Hz minimum background firing rate

        if ext_nx is not None:
            if ext_xmin is not None and ext_xmax is not None:
                self._xbins = np.linspace(ext_xmin, ext_xmax, ext_nx+1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if ext_ny is not None:
            if ext_ymin is not None and ext_ymax is not None:
                self._ybins = np.linspace(ext_ymin, ext_ymax, ext_ny+1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        if min_duration is None:
            min_duration = 0

        self._min_duration = min_duration
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

        # TODO: support 2D sigma
        if sigma is not None:
            if sigma > 0:
                self.smooth(sigma=sigma, bw=bw, inplace=True)

        # optionally detach _bst and _extern to save space when pickling, for example
        self._detach()

    def spatial_information(self):
        """Compute the spatial information and firing sparsity...

        The specificity index examines the amount of information
        (in bits) that a single spike conveys about the animal's
        location (i.e., how well cell firing predicts the animal's
        location).The spatial information content of cell discharge was
        calculated using the formula:
            information content = \Sum P_i(R_i/R)log_2(R_i/R)
        where i is the bin number, P_i, is the probability for occupancy
        of bin i, R_i, is the mean firing rate for bin i, and R is the
        overall mean firing rate.

        In order to account for the effects of low firing rates (with
        fewer spikes there is a tendency toward higher information
        content) or random bursts of firing, the spike firing
        time-series was randomly offset in time from the rat location
        time-series, and the information content was calculated. A
        distribution of the information content based on 100 such random
        shifts was obtained and was used to compute a standardized score
        (Zscore) of information content for that cell. While the
        distribution is not composed of independent samples, it was
        nominally normally distributed, and a Z value of 2.29 was chosen
        as a cut-off for significance (the equivalent of a one-tailed
        t-test with P = 0.01 under a normal distribution).

        Reference(s)
        ------------
        Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
            and Skaggs, W. E. (1994). "Spatial information content and
            reliability of hippocampal CA1 neurons: effects of visual
            input", Hippocampus, 4(4), 410-421.

        Parameters
        ----------

        Returns
        -------
        si : array of shape (n_units,)
            spatial information (in bits) per unit
        sparsity: array of shape (n_units,)
            sparsity (in percent) for each unit
        """

        return utils.spatial_information(ratemap=self.ratemap)

    def spatial_sparsity(self):
        """Compute the spatial information and firing sparsity...

        The specificity index examines the amount of information
        (in bits) that a single spike conveys about the animal's
        location (i.e., how well cell firing predicts the animal's
        location).The spatial information content of cell discharge was
        calculated using the formula:
            information content = \Sum P_i(R_i/R)log_2(R_i/R)
        where i is the bin number, P_i, is the probability for occupancy
        of bin i, R_i, is the mean firing rate for bin i, and R is the
        overall mean firing rate.

        In order to account for the effects of low firing rates (with
        fewer spikes there is a tendency toward higher information
        content) or random bursts of firing, the spike firing
        time-series was randomly offset in time from the rat location
        time-series, and the information content was calculated. A
        distribution of the information content based on 100 such random
        shifts was obtained and was used to compute a standardized score
        (Zscore) of information content for that cell. While the
        distribution is not composed of independent samples, it was
        nominally normally distributed, and a Z value of 2.29 was chosen
        as a cut-off for significance (the equivalent of a one-tailed
        t-test with P = 0.01 under a normal distribution).

        Reference(s)
        ------------
        Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
            and Skaggs, W. E. (1994). "Spatial information content and
            reliability of hippocampal CA1 neurons: effects of visual
            input", Hippocampus, 4(4), 410-421.

        Parameters
        ----------

        Returns
        -------
        si : array of shape (n_units,)
            spatial information (in bits) per unit
        sparsity: array of shape (n_units,)
            sparsity (in percent) for each unit
        """
        return utils.spatial_sparsity(ratemap=self.ratemap)

    def __add__(self, other):
        out = copy.copy(self)

        if isinstance(other, numbers.Number):
            out._ratemap = out.ratemap + other
        elif isinstance(other, TuningCurve2D):
            # TODO: this should merge two TuningCurve2D objects
            raise NotImplementedError
        else:
            raise TypeError("unsupported operand type(s) for +: 'TuningCurve2D' and '{}'".format(str(type(other))))
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

    def _init_from_ratemap(self, ratemap, occupancy=None, ext_xmin=0,
                           ext_xmax=1, ext_ymin=0, ext_ymax=1,
                           extlabels=None, unit_ids=None, unit_labels=None,
                           unit_tags=None, label=None):
        """Initialize a TuningCurve2D object from a ratemap.

        Parameters
        ----------
        ratemap : array
            Array of shape (n_units, ext_nx, ext_ny)

        Returns
        -------

        """
        n_units, ext_nx, ext_ny = ratemap.shape

        if occupancy is None:
            # assume uniform occupancy
            self._occupancy = np.ones((ext_nx, ext_ny))

        if ext_xmin is None:
            ext_xmin = 0
        if ext_xmax is None:
            ext_xmax = ext_xmin + 1

        if ext_ymin is None:
            ext_ymin = 0
        if ext_ymax is None:
            ext_ymax = ext_ymin + 1

        self._xbins = np.linspace(ext_xmin, ext_xmax, ext_nx+1)
        self._ybins = np.linspace(ext_ymin, ext_ymax, ext_ny+1)
        self._ratemap = ratemap

        # inherit unit IDs if available, otherwise initialize to default
        if unit_ids is None:
            unit_ids = list(range(1,n_units + 1))

        unit_ids = np.array(unit_ids, ndmin=1)  # standardize unit_ids

        # if unit_labels is empty, default to unit_ids
        if unit_labels is None:
            unit_labels = unit_ids

        unit_labels = np.array(unit_labels, ndmin=1)  # standardize

        self._unit_ids = unit_ids
        self._unit_labels = unit_labels
        self._unit_tags = unit_tags  # no input validation yet
        if label is not None:
            self.label = label

        return self


    def _detach(self):
        """Detach bst and extern from tuning curve."""
        self._bst = None
        self._extern = None

    @property
    def n_bins(self):
        """(int) Number of external correlates (bins)."""
        return self.n_xbins*self.n_ybins

    @property
    def n_xbins(self):
        """(int) Number of external correlates (bins)."""
        return len(self.xbins) - 1

    @property
    def n_ybins(self):
        """(int) Number of external correlates (bins)."""
        return len(self.ybins) - 1

    @property
    def xbins(self):
        """External correlate bins."""
        return self._xbins

    @property
    def ybins(self):
        """External correlate bins."""
        return self._ybins

    @property
    def xbin_centers(self):
        """External correlate bin centers."""
        return (self.xbins + (self.xbins[1] - self.xbins[0])/2)[:-1]

    @property
    def ybin_centers(self):
        """External correlate bin centers."""
        return (self.ybins + (self.ybins[1] - self.ybins[0])/2)[:-1]

    @property
    def bins(self):
        """External correlate bins."""
        return (self.xbins, self.ybins)

    def _trans_func(self, extern, at):
        """Default transform function to map extern into numerical bins.

        Assumes first signal is x-dim, second is y-dim.
        """

        _, ext = extern.asarray(at=at)
        x, y = ext[0,:], ext[1,:]

        return np.atleast_1d(x), np.atleast_1d(y)

    def _compute_occupancy(self):

        x, y = self.trans_func(self._extern, at=self._bst.bin_centers)

        xmin = self.xbins[0]
        xmax = self.xbins[-1]
        ymin = self.ybins[0]
        ymax = self.ybins[-1]

        occupancy, _, _ = np.histogram2d(x, y, bins=[self.xbins, self.ybins], range=([[xmin, xmax], [ymin, ymax]]))

        return occupancy

    def _compute_ratemap(self, min_duration=None):
        """

        min_duration is the min duration in seconds for a bin to be
        considered 'valid'; if too few observations were made, then the
        firing rate is kept at an estimate of 0. If min_duration == 0,
        then all the spikes are used.
        """

        if min_duration is None:
            min_duration = self._min_duration

        x, y = self.trans_func(self._extern, at=self._bst.bin_centers)

        ext_bin_idx_x = np.digitize(x, self.xbins, True)
        ext_bin_idx_y = np.digitize(y, self.ybins, True)

        # make sure that all the events fit between extmin and extmax:
        # TODO: this might rather be a warning, but it's a pretty serious warning...
        if ext_bin_idx_x.max() > self.n_xbins:
            raise ValueError("ext values greater than 'ext_xmax'")
        if ext_bin_idx_x.min() == 0:
            raise ValueError("ext values less than 'ext_xmin'")
        if ext_bin_idx_y.max() > self.n_ybins:
            raise ValueError("ext values greater than 'ext_ymax'")
        if ext_bin_idx_y.min() == 0:
            raise ValueError("ext values less than 'ext_ymin'")

        ratemap = np.zeros((self.n_units, self.n_xbins, self.n_ybins))

        for tt, (bidxx, bidxy) in enumerate(zip(ext_bin_idx_x, ext_bin_idx_y)):
            ratemap[:,bidxx-1, bidxy-1] += self._bst.data[:,tt]

        # apply minimum observation duration
        for uu in range(self.n_units):
            ratemap[uu][self.occupancy*self._bst.ds < min_duration] = 0

        return ratemap / self._bst.ds

    def normalize(self, inplace=False):
        """Normalize firing rates. For visualization."""

        raise NotImplementedError

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
        denom = np.tile(self.occupancy, (self.n_units,1,1))
        denom[denom==0] = 1
        ratemap = self.ratemap / denom
        return ratemap

    @property
    def is2d(self):
        return True

    @property
    def occupancy(self):
        return self._occupancy

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
        """(tuple) The shape of the TuningCurve2D ratemap."""
        if self.isempty:
            return (self.n_units, 0, 0)
        if len(self.ratemap.shape) ==1:
            return ( self.ratemap.shape[0], 1, 1)
        return self.ratemap.shape

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty TuningCurve2D" + address_str + ">"
        shapestr = " with shape (%s, %s, %s)" % (self.shape[0], self.shape[1], self.shape[2])
        return "<TuningCurve2D%s>%s" % (address_str, shapestr)

    @property
    def isempty(self):
        """(bool) True if TuningCurve1D is empty"""
        try:
            return len(self.ratemap) == 0
        except TypeError: #TypeError should happen if ratemap = []
            return True

    @property
    def ratemap(self):
        return self._ratemap

    def __len__(self):
        return self.n_units

    def smooth(self, *, sigma=None, bw=None, inplace=False):
        """Smooths the tuning curve
        """
        if sigma is None:
            sigma = 0.1 # in units of extern
        if bw is None:
            bw = 4

        ds_x = (self.xbins[-1] - self.xbins[0])/self.n_xbins
        ds_y = (self.ybins[-1] - self.ybins[0])/self.n_ybins
        sigma_x = sigma / ds_x
        sigma_y = sigma / ds_y

        if not inplace:
            out = copy.deepcopy(self)
        else:
            out = self

        if self.n_units > 1:
            out._ratemap = scipy.ndimage.filters.gaussian_filter(self.ratemap, sigma=(0,sigma_x, sigma_y), truncate=bw)
        else:
            out._ratemap = scipy.ndimage.filters.gaussian_filter(self.ratemap, sigma=(sigma_x, sigma_y), truncate=bw)

        return out

    def reorder_units_by_ids(self, neworder, *, inplace=False):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,) and in terms of
        unit_ids

        Return
        ------
        out : reordered TuningCurve2D
        """
        def swap_units(arr, frm, to):
            """swap 'units' of a 3D np.array"""
            arr[[frm, to],:,:] = arr[[to, frm],:,:]

        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        unit_ids = list(self.unit_ids)

        neworder = [self.unit_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            swap_units(out._ratemap, frm, to)
            out._unit_ids[frm], out._unit_ids[to] = out._unit_ids[to], out._unit_ids[frm]
            out._unit_labels[frm], out._unit_labels[to] = out._unit_labels[to], out._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        return out

    @property
    def unit_ids(self):
        """Unit IDs contained in the SpikeTrain."""
        return list(self._unit_ids)

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

    __attributes__ = ["_ratemap", "_occupancy",  "_unit_ids", "_unit_labels", "_unit_tags", "_label"]

    def __init__(self, *, bst=None, extern=None, ratemap=None, sigma=None, bw=None, n_extern=None, transform_func=None, minbgrate=None, extmin=0, extmax=1, extlabels=None, unit_ids=None, unit_labels=None, unit_tags=None, label=None, min_duration=None, empty=False):
        """

        If sigma is nonzero, then smoothing is applied.

        We always require bst and extern, and then some combination of
            (1) bin edges, transform_func*
            (2) n_extern, transform_func*
            (3) n_extern, x_min, x_max, transform_func*

            transform_func operates on extern and returns a value that TuninCurve1D can interpret. If no transform is specified, the identity operator is assumed.
        """
        # TODO: input validation
        if not empty:
            if ratemap is None:
                assert bst is not None, "bst must be specified or ratemap must be specified!"
                assert extern is not None, "extern must be specified or ratemap must be specified!"
            else:
                assert bst is None, "ratemap and bst cannot both be specified!"
                assert extern is None, "ratemap and extern cannot both be specified!"

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        if ratemap is not None:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._init_from_ratemap(ratemap=ratemap,
                                    extmin=extmin,
                                    extmax=extmax,
                                    extlabels=extlabels,
                                    unit_ids=unit_ids,
                                    unit_labels=unit_labels,
                                    unit_tags=unit_tags,
                                    label=label)
            return

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

        if min_duration is None:
            min_duration = 0

        self._min_duration = min_duration

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

        # optionally detach _bst and _extern to save space when pickling, for example
        self._detach()

    @property
    def is2d(self):
        return False

    def spatial_information(self):
        """Compute the spatial information and firing sparsity...

        The specificity index examines the amount of information
        (in bits) that a single spike conveys about the animal's
        location (i.e., how well cell firing predicts the animal's
        location).The spatial information content of cell discharge was
        calculated using the formula:
            information content = \Sum P_i(R_i/R)log_2(R_i/R)
        where i is the bin number, P_i, is the probability for occupancy
        of bin i, R_i, is the mean firing rate for bin i, and R is the
        overall mean firing rate.

        In order to account for the effects of low firing rates (with
        fewer spikes there is a tendency toward higher information
        content) or random bursts of firing, the spike firing
        time-series was randomly offset in time from the rat location
        time-series, and the information content was calculated. A
        distribution of the information content based on 100 such random
        shifts was obtained and was used to compute a standardized score
        (Zscore) of information content for that cell. While the
        distribution is not composed of independent samples, it was
        nominally normally distributed, and a Z value of 2.29 was chosen
        as a cut-off for significance (the equivalent of a one-tailed
        t-test with P = 0.01 under a normal distribution).

        Reference(s)
        ------------
        Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
            and Skaggs, W. E. (1994). "Spatial information content and
            reliability of hippocampal CA1 neurons: effects of visual
            input", Hippocampus, 4(4), 410-421.

        Parameters
        ----------

        Returns
        -------
        si : array of shape (n_units,)
            spatial information (in bits) per unit
        sparsity: array of shape (n_units,)
            sparsity (in percent) for each unit
        """

        # Pi = self.occupancy / np.sum(self.occupancy)
        # R = self.ratemap.mean(axis=1) # mean firing rate
        # Ri = self.ratemap.T
        # si = np.sum((Pi*((Ri / R)*np.log2(Ri / R)).T), axis=1)

        # sparsity = np.sum((Pi*Ri.T), axis=1)/(R**2)

        return utils.spatial_information(ratemap=self.ratemap)

    def spatial_sparsity(self):
        """Compute the spatial information and firing sparsity...

        The specificity index examines the amount of information
        (in bits) that a single spike conveys about the animal's
        location (i.e., how well cell firing redicts the animals
        location).The spatial information content of cell discharge was
        calculated using the formula:
            information content = \Sum P_i(R_i/R)log_2(R_i/R)
        where i is the bin number, P, is the probability for occupancy
        of bin i, R, is the mean firing rate for bin i, and R is the
        overall mean firing rate.

        In order to account for the effects of low firing rates (with
        fewer spikes there is a tendency toward higher information
        content) or random bursts of firing, the spike firing
        time-series was randomly offset in time from the rat location
        time-series, and the information content was calculated. A
        distribution of the information content based on 100 such random
        shifts was obtained and was used to compute a standardized score
        (Zscore) of information content for that cell. While the
        distribution is not composed of independent samples, it was
        nominally normally distributed, and a Z value of 2.29 was chosen
        as a cut-off for significance (the equivalent of a one-tailed
        t-test with P = 0.01 under a normal distribution).

        Reference(s)
        ------------
        Markus, E. J., Barnes, C. A., McNaughton, B. L., Gladden, V. L.,
            and Skaggs, W. E. (1994). "Spatial information content and
            reliability of hippocampal CA1 neurons: effects of visual
            input", Hippocampus, 4(4), 410-421.

        Parameters
        ----------

        Returns
        -------
        si : array of shape (n_units,)
            spatial information (in bits) per unit
        sparsity: array of shape (n_units,)
            sparsity (in percent) for each unit
        """
        return utils.spatial_sparsity(ratemap=self.ratemap)

    def _init_from_ratemap(self, ratemap, occupancy=None, extmin=0, extmax=1, extlabels=None, unit_ids=None, unit_labels=None, unit_tags=None, label=None):
        """Initialize a TuningCurve1D object from a ratemap.

        Parameters
        ----------
        ratemap : array
            Array of shape (n_units, n_extern)

        Returns
        -------

        """
        n_units, n_extern = ratemap.shape

        if occupancy is None:
            # assume uniform occupancy
            self._occupancy = np.ones(n_extern)

        if extmin is None:
            extmin = 0
        if extmax is None:
            extmax = extmin + 1

        self._bins = np.linspace(extmin, extmax, n_extern+1)
        self._ratemap = ratemap

        # inherit unit IDs if available, otherwise initialize to default
        if unit_ids is None:
            unit_ids = list(range(1,n_units + 1))

        unit_ids = np.array(unit_ids, ndmin=1)  # standardize unit_ids

        # if unit_labels is empty, default to unit_ids
        if unit_labels is None:
            unit_labels = unit_ids

        unit_labels = np.array(unit_labels, ndmin=1)  # standardize

        self._unit_ids = unit_ids
        self._unit_labels = unit_labels
        self._unit_tags = unit_tags  # no input validation yet
        if label is not None:
            self.label = label

        return self

    def mean(self,*,axis=None):
        """Returns the mean of firing rate (in Hz).
        Parameters
        ----------
        axis : int, optional
            When axis is None, the global mean firing rate is returned.
            When axis is 0, the mean firing rates across units, as a
            function of the external correlate (e.g. position) are
            returned.
            When axis is 1, the mean firing rate for each unit is
            returned.
        Returns
        -------
        mean :
        """
        means = np.mean(self.ratemap, axis=axis).squeeze()
        if means.size == 1:
            return np.asscalar(means)
        return means

    def max(self,*,axis=None):
        """Returns the mean of firing rate (in Hz).
        Parameters
        ----------
        axis : int, optional
            When axis is None, the global mean firing rate is returned.
            When axis is 0, the mean firing rates across units, as a
            function of the external correlate (e.g. position) are
            returned.
            When axis is 1, the mean firing rate for each unit is
            returned.
        Returns
        -------
        mean :
        """
        maxes = np.max(self.ratemap, axis=axis).squeeze()
        if maxes.size == 1:
            return np.asscalar(maxes)
        return maxes

    def min(self,*,axis=None):
        """Returns the mean of firing rate (in Hz).
        Parameters
        ----------
        axis : int, optional
            When axis is None, the global mean firing rate is returned.
            When axis is 0, the mean firing rates across units, as a
            function of the external correlate (e.g. position) are
            returned.
            When axis is 1, the mean firing rate for each unit is
            returned.
        Returns
        -------
        mean :
        """
        mins = np.min(self.ratemap, axis=axis).squeeze()
        if mins.size == 1:
            return np.asscalar(mins)
        return mins

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

    @property
    def bin_centers(self):
        """External correlate bin centers."""
        return (self.bins + (self.bins[1] - self.bins[0])/2)[:-1]

    def _trans_func(self, extern, at):
        """Default transform function to map extern into numerical bins"""

        _, ext = extern.asarray(at=at)

        return np.atleast_1d(ext)

    def _compute_occupancy(self):

        ext = self.trans_func(self._extern, at=self._bst.bin_centers)

        xmin = self.bins[0]
        xmax = self.bins[-1]
        occupancy, _ = np.histogram(ext, bins=self.bins, range=(xmin, xmax))
        # xbins = (bins + xmax/n_xbins)[:-1] # for plotting
        return occupancy

    def _compute_ratemap(self, min_duration=None):

        if min_duration is None:
            min_duration = self._min_duration

        ext = self.trans_func(self._extern, at=self._bst.bin_centers)

        ext_bin_idx = np.digitize(ext, self.bins, True)
        # make sure that all the events fit between extmin and extmax:
        # TODO: this might rather be a warning, but it's a pretty serious warning...
        if ext_bin_idx.max() > self.n_bins:
            raise ValueError("ext values greater than 'ext_max'")
        if ext_bin_idx.min() == 0:
            raise ValueError("ext values less than 'ext_min'")

        ratemap = np.zeros((self.n_units, self.n_bins))

        for tt, bidx in enumerate(ext_bin_idx):
            ratemap[:,bidx-1] += self._bst.data[:,tt]

        # apply minimum observation duration
        for uu in range(self.n_units):
            ratemap[uu][self.occupancy*self._bst.ds < min_duration] = 0

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
        return list(self._unit_ids)

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

        if isinstance(other, numbers.Number):
            out._ratemap = out.ratemap + other
        elif isinstance(other, TuningCurve1D):
            # TODO: this should merge two TuningCurve1D objects
            raise NotImplementedError
        else:
            raise TypeError("unsupported operand type(s) for +: 'TuningCurve1D' and '{}'".format(str(type(other))))
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

    def _unit_subset(self, unit_list):
        """Return a TuningCurve1D restricted to a subset of units.

        Parameters
        ----------
        unit_list : array-like
            Array or list of unit_ids.
        """
        unit_subset_ids = []
        for unit in unit_list:
            try:
                id = self.unit_ids.index(unit)
            except ValueError:
                warnings.warn("unit_id " + str(unit) + " not found in TuningCurve1D; ignoring")
                pass
            else:
                unit_subset_ids.append(id)

        new_unit_ids = (np.asarray(self.unit_ids)[unit_subset_ids]).tolist()
        new_unit_labels = (np.asarray(self.unit_labels)[unit_subset_ids]).tolist()

        if len(unit_subset_ids) == 0:
            warnings.warn("no units remaining in requested unit subset")
            return TuningCurve1D(empty=True)

        newtuningcurve = copy.copy(self)
        newtuningcurve._unit_ids = new_unit_ids
        newtuningcurve._unit_labels = new_unit_labels
        # TODO: implement tags
        # newtuningcurve._unit_tags =
        newtuningcurve._ratemap = self.ratemap[unit_subset_ids,:]
        # TODO: shall we restrict _bst as well? This will require a copy to be made...
        # newtuningcurve._bst =

        return newtuningcurve

    def _get_peak_firing_order_idx(self):
        """Docstring goes here

        ratemap has shape (n_units, n_ext)
        """
        peakorder = np.argmax(self.ratemap, axis=1).argsort()

        return peakorder.tolist()

    def get_peak_firing_order_ids(self):
        """Docstring goes here

        ratemap has shape (n_units, n_ext)
        """
        peakorder = np.argmax(self.ratemap, axis=1).argsort()

        return (np.asanyarray(self.unit_ids)[peakorder]).tolist()

    def _reorder_units_by_idx(self, neworder=None, *, inplace=False):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,) and in 0,..n_units
        and not in terms of unit_ids

        Return
        ------
        out : reordered TuningCurve1D
        """
        if neworder is None:
            neworder = self._get_peak_firing_order_idx()
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._ratemap, frm, to)
            out._unit_ids[frm], out._unit_ids[to] = out._unit_ids[to], out._unit_ids[frm]
            out._unit_labels[frm], out._unit_labels[to] = out._unit_labels[to], out._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        return out

    def reorder_units_by_ids(self, neworder=None, *, inplace=False):
        """Reorder units according to a specified order.

        neworder must be list-like, of size (n_units,) and in terms of
        unit_ids

        Return
        ------
        out : reordered TuningCurve1D
        """
        if neworder is None:
            neworder = self.get_peak_firing_order_ids()
        if inplace:
            out = self
        else:
            out = copy.deepcopy(self)

        # unit_ids = list(unit_ids)
        neworder = [self.unit_ids.index(x) for x in neworder]

        oldorder = list(range(len(neworder)))
        for oi, ni in enumerate(neworder):
            frm = oldorder.index(ni)
            to = oi
            utils.swap_rows(out._ratemap, frm, to)
            out._unit_ids[frm], out._unit_ids[to] = out._unit_ids[to], out._unit_ids[frm]
            out._unit_labels[frm], out._unit_labels[to] = out._unit_labels[to], out._unit_labels[frm]
            # TODO: re-build unit tags (tag system not yet implemented)
            oldorder[frm], oldorder[to] = oldorder[to], oldorder[frm]

        return out

    def reorder_units(self, inplace=False):
        """Convenience function to reorder units by peak firing location."""
        return self.reorder_units_by_ids(inplace=inplace)

    def _detach(self):
        """Detach bst and extern from tuning curve."""
        self._bst = None
        self._extern = None

#----------------------------------------------------------------------#
#======================================================================#

########################################################################
# class TuningCurve1D
########################################################################
# class TuningCurve2D:
#     """Tuning curves (2-dimensional) of multiple units.
#     """

#     __attributes__ = ["_ratemap", "_occupancy",  "_unit_ids", "_unit_labels", "_unit_tags", "_label"]

#     def __init__(self, *, bst=None, extern=None, ratemap=None, sigma=None, bw=None, n_extern=None, transform_func=None, minbgrate=None, extmin=0, extmax=1, extlabels=None, unit_ids=None, unit_labels=None, unit_tags=None, label=None, empty=False):
#         """

#         If sigma is nonzero, then smoothing is applied.

#         We always require bst and extern, and then some combination of
#             (1) bin edges, transform_func*
#             (2) n_extern, transform_func*
#             (3) n_extern, x_min, x_max, transform_func*

#             transform_func operates on extern and returns a value that TuninCurve1D can interpret. If no transform is specified, the identity operator is assumed.
#         """
#     raise NotImplementedError


#----------------------------------------------------------------------#
#======================================================================#

class DirectionalTuningCurve1D(TuningCurve1D):
    """Directional tuning curves (1-dimensional) of multiple units.

    Get in BST
    Get in queriable object for external correlates

    Get in bins, binlabels
    Get in n_bins, xmin, xmax
    Get in a transform function f

    # idea:
    # 1. estimate stratified tuning curves
    # 2. eliminate inactive cells from each stratification
    # 3. find subset that belongs to all (both) stratifications
    # 4. re-estimate tuning curves for common cells using all the epochs
    # 5. remove common cells from stratifications
    #
    # another option is to combine these as three separate TuningCurve1Ds

    Parameters
    ----------

    Attributes
    ----------

    """

    __attributes__ = ["_unit_ids_l2r", "_unit_ids_r2l"]
    __attributes__.extend(TuningCurve1D.__attributes__)

    def __init__(self, *, bst_l2r, bst_r2l, bst_combined, extern, sigma=None, bw=None, n_extern=None, transform_func=None, minbgrate=None, extmin=0, extmax=1, extlabels=None, unit_ids=None, unit_labels=None, unit_tags=None, label=None, empty=False,
    min_peakfiringrate=None, max_avgfiringrate=None, unimodal=False):
        """

        If sigma is nonzero, then smoothing is applied.

        We always require bst and extern, and then some combination of
            (1) bin edges, transform_func*
            (2) n_extern, transform_func*
            (3) n_extern, x_min, x_max, transform_func*

            transform_func operates on extern and returns a value that TuninCurve1D can interpret. If no transform is specified, the identity operator is assumed.
        """
        # TODO: input validation

        # if an empty object is requested, return it:
        if empty:
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            return

        # self._bst_combined = bst_combined
        self._extern = extern

        if min_peakfiringrate is None:
            min_peakfiringrate = 1.5 # Hz minimum peak firing rate

        if max_avgfiringrate is None:
            max_avgfiringrate = 10 # Hz maximum average firing rate

        if minbgrate is None:
            minbgrate = 0.01 # Hz minimum background firing rate

        if n_extern is not None:
            if extmin is not None and extmax is not None:
                self._bins = np.linspace(extmin, extmax, n_extern+1)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        self._min_peakfiringrate = min_peakfiringrate
        self._max_avgfiringrate = max_avgfiringrate
        self._unimodal = unimodal
        self._unit_ids = bst_combined.unit_ids
        self._unit_labels = bst_combined.unit_labels
        self._unit_tags = bst_combined.unit_tags  # no input validation yet
        self.label = label

        if transform_func is None:
            self.trans_func = self._trans_func

        # left to right:
        self._bst = bst_l2r
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
        # store l2r ratemap
        ratemap_l2r = self.ratemap.copy()

        # right to left:
        self._bst = bst_r2l
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
        # store r2l ratemap
        ratemap_r2l = self.ratemap.copy()

        # combined (non-directional):
        self._bst = bst_combined
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
        # store combined ratemap
        ratemap = self.ratemap

        # determine unit membership:
        l2r_unit_ids = self.restrict_units(ratemap_l2r)
        r2l_unit_ids = self.restrict_units(ratemap_r2l)

        common_unit_ids = list(r2l_unit_ids.intersection(l2r_unit_ids))
        l2r_only_unit_ids = list(l2r_unit_ids.difference(common_unit_ids))
        r2l_only_unit_ids = list(r2l_unit_ids.difference(common_unit_ids))

        # update ratemap with directional tuning curves
        for unit_id in l2r_only_unit_ids:
            unit_idx = self.unit_ids.index(unit_id)
            # print('replacing', self._ratemap[unit_idx, :])
            # print('with', ratemap_l2r[unit_idx, :])
            self._ratemap[unit_idx, :] = ratemap_l2r[unit_idx, :]
        for unit_id in r2l_only_unit_ids:
            unit_idx = self.unit_ids.index(unit_id)
            self._ratemap[unit_idx, :] = ratemap_r2l[unit_idx, :]

        self._unit_ids_l2r = l2r_only_unit_ids
        self._unit_ids_r2l = r2l_only_unit_ids

        # optionally detach _bst and _extern to save space when pickling, for example
        self._detach()

    def restrict_units(self, ratemap=None):

        if ratemap is None:
            ratemap = self.ratemap

        # enforce minimum peak firing rate
        unit_ids_to_keep = set(np.asanyarray(self.unit_ids)[np.argwhere(ratemap.max(axis=1)>self._min_peakfiringrate).squeeze().tolist()])
        # enforce maximum average firing rate
        unit_ids_to_keep = unit_ids_to_keep.intersection(set( np.asanyarray(self.unit_ids)[np.argwhere(ratemap.mean(axis=1)<self._max_avgfiringrate).squeeze().tolist()]))
        # remove multimodal units
        if self._unimodal:
            raise NotImplementedError("restriction to unimodal cells not yet implemented!")
            # placecellidx = placecellidx.intersection(set(unimodal_cells))

        return unit_ids_to_keep

    @property
    def unit_ids_l2r(self):
        return self._unit_ids_l2r

    @property
    def unit_ids_r2l(self):
        return self._unit_ids_r2l