"""octagonal maze linearization"""

import copy
import numpy as np
# from scipy import interpolate
# from collections import namedtuple

from ..core import _analogsignalarray, _epocharray
from ..auxiliary import _position
from .. import utils
from ..plotting.core import colorline

# TODO: linsmooth(), wrap(), unwrap(), assign_segments(), lin_to_ideal(), ideal_to_lin(), 2d_to_ideal()


def idealize(asa, segments, segment_assignments=None):
    """ """
    if not segment_assignments:
        segment_assignments = _get_closest_segments(asa, segments)
    ideal_asa = _project_onto_segments(asa, segments, segment_assignments)

    return ideal_asa, segment_assignments


def linearize(ideal_asa, segments, segment_assignments):
    """ """

    linear_asa = _ideal_to_linear(ideal_asa, segments, segment_assignments)
    linear_asa = RingTrajectory(linear_asa, segments=segments)

    return linear_asa


def _linear_to_ideal(points, segments):
    """transform points in the linearized space to points in the idealized (segment-bound) space."""

    if isinstance(points, _analogsignalarray.AnalogSignalArray):
        pts = points.data.T.squeeze()
    else:
        pts = points.squeeze()

    n_segments = len(segments)
    n_pts = np.size(pts)

    segment_lengths = np.sqrt(np.sum(np.diff(segments, axis=1) ** 2, axis=2)).squeeze()
    cum_lengths = np.cumsum(segment_lengths)
    # identify appropriate segment
    segment_assignments = np.searchsorted(cum_lengths, pts).squeeze()
    segment_assignments[segment_assignments >= n_segments] = n_segments - 1

    # get distance along each segment, for each point:
    alpha = (cum_lengths[segment_assignments] - pts) / segment_lengths[
        segment_assignments
    ]

    coords = segments[segment_assignments]
    starts = coords[:, 0]
    stops = coords[:, 1]

    # do convex combination on segment
    new_coords = (
        np.reshape((alpha), (n_pts, 1)) * starts
        + np.reshape((1 - alpha), (n_pts, 1)) * stops
    )

    return new_coords


def linear_to_ideal(linear_asa, segments):
    new_coords = _linear_to_ideal(linear_asa, segments)
    out = _analogsignalarray.AnalogSignalArray(empty=True)
    out._data = new_coords.T
    out._support = linear_asa.support
    out._fs = linear_asa.fs
    out._time = linear_asa.time
    out.__renew__()
    # TODO: cast out to OctagonalMazeArray or something other than RingTrajectory...
    return out


# def _smooth_unwrapped(self, *, fs=None, sigma=None, bw=None, inplace=False):
#     """Smooths the regularly sampled AnalogSignalArray with a Gaussian kernel.

#     Smoothing is applied in time, and the same smoothing is applied to each
#     signal in the AnalogSignalArray.

#     Smoothing is applied within each epoch.

#     Parameters
#     ----------
#     fs : float, optional
#         Sampling rate (in Hz) of AnalogSignalArray. If not provided, it will
#         be obtained from asa.fs
#     sigma : float, optional
#         Standard deviation of Gaussian kernel, in seconds. Default is 0.05 (50 ms)
#     bw : float, optional
#         Bandwidth outside of which the filter value will be zero. Default is 4.0
#     inplace : bool
#         If True the data will be replaced with the smoothed data.
#         Default is False.

#     Returns
#     -------
#     out : AnalogSignalArray
#         An AnalogSignalArray with smoothed data is returned.
#     """
#     kwargs = {"inplace": inplace, "fs": fs, "sigma": sigma, "bw": bw}
#     out = copy.deepcopy(self)
#     out._data = np.atleast_2d(out._unwrap(out.data.squeeze()))
#     out = utils.gaussian_filter(out, **kwargs)
#     out._data = np.atleast_2d(out._wrap(out.data.squeeze()))
#     if inplace:
#         self._data = out._data
#     out.__renew__()
#     self.__renew__()

#     # kwargs = {'inplace' : inplace,
#     #         'fs' : fs,
#     #         'sigma' : sigma,
#     #         'bw' : bw}
#     # data = copy.deepcopy(self.data)
#     # self._data = np.atleast_2d(self._unwrap(self.data.squeeze()))
#     # out = utils.gaussian_filter(self, **kwargs)
#     # out._data = np.atleast_2d(self._wrap(out.data.squeeze()))
#     # out.__renew__()

#     # if inplace:
#     #     self._data = out._data
#     # else:
#     #     self._data =data
#     # self.__renew__()

#     return out


def _ideal_to_linear(points, segments, segment_assignments):
    """transform points in the idealized (segment-based) space to points in the linearized space."""

    if isinstance(points, _analogsignalarray.AnalogSignalArray):
        pts = points.data.T
    else:
        pts = points

    segment_lengths = np.sqrt(np.sum(np.diff(segments, axis=1) ** 2, axis=2)).squeeze()
    cum_lengths = np.cumsum(segment_lengths)

    coords = segments[segment_assignments]
    starts = coords[:, 0]

    # determine distance from segment start point, and offset by cumulative segment lengths from start
    linearized = (
        np.sqrt(np.sum((starts - pts) ** 2, axis=1))
        + np.insert(cum_lengths, 0, 0)[segment_assignments]
    )

    if isinstance(points, _analogsignalarray.AnalogSignalArray):
        out = copy.deepcopy(points)
        out._data = np.atleast_2d(linearized)
        linearized = out

    return linearized


def point_to_line_segment_dist(point, line_segment):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: list of numpy.core.multiarray.ndarray
    :return: The minimum distance to a point.
    :rtype: float
    """
    # unit vector
    unit_line = line_segment[1] - line_segment[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = np.linalg.norm(
        np.cross(line_segment[1] - line_segment[0], line_segment[0] - point)
    ) / np.linalg.norm(unit_line)

    diff = (norm_unit_line[0] * (point[0] - line_segment[0][0])) + (
        norm_unit_line[1] * (point[1] - line_segment[0][1])
    )

    x_seg = (norm_unit_line[0] * diff) + line_segment[0][0]
    y_seg = (norm_unit_line[1] * diff) + line_segment[0][1]

    endpoint_dist = min(
        np.linalg.norm(line_segment[0] - point), np.linalg.norm(line_segment[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line_segment[0][0]  # line point 1 x
    lp1_y = line_segment[0][1]  # line point 1 y
    lp2_x = line_segment[1][0]  # line point 2 x
    lp2_y = line_segment[1][1]  # line point 2 y
    is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
    is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist


def _get_closest_segments(pts, segments):
    """for each point in the trajectory, determine the segment assignment (with smoothing and warning on ties)"""
    # TODO: warning not yet issued on ties

    if isinstance(pts, _analogsignalarray.AnalogSignalArray):
        pts = pts.data.T

    n_pts = len(pts)
    n_segments = len(segments)

    dist_to_segments = np.zeros((n_pts, n_segments))

    for ss, segment in enumerate(segments):
        for pp, point in enumerate(pts):
            dist_to_segments[pp, ss] = point_to_line_segment_dist(point, segment)

    segment_assignments = np.argmin(
        dist_to_segments,
        axis=1,
    )
    return segment_assignments


def _project_onto_segments(points, segments, segment_assignments):
    """for each point, project onto assigned segment (this is idealized position, not linearized yet!)

    x1, x2, x3 = 5, 20, -25
    y1, y2, y3 = 5, 35, -30

    dx = x2 - x1
    dy = y2 - y1
    d2 = dx*dx + dy*dy
    nx = ((x3-x1)*dx + (y3-y1)*dy) / d2

    # restrict to line segment:

    nx = min(1, max(0, nx))

    x = (x2 - x1) * nx + x1
    y = (y2 - y1) * nx + y1
    """

    if isinstance(points, _analogsignalarray.AnalogSignalArray):
        pts = points.data.T
    else:
        pts = points

    n_pts = len(pts)
    n_segments = len(segments)

    idealized = np.zeros((n_pts, 2))

    for ss, segment in enumerate(segments):
        subset = np.argwhere(segment_assignments == ss).squeeze()
        pts_for_segment = pts[subset, :]

        x3, y3 = pts_for_segment[:, 0], pts_for_segment[:, 1]

        segment = segments[ss]

        dx = segment[1, 0] - segment[0, 0]
        dy = segment[1, 1] - segment[0, 1]
        d2 = dx * dx + dy * dy

        nx = ((x3 - segment[0, 0]) * dx + (y3 - segment[0, 1]) * dy) / d2

        # restrict to line segment:
        nx[nx > 1] = 1
        nx[nx < 0] = 0

        x = dx * nx + segment[0, 0]
        y = dy * nx + segment[0, 1]

        idealized[subset, 0] = x
        idealized[subset, 1] = y

    if isinstance(points, _analogsignalarray.AnalogSignalArray):
        from copy import deepcopy

        out = deepcopy(points)
        out._data = idealized.T
        idealized = out

    return idealized


def get_midpoint_radius(pos):
    """Return the midpoint and radius of the hex maze as a tuple (x,y), radius.

    Params
    ======
    pos: PositionArray
        nelpy PositionArray containing the trajectory data.

    Returns
    =======
    midpoint: (x0, y0)
    radius: float
    """
    # make a local copy of the trajectory data
    local_pos = copy.copy(pos)

    # merge the underlyng support to make computations easier
    local_pos._support = pos.support.merge(gap=10)

    # apply smoothing to tame some outliers:
    local_pos = local_pos.smooth(sigma=0.02)

    midpoint = local_pos.min() + (local_pos.max() - local_pos.min()) / 2
    radius = ((local_pos.max() - local_pos.min()) / 2).mean()

    return midpoint, radius


def excise_disk(pos, midpoint, radius, radius_pct=None):
    """Excise all points within a disk from pos.

    Params
    ======
    pos:
    midpoint:
    radius:
    radius_pct: float
        percent of radius within which to excise points. Default is 0.65 (65%).

    Returns
    =======
    newpos:
    """

    if radius_pct is None:
        radius_pct = 0.65

    dist_to_midpoint = np.sqrt(((pos.data.T - midpoint) ** 2).sum(axis=1))
    indisk_idx = np.argwhere(dist_to_midpoint > radius_pct * radius).squeeze()

    local_pos = _position.PositionArray(
        pos.data[:, indisk_idx], timestamps=pos.time[indisk_idx]
    )

    return local_pos


class OctagonalMazeTrajectory(_analogsignalarray.AnalogSignalArray):

    __attributes__ = ["_vertices"]  # PositionArray-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)

    def __init__(
        self,
        data=[],
        *,
        timestamps=None,
        fs=None,
        step=None,
        merge_sample_gap=0,
        support=None,
        in_memory=True,
        labels=None,
        empty=False
    ):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        # cast an AnalogSignalArray to a PositionArray:
        if isinstance(data, _analogsignalarray.AnalogSignalArray):
            self.__dict__ = copy.deepcopy(data.__dict__)
            self.__renew__()
        else:
            kwargs = {
                "data": data,
                "timestamps": timestamps,
                "fs": fs,
                "step": step,
                "merge_sample_gap": merge_sample_gap,
                "support": support,
                "in_memory": in_memory,
                "labels": labels,
            }

            # initialize super:
            super().__init__(**kwargs)

        # if self._vertices does not exist, then create it:
        if not "_vertices" in self.__dict__:
            self._vertices = None
            self._segments = None
            self._segment_lengths = None
            self._segment_assignments = None
            self._idealized = None
            self._linearized = None

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty OctagonalMazeTrajectory" + address_str + ">"
        if self.n_epochs > 1:
            epstr = ": {} segments".format(self.n_epochs)
        else:
            epstr = ""
        dstr = " for a total of {}".format(utils.PrettyDuration(self.support.duration))
        if self.is_2d:
            return "<2D OctagonalMazeTrajectory%s%s>%s" % (address_str, epstr, dstr)
        return "<1D OctagonalMazeTrajectory%s%s>%s" % (address_str, epstr, dstr)

    @property
    def is_2d(self):
        try:
            return self.n_signals == 2
        except IndexError:
            return False

    @property
    def is_1d(self):
        try:
            return self.n_signals == 1
        except IndexError:
            return False

    @property
    def x(self):
        """return x-values, as numpy array."""
        return self.data[0, :]

    @property
    def y(self):
        """return y-values, as numpy array."""
        if self.is_2d:
            return self.data[1, :]
        raise ValueError(
            "OctagonalMazeTrajectory is not 2 dimensional, so y-values are undefined!"
        )

    @property
    def path_length(self):
        """Return the path length along the trajectory."""
        raise NotImplementedError
        lengths = np.sqrt(np.sum(np.diff(self._data_colsig, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)
        return total_length

    @property
    def n_vertices(self):
        """Number of vertices specifying segmented maze."""
        try:
            return len(self._vertices)
        except TypeError:
            return 0

    @property
    def vertices(self):
        """

        array([[ x0,  y0 ],
               [ x1,  y1 ],
               ...
               [ xN,  yN ]])
        """
        return self._vertices

    @vertices.setter
    def vertices(self, val):

        # TODO: do data integrity cheking / validation
        self._vertices = val
        self._compute_segments()
        self._compute_segment_lengths()
        self._segment_assignments = None

    @property
    def segments(self):
        return self._segments

    @property
    def segment_lengths(self):
        return self._segment_lengths

    def speed(self, sigma_pos=None, simga_spd=None):
        """Return the speed, as an AnalogSignalArray."""
        # Idea is to smooth the position with a good default, and then to
        # compute the speed on the smoothed position. Optionally, then, the
        # speed should be smoothed as well.
        raise NotImplementedError

    def direction(self):
        """Return the instantaneous direction estimate as an AnalogSignalArray."""
        # If 1D, then left/right or up/down or fwd/reverse or whatever might
        # make sense, so this signal could be binarized.
        # When 2D, then a continuous cartesian direction vector might make sense
        # (that is, (x,y)-components), or a polar coordinate system might be
        # better suited, with the direction signal being a continuous theta angle

        raise NotImplementedError

    def idealize(self, segments):
        """Project the position onto idealized segments."""
        # plan is to return a PositionArray constrained to the desired segments.

        raise NotImplementedError

    def linearize(self):
        """Linearize the position estimates."""

        # This is unclear how we should do it universally? Maybe have a few presets?

        raise NotImplementedError

    def bin(self, **kwargs):
        """Bin position into grid."""
        raise NotImplementedError

    def _unwrap(self, arr):

        lin = copy.deepcopy(arr)
        for ii in range(1, len(lin)):
            if lin[ii] - lin[ii - 1] >= self.track_length / 2:
                lin[ii:] = lin[ii:] - self.track_length
            elif lin[ii] - lin[ii - 1] < -self.track_length / 2:
                lin[ii:] = lin[ii:] + self.track_length

        return lin

    def _wrap(self, arr):
        return arr % self.track_length

    def _compute_segment_assignments(self):
        segment_assignments = self._get_closest_segments(self.data.T, self.segments)
        self._segment_assignments = segment_assignments

    def _compute_segments(self):
        """make line segments from vertices

        a segment is of shape [[x0, y0], [x1, y2]]

        segments is of shape [[[x0, y0], [x1, y2]],
                                    ...
                            [[x0, y0], [x1, y2]]]

        i.e., shape (n_segments, 2, 2)

        """

        if self.vertices is None:
            self._segments = None

        v1, v2, v3, v4, v5, v6, v7, v8 = self.vertices

        a = [v1, v2]
        b = [v2, v3]
        c = [v3, v4]
        d = [v4, v5]
        e = [v5, v6]
        f = [v6, v7]
        g = [v7, v8]
        h = [v8, v1]

        segments = np.array([a, b, c, d, e, f, g, h])

        self._segments = segments

    @property
    def segment_assignments(self):
        if self._segment_assignments is None:
            self._compute_segment_assignments()
        return self._segment_assignments

    def _compute_segment_lengths(self):
        if self.vertices is None:
            self._segment_lengths = None
            self._segment_assignments = None

        self._segment_lengths = np.sqrt(
            np.sum(np.diff(self._segments, axis=1) ** 2, axis=2)
        ).squeeze()

    @property
    def track_length(self):
        """The total idealized track length."""
        if self.vertices is None:
            return 0
        return np.sum(self._segment_lengths)

    @property
    def unwrapped(self):
        """Unwrapped linearized trajectory."""
        raise NotImplementedError

    def plot_track(self, show_points=True):
        """Plot the idealized track.

        segment_assignments = get_closest_segments(pos, segments)

        xvals, yvals = pos.asarray().yvals
        for ss in range(len(segments)):
            subset = np.argwhere(segment_assignments==ss).squeeze()
            plt.plot(xvals[subset], yvals[subset], '.', markersize=3)
        plt.title('Initial segment assignment')
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.show()

        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        # todo: plot using colorline and show start, stop, and colorbar
        fig, ax = plt.subplots()

        if show_points:
            x, y = self.data
            plt.plot(x, y, ".", color="0.3", markersize=1)
        else:
            xmin, ymin = self.min()
            xmax, ymax = self.max()
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)

        x, y = self.vertices.T
        colorline(
            np.insert(x, self.n_vertices, x[0]),
            np.insert(y, self.n_vertices, y[0]),
            cmap=plt.cm.Spectral_r,
            lw=5,
            cm_range=(0, 1),
        )
        ax.set_aspect("equal")

        cax = fig.add_axes([0.27, 0.95, 0.5, 0.05])
        cb1 = mpl.colorbar.ColorbarBase(
            cax, cmap=plt.cm.Spectral_r, ticks=[0, 1], orientation="horizontal"
        )
        cax.set_xticklabels(["start", "stop"])

    def point_to_line_segment_dist(self, point, line_segment):
        """Calculate the distance between a point and a line segment.

        To calculate the closest distance to a line segment, we first need to check
        if the point projects onto the line segment.  If it does, then we calculate
        the orthogonal distance from the point to the line.
        If the point does not project to the line segment, we calculate the
        distance to both endpoints and take the shortest distance.

        :param point: Numpy array of form [x,y], describing the point.
        :type point: numpy.core.multiarray.ndarray
        :param line: list of endpoint arrays of form [P1, P2]
        :type line: list of numpy.core.multiarray.ndarray
        :return: The minimum distance to a point.
        :rtype: float
        """
        # unit vector
        unit_line = line_segment[1] - line_segment[0]
        norm_unit_line = unit_line / np.linalg.norm(unit_line)

        # compute the perpendicular distance to the theoretical infinite line
        segment_dist = np.linalg.norm(
            np.cross(line_segment[1] - line_segment[0], line_segment[0] - point)
        ) / np.linalg.norm(unit_line)

        diff = (norm_unit_line[0] * (point[0] - line_segment[0][0])) + (
            norm_unit_line[1] * (point[1] - line_segment[0][1])
        )

        x_seg = (norm_unit_line[0] * diff) + line_segment[0][0]
        y_seg = (norm_unit_line[1] * diff) + line_segment[0][1]

        endpoint_dist = min(
            np.linalg.norm(line_segment[0] - point),
            np.linalg.norm(line_segment[1] - point),
        )

        # decide if the intersection point falls on the line segment
        lp1_x = line_segment[0][0]  # line point 1 x
        lp1_y = line_segment[0][1]  # line point 1 y
        lp2_x = line_segment[1][0]  # line point 2 x
        lp2_y = line_segment[1][1]  # line point 2 y
        is_betw_x = lp1_x <= x_seg <= lp2_x or lp2_x <= x_seg <= lp1_x
        is_betw_y = lp1_y <= y_seg <= lp2_y or lp2_y <= y_seg <= lp1_y
        if is_betw_x and is_betw_y:
            return segment_dist
        else:
            # if not, then return the minimum distance to the segment endpoints
            return endpoint_dist

    def _get_closest_segments(self, pts, segments):
        """for each point in the trajectory, determine the segment assignment (with smoothing and warning on ties)"""
        # TODO: warning not yet issued on ties

        if isinstance(pts, _analogsignalarray.AnalogSignalArray):
            pts = pts.data.T

        n_pts = len(pts)
        n_segments = len(segments)

        dist_to_segments = np.zeros((n_pts, n_segments))

        for ss, segment in enumerate(segments):
            for pp, point in enumerate(pts):
                dist_to_segments[pp, ss] = self.point_to_line_segment_dist(
                    point, segment
                )

        segment_assignments = np.argmin(
            dist_to_segments,
            axis=1,
        )
        return segment_assignments

    def _project_onto_segments(self, points, segments, segment_assignments):
        """for each point, project onto assigned segment (this is idealized position, not linearized yet!)

        x1, x2, x3 = 5, 20, -25
        y1, y2, y3 = 5, 35, -30

        dx = x2 - x1
        dy = y2 - y1
        d2 = dx*dx + dy*dy
        nx = ((x3-x1)*dx + (y3-y1)*dy) / d2

        # restrict to line segment:

        nx = min(1, max(0, nx))

        x = (x2 - x1) * nx + x1
        y = (y2 - y1) * nx + y1
        """

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            pts = points.data.T
        else:
            pts = points

        n_pts = len(pts)
        n_segments = len(segments)

        idealized = np.zeros((n_pts, 2))

        for ss, segment in enumerate(segments):
            subset = np.argwhere(segment_assignments == ss).squeeze()
            pts_for_segment = pts[subset, :]

            x3, y3 = pts_for_segment[:, 0], pts_for_segment[:, 1]

            segment = segments[ss]

            dx = segment[1, 0] - segment[0, 0]
            dy = segment[1, 1] - segment[0, 1]
            d2 = dx * dx + dy * dy

            nx = ((x3 - segment[0, 0]) * dx + (y3 - segment[0, 1]) * dy) / d2

            # restrict to line segment:
            nx[nx > 1] = 1
            nx[nx < 0] = 0

            x = dx * nx + segment[0, 0]
            y = dy * nx + segment[0, 1]

            idealized[subset, 0] = x
            idealized[subset, 1] = y

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            from copy import deepcopy

            out = deepcopy(points)
            out._data = idealized.T
            idealized = out

        return idealized

    def _linear_to_ideal(self, points, segments):
        """transform points in the linearized space to points in the idealized (segment-bound) space."""

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            pts = points.data.T.squeeze()
        else:
            pts = points.squeeze()

        n_segments = len(segments)
        n_pts = np.size(pts)

        segment_lengths = np.sqrt(
            np.sum(np.diff(segments, axis=1) ** 2, axis=2)
        ).squeeze()
        cum_lengths = np.cumsum(segment_lengths)
        # identify appropriate segment
        segment_assignments = np.searchsorted(cum_lengths, pts).squeeze()
        segment_assignments[segment_assignments >= n_segments] = n_segments - 1

        # get distance along each segment, for each point:
        alpha = (cum_lengths[segment_assignments] - pts) / segment_lengths[
            segment_assignments
        ]

        coords = segments[segment_assignments]
        starts = coords[:, 0]
        stops = coords[:, 1]

        # do convex combination on segment
        new_coords = (
            np.reshape((alpha), (n_pts, 1)) * starts
            + np.reshape((1 - alpha), (n_pts, 1)) * stops
        )

        return new_coords

    def linear_to_ideal(self, inplace=False):
        new_coords = self._linear_to_ideal(self.linearized, self.segments)
        out = copy.deepcopy(self)
        out._data = new_coords.T
        out._support = self.linearized.support

        if inplace:
            self._idealized = out

        return out

    def _ideal_to_linear(self, points, segments, segment_assignments=None):
        """transform points in the idealized (segment-based) space to points in the linearized space."""

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            pts = points.data.T
        else:
            pts = points

        # identify appropriate segment
        if segment_assignments is None:
            segment_assignments = self.get_closest_segments(pts, segments)

        segment_lengths = np.sqrt(
            np.sum(np.diff(segments, axis=1) ** 2, axis=2)
        ).squeeze()
        cum_lengths = np.cumsum(segment_lengths)

        coords = segments[segment_assignments]
        starts = coords[:, 0]

        # determine distance from segment start point, and offset by cumulative segment lengths from start
        linearized = (
            np.sqrt(np.sum((starts - pts) ** 2, axis=1))
            + np.insert(cum_lengths, 0, 0)[segment_assignments]
        )

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            out = copy.deepcopy(points)
            out._data = np.atleast_2d(linearized)
            linearized = out

        return linearized

    @property
    def idealized(self):
        if self._idealized is None:
            idealized = self._project_onto_segments(
                self, self.segments, self.segment_assignments
            )
            self._idealized = idealized
        return self._idealized

    @property
    def linearized(self):
        if self._linearized is None:
            linearized = self._ideal_to_linear(
                self.idealized, self.segments, self.segment_assignments
            )
            linearized.smooth = linearized._smooth_unwrapped
            self._linearized = linearized

        return self._linearized

    def _smooth_unwrapped(self, *, fs=None, sigma=None, bw=None, inplace=False):
        """Smooths the regularly sampled AnalogSignalArray with a Gaussian kernel.

        Smoothing is applied in time, and the same smoothing is applied to each
        signal in the AnalogSignalArray.

        Smoothing is applied within each epoch.

        Parameters
        ----------
        fs : float, optional
            Sampling rate (in Hz) of AnalogSignalArray. If not provided, it will
            be obtained from asa.fs
        sigma : float, optional
            Standard deviation of Gaussian kernel, in seconds. Default is 0.05 (50 ms)
        bw : float, optional
            Bandwidth outside of which the filter value will be zero. Default is 4.0
        inplace : bool
            If True the data will be replaced with the smoothed data.
            Default is False.

        Returns
        -------
        out : AnalogSignalArray
            An AnalogSignalArray with smoothed data is returned.
        """
        kwargs = {"inplace": inplace, "fs": fs, "sigma": sigma, "bw": bw}
        out = copy.deepcopy(self)
        out._data = np.atleast_2d(out._unwrap(out.data.squeeze()))
        out = utils.gaussian_filter(out, **kwargs)
        out._data = np.atleast_2d(out._wrap(out.data.squeeze()))
        if inplace:
            self._data = out._data
        out.__renew__()
        self.__renew__()

        # kwargs = {'inplace' : inplace,
        #         'fs' : fs,
        #         'sigma' : sigma,
        #         'bw' : bw}
        # data = copy.deepcopy(self.data)
        # self._data = np.atleast_2d(self._unwrap(self.data.squeeze()))
        # out = utils.gaussian_filter(self, **kwargs)
        # out._data = np.atleast_2d(self._wrap(out.data.squeeze()))
        # out.__renew__()

        # if inplace:
        #     self._data = out._data
        # else:
        #     self._data =data
        # self.__renew__()

        return out


class RingTrajectory(_analogsignalarray.AnalogSignalArray):

    __attributes__ = []  # RingTrajectory-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)

    def __init__(
        self,
        data=[],
        *,
        segments,
        timestamps=None,
        fs=None,
        step=None,
        merge_sample_gap=0,
        support=None,
        in_memory=True,
        labels=None,
        empty=False
    ):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        # cast an AnalogSignalArray to a RingTrajectory:
        if isinstance(data, _analogsignalarray.AnalogSignalArray):
            self.__dict__ = copy.deepcopy(data.__dict__)
            self.__renew__()
        else:
            kwargs = {
                "data": data,
                "timestamps": timestamps,
                "fs": fs,
                "step": step,
                "merge_sample_gap": merge_sample_gap,
                "support": support,
                "in_memory": in_memory,
                "labels": labels,
            }

            # initialize super:
            super().__init__(**kwargs)

        self._segment_lengths = np.sqrt(
            np.sum(np.diff(segments, axis=1) ** 2, axis=2)
        ).squeeze()

    def __repr__(self):
        address_str = " at " + str(hex(id(self)))
        if self.isempty:
            return "<empty RingTrajectory" + address_str + ">"
        if self.n_epochs > 1:
            epstr = ": {} segments".format(self.n_epochs)
        else:
            epstr = ""
        dstr = " for a total of {}".format(utils.PrettyDuration(self.support.duration))
        return "<1D RingTrajectory%s%s>%s" % (address_str, epstr, dstr)

    def _unwrap(self, arr):
        lin = copy.deepcopy(arr)
        for ii in range(1, len(lin)):
            if lin[ii] - lin[ii - 1] >= self.track_length / 2:
                lin[ii:] = lin[ii:] - self.track_length
            elif lin[ii] - lin[ii - 1] < -self.track_length / 2:
                lin[ii:] = lin[ii:] + self.track_length

        return lin

    @property
    def track_length(self):
        """The total idealized track length."""
        return np.sum(self._segment_lengths)

    def _wrap(self, arr):
        return arr % self.track_length

    def wrap(self):
        """"""
        self._data = np.atleast_2d(self._wrap(self._data.squeeze()))

    def unwrap(self):
        """"""
        self._data = np.atleast_2d(self._unwrap(self._data.squeeze()))

    # def smooth(self, *, fs=None, sigma=None, bw=None, inplace=False):
    #     """Smooths the regularly sampled AnalogSignalArray with a Gaussian kernel.

    #     Smoothing is applied in time, and the same smoothing is applied to each
    #     signal in the AnalogSignalArray.

    #     Smoothing is applied within each epoch.

    #     Parameters
    #     ----------
    #     fs : float, optional
    #         Sampling rate (in Hz) of AnalogSignalArray. If not provided, it will
    #         be obtained from asa.fs
    #     sigma : float, optional
    #         Standard deviation of Gaussian kernel, in seconds. Default is 0.05 (50 ms)
    #     bw : float, optional
    #         Bandwidth outside of which the filter value will be zero. Default is 4.0
    #     inplace : bool
    #         If True the data will be replaced with the smoothed data.
    #         Default is False.

    #     Returns
    #     -------
    #     out : AnalogSignalArray
    #         An AnalogSignalArray with smoothed data is returned.
    #     """
    #     # kwargs = {'inplace' : inplace,
    #     #         'fs' : fs,
    #     #         'sigma' : sigma,
    #     #         'bw' : bw}
    #     # out = copy.deepcopy(self)
    #     # out._data = np.atleast_2d(out._unwrap(out.data.squeeze()))
    #     # out = utils.gaussian_filter(out, **kwargs)
    #     # out._data = np.atleast_2d(out._wrap(out.data.squeeze()))
    #     # if inplace:
    #     #     self._data = out._data
    #     # out.__renew__()
    #     # self.__renew__()

    #     kwargs = {'inplace' : inplace,
    #             'fs' : fs,
    #             'sigma' : sigma,
    #             'bw' : bw}
    #    data = copy.deepcopy(self.data)
    #     self._data = np.atleast_2d(self._unwrap(self.data.squeeze()))
    #     out = utils.gaussian_filter(self, **kwargs)
    #     out._data = np.atleast_2d(self._wrap(out.data.squeeze()))
    #     out.__renew__()

    #     # if inplace:
    #     #     self._data = out._data
    #     # else:
    #     #     self._data =data
    #     # self.__renew__()

    #     return out

    # def _get_interp1d(self,* , kind='linear', copy=True, bounds_error=False,
    #                   fill_value=np.nan, assume_sorted=None):
    #     """returns a scipy interp1d object, extended to have values at all epoch
    #     boundaries!
    #     """

    #     if assume_sorted is None:
    #         assume_sorted = utils.is_sorted(self.time)

    #     if self.n_signals > 1:
    #         axis = 1
    #     else:
    #         axis = -1

    #     time = self.time
    #     yvals = self._unwrap(self._data_rowsig)
    #     lengths = self.lengths
    #     empty_epoch_ids = np.argwhere(lengths==0).squeeze().tolist()
    #     first_timestamps_per_epoch_idx = np.insert(np.cumsum(lengths[:-1]),0,0)
    #     first_timestamps_per_epoch_idx[empty_epoch_ids] = 0
    #     last_timestamps_per_epoch_idx = np.cumsum(lengths)-1
    #     last_timestamps_per_epoch_idx[empty_epoch_ids] = 0
    #     first_timestamps_per_epoch = self.time[first_timestamps_per_epoch_idx]
    #     last_timestamps_per_epoch = self.time[last_timestamps_per_epoch_idx]

    #     boundary_times = []
    #     boundary_vals = []
    #     for ii, (start, stop) in enumerate(self.support.time):
    #         if lengths[ii] == 0:
    #             continue
    #         if first_timestamps_per_epoch[ii] > start:
    #             boundary_times.append(start)
    #             boundary_vals.append(yvals[:,first_timestamps_per_epoch_idx[ii]])
    #             # print('adding {} at time {}'.format(yvals[:,first_timestamps_per_epoch_idx[ii]], start))
    #         if last_timestamps_per_epoch[ii] < stop:
    #             boundary_times.append(stop)
    #             boundary_vals.append(yvals[:,last_timestamps_per_epoch_idx[ii]])

    #     if boundary_times:
    #         insert_locs = np.searchsorted(time, boundary_times)
    #         time = np.insert(time, insert_locs, boundary_times)
    #         yvals = np.insert(yvals, insert_locs, np.array(boundary_vals).T, axis=1)

    #         time, unique_idx = np.unique(time, return_index=True)
    #         yvals = yvals[:,unique_idx]

    #     f = interpolate.interp1d(x=time,
    #                              y=yvals,
    #                              kind=kind,
    #                              axis=axis,
    #                              copy=copy,
    #                              bounds_error=bounds_error,
    #                              fill_value=fill_value,
    #                              assume_sorted=assume_sorted)
    #     return f

    # def asarray(self,*, where=None, at=None, kind='linear', copy=True,
    #             bounds_error=False, fill_value=np.nan, assume_sorted=None,
    #             recalculate=False, store_interp=True, n_points=None,
    #             split_by_epoch=False):
    #     """returns adata_like array at requested points.

    #     Parameters
    #     ----------
    #     where : array_like or tuple, optional
    #         array corresponding to np where condition
    #         e.g., where=(data[1,:]>5) or tuple where=(speed>5,tspeed)
    #     at : array_like, optional
    #         Array of oints to evaluate array at. If none given, use
    #         self.time together with 'where' if applicable.
    #     n_points: int, optional
    #         Number of points to interplate at. These points will be
    #         distributed uniformly from self.support.start to stop.
    #     split_by_epoch: bool
    #         If True, separate arrays by epochs and return in a list.
    #     Returns
    #     -------
    #     out : (array, array)
    #         namedtuple tuple (xvals, yvals) of arrays, where xvals is an
    #         array of time points for which (interpolated)data are
    #         returned.
    #     """

    #     # TODO: implement splitting by epoch

    #     if split_by_epoch:
    #         raise NotImplementedError("split_by_epoch not yet implemented...")

    #     XYArray = namedtuple('XYArray', ['xvals', 'yvals'])

    #     if at is None and where is None and split_by_epoch is False and n_points is None:
    #         xyarray = XYArray(self.time, self._data_rowsig.squeeze())
    #         return xyarray

    #     if where is not None:
    #         assert at is None and n_points is None, "'where', 'at', and 'n_points' cannot be used at the same time"
    #         if isinstance(where, tuple):
    #             y = np.array(where[1]).squeeze()
    #             x = where[0]
    #             assert len(x) == len(y), "'where' condition and array must have same number of elements"
    #             at = y[x]
    #         else:
    #             x = np.asanyarray(where).squeeze()
    #             assert len(x) == len(self.time), "'where' condition must have same number of elements as self.time"
    #             at = self.time[x]
    #     elif at is not None:
    #         assert n_points is None, "'at' and 'n_points' cannot be used at the same time"
    #     else:
    #         at = np.linspace(self.support.start, self.support.stop, n_points)

    #     # if we made it this far, either at or where has been specified, and at is now well defined.

    #     kwargs = {'kind':kind,
    #               'copy':copy,
    #               'bounds_error':bounds_error,
    #               'fill_value':fill_value,
    #               'assume_sorted':assume_sorted}

    #     # retrieve an existing, or construct a new interpolation object
    #     if recalculate:
    #         interpobj = self._get_interp1d(**kwargs)
    #     else:
    #         try:
    #             interpobj = self._interp
    #             if interpobj is None:
    #                 interpobj = self._get_interp1d(**kwargs)
    #         except AttributeError: # does not exist yet
    #             interpobj = self._get_interp1d(**kwargs)

    #     # store interpolation object, if desired
    #     if store_interp:
    #         self._interp = interpobj

    #     # do the actual interpolation
    #     try:
    #         out = interpobj(at)
    #     except SystemError:
    #         interpobj = self._get_interp1d(**kwargs)
    #         if store_interp:
    #             self._interp = interpobj
    #         out = interpobj(at)

    #     # TODO: set all values outside of self.support to fill_value

    #     xyarray = XYArray(xvals=np.asanyarray(at), yvals=np.asanyarray(out).squeeze())
    #     return xyarray
