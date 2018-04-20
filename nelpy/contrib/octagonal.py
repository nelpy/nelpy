"""octagonal maze linearization"""

import copy
import numpy as np

from ..core import _analogsignalarray, _epocharray
from .. import utils

#TODO: linsmooth(), wrap(), unwrap(), assign_segments(), lin_to_ideal(), ideal_to_lin(), 2d_to_ideal()

class OctagonalMazeTrajectory(_analogsignalarray.AnalogSignalArray):

    __attributes__ = ['_vertices'] # PositionArray-specific attributes
    __attributes__.extend(_analogsignalarray.AnalogSignalArray.__attributes__)
    def __init__(self, ydata=[], *, timestamps=None, fs=None,
                 step=None, merge_sample_gap=0, support=None,
                 in_memory=True, labels=None, empty=False):

        # if an empty object is requested, return it:
        if empty:
            super().__init__(empty=True)
            for attr in self.__attributes__:
                exec("self." + attr + " = None")
            self._support = _epocharray.EpochArray(empty=True)
            return

        # cast an AnalogSignalArray to a PositionArray:
        if isinstance(ydata, _analogsignalarray.AnalogSignalArray):
            self.__dict__ = copy.deepcopy(ydata.__dict__)
            self.__renew__()
        else:
            kwargs = {"ydata": ydata,
                    "timestamps": timestamps,
                    "fs": fs,
                    "step": step,
                    "merge_sample_gap": merge_sample_gap,
                    "support": support,
                    "in_memory": in_memory,
                    "labels": labels}

            # initialize super:
            super().__init__(**kwargs)

        # if self._vertices does not exist, then create it:
        if not '_vertices' in self.__dict__:
            self._vertices = None

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
        return self.ydata[0,:]

    @property
    def y(self):
        """return y-values, as numpy array."""
        if self.is_2d:
            return self.ydata[1,:]
        raise ValueError("OctagonalMazeTrajectory is not 2 dimensional, so y-values are undefined!")

    @property
    def path_length(self):
        """Return the path length along the trajectory."""
        raise NotImplementedError
        lengths = np.sqrt(np.sum(np.diff(self._ydata_colsig, axis=0)**2, axis=1))
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

    def unwrap(self, arr):

        lin = copy.deepcopy(arr)
        for ii in range(1, len(lin)):
            if lin[ii] - lin[ii-1] >= self.segments.length/2:
                lin[ii:] = lin[ii:] - self.segments.length
            elif lin[ii] - lin[ii-1] < -self.segments.length/2:
                lin[ii:] = lin[ii:] + self.segments.length

        return lin

    def _wrap(self, arr):
        return arr % self.track_length

    @property
    def track_length(self):
        """The total idealized track length."""
        raise NotImplementedError

    @property
    def unwrapped(self):
        """Unwrapped linearized trajectory."""
        raise NotImplementedError

    def plot_track(self, show_points=True):
        """Plot the idealized track."""
        # todo: plot using colorline and show start, stop, and colorbar
        raise NotImplementedError

    def make_segments_from_vertices(self, vertices):
        """make line segments from vertices

        a segment is of shape [[x0, y0], [x1, y2]]

        segments is of shape [[[x0, y0], [x1, y2]],
                                    ...
                            [[x0, y0], [x1, y2]]]

        i.e., shape (n_segments, 2, 2)

        """

        v1, v2, v3, v4, v5, v6, v7, v8 = vertices

        a = [v1, v2]
        b = [v2, v3]
        c = [v3, v4]
        d = [v4, v5]
        e = [v5, v6]
        f = [v6, v7]
        g = [v7, v8]
        h = [v8, v1]

        vertices = np.array([v1, v2, v3, v4, v5, v6, v7, v8])
        segments = np.array([a,b,c,d,e,f,g,h])

        return segments, vertices

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
        segment_dist = (
            np.linalg.norm(np.cross(line_segment[1] - line_segment[0], line_segment[0] - point)) /
            np.linalg.norm(unit_line)
        )

        diff = (
            (norm_unit_line[0] * (point[0] - line_segment[0][0])) +
            (norm_unit_line[1] * (point[1] - line_segment[0][1]))
        )

        x_seg = (norm_unit_line[0] * diff) + line_segment[0][0]
        y_seg = (norm_unit_line[1] * diff) + line_segment[0][1]

        endpoint_dist = min(
            np.linalg.norm(line_segment[0] - point),
            np.linalg.norm(line_segment[1] - point)
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

    def get_closest_segments(self, pts, segments):
        """for each point in the trajectory, determine the segment assignment (with smoothing and warning on ties)"""
        # TODO: warning not yet issued on ties

        if isinstance(pts, _analogsignalarray.AnalogSignalArray):
            pts = pts.ydata.T

        n_pts = len(pts)
        n_segments = len(segments)

        dist_to_segments = np.zeros((n_pts, n_segments))

        for ss, segment in enumerate(segments):
            for pp, point in enumerate(pts):
                dist_to_segments[pp, ss] = self.point_to_line_segment_dist(point, segment)

        segment_assignments = np.argmin(dist_to_segments, axis=1, )
        return segment_assignments

    def project_onto_segments(self, points, segments, segment_assignments):
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
            pts = points.ydata.T
        else:
            pts = points

        n_pts = len(pts)
        n_segments = len(segments)

        idealized = np.zeros((n_pts,2))

        for ss, segment in enumerate(segments):
            subset = np.argwhere(segment_assignments==ss).squeeze()
            pts_for_segment = pts[subset,:]

            x3, y3 = pts_for_segment[:,0], pts_for_segment[:,1]

            segment = segments[ss]

            dx = segment[1,0] - segment[0,0]
            dy = segment[1,1] - segment[0,1]
            d2 = dx*dx + dy*dy

            nx = ((x3-segment[0,0])*dx + (y3-segment[0,1])*dy) / d2

            # restrict to line segment:
            nx[nx>1] = 1
            nx[nx<0] = 0

            x = dx*nx + segment[0,0]
            y = dy*nx + segment[0,1]

            idealized[subset,0] = x
            idealized[subset,1] = y

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            from copy import deepcopy
            out = deepcopy(points)
            out._ydata = idealized.T
            idealized = out

        return idealized

    def linear_to_ideal(self, points, segments):
        """transform points in the linearized space to points in the idealized (segment-bound) space."""

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            pts = points.ydata.T
        else:
            pts = points

        n_segments = len(segments)
        n_pts = np.size(pts)

        segment_lengths = np.sqrt(np.sum(np.diff(segments, axis=1)**2, axis=2)).squeeze()
        cum_lengths = np.cumsum(segment_lengths)
        # identify appropriate segment
        segment_assignments = np.searchsorted(cum_lengths, pts)
        segment_assignments[segment_assignments >= n_segments] = n_segments - 1

        # get distance along each segment, for each point:
        alpha = (cum_lengths[segment_assignments] - pts) / segment_lengths[segment_assignments]

        coords = segments[segment_assignments]
        starts = coords[:,0]
        stops = coords[:,1]

        # do convex combination on segment
        new_coords = np.reshape((alpha), (n_pts,1)) * starts + np.reshape((1-alpha), (n_pts,1)) * stops

        return new_coords

    def ideal_to_linear(self, points, segments):
        """transform points in the idealized (segment-based) space to points in the linearized space."""

        if isinstance(points, _analogsignalarray.AnalogSignalArray):
            pts = points.ydata.T
        else:
            pts = points

        # identify appropriate segment
        segment_assignments = self.get_closest_segments(pts, segments)

        segment_lengths = np.sqrt(np.sum(np.diff(segments, axis=1)**2, axis=2)).squeeze()
        cum_lengths = np.cumsum(segment_lengths)

        coords = segments[segment_assignments]
        starts = coords[:,0]

        # determine distance from segment start point, and offset by cumulative segment lengths from start
        linearized = np.sqrt(np.sum((starts - pts)**2, axis=1)) + np.insert(cum_lengths,0,0)[segment_assignments]

        return linearized