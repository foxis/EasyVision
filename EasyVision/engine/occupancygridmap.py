# -*- coding: utf-8 -*-
"""Implements simple OccupancyGridMap using 3d features provided with a pose
"""
from EasyVision.engine.base import MapBase
from EasyVision.engine.base import Pose
import numpy as np
import cv2
from math import sin, cos, acos, asin


class OccupancyGridMap(MapBase):
    """Class implementing Occupancy Grid Mapping.
    Should be used with conjunction with VisualOdometry engine.
    Uses 3d feature points of the Pose. Z coordinate denotes forward, and Y coordinate - up.

    """

    def __init__(self, _map, scale=.001, theta=.01, alpha=.6, beta=-.4, min_y=-10, max_y=5000, max_d=100000, poses=[], *args, **kwargs):
        """Instance initialization.

        :param _map: Accepts either a tuple of (map_width, map_height) or a np.float32 two dimensional array
        :param scale: Scale of the map
        :param theta: obstacle reading spread
        :param alpha: obstacle weight
        :param beta: freespace before obstacle weight
        :param min_y: minimum Y coordinate of the feature
        :param max_y: maximum Y coordinate of the feature
        :param max_d: maximum distance of the feature
        :param poses: a list of poses to initialize a map with
        """
        if not isinstance(poses, list) or not all(isinstance(i, Pose) for i in poses):
            raise TypeError("Poses must be a list of Pose")

        if isinstance(_map, np.ndarray) and len(_map.shape) == 2:
            self._map = cv2.normalize(_map, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)
        elif isinstance(_map, tuple) and len(_map) == 2:
            self._map = np.zeros(_map + (1,), dtype=np.float32)
        else:
            raise TypeError("Map must be either two dimentional numpy array or a tuple e.g. (width, height)")

        self._obstacles = np.zeros(self._map.shape, dtype=self._map.dtype)
        self._sensor_model = np.zeros(self._map.shape, dtype=self._map.dtype)

        self._poses = poses
        self._scale = scale
        self._min_y = min_y
        self._max_y = max_y
        self._max_d = max_d
        self._alpha = alpha
        self._beta = beta
        self._theta = theta
        super(OccupancyGridMap, self).__init__(*args, **kwargs)

    @property
    def map_raw(self):
        return self._map

    @property
    def pose(self):
        return self._poses[-1]

    @property
    def path(self):
        return self._poses

    def __iter__(self):
        return self._poses

    def __len__(self):
        return len(self._poses)

    def next(self):
        super(OccupancyGridMap, self).next()

    def setup(self):
        super(OccupancyGridMap, self).setup()

    def release(self):
        super(OccupancyGridMap, self).release()

    @property
    def description(self):
        return "Log Odd Occupancy Grid map for 3D features used together with Odometry"

    def update(self, pose, **kwargs):
        """Will add a pose to the path, and update occupancy grid if points3d are provided.
        If alpha/beta/theta provided - will override the ones provided at the instantiation.
        """
        if not isinstance(pose, Pose):
            raise TypeError("Pose must be of type Pose")

        self._poses += [pose]

        if pose.features is None or pose.features.points3d is None:
            return pose

        alpha = kwargs.get('alpha', self._alpha)
        beta = kwargs.get('beta', self._beta)
        theta = kwargs.get('theta', self._theta)
        scale = kwargs.get('scale', 1.0)
        max_d = kwargs.get('max_d', self._max_d)

        R, t = pose.rotation, pose.translation

        pts = np.float32([[[a] for a in pt] for pt in pose.features.points3d if self._min_y < pt[1] < self._max_y])

        self._obstacles[:] = 0
        self._sensor_model[:] = 0

        for pt in pts:
            dd = (pt[0][0] ** 2 + pt[2][0] ** 2) ** .5
            a = acos(pt[0][0] / dd)

            _arc = np.float32([
                [[0], [0], [0]],
                [[dd * cos(a + theta)], [0], [dd * sin(a + theta)]],
                [[dd * cos(a - theta)], [0], [dd * sin(a - theta)]]
            ])

            p = (R.dot(pt * scale) + t) * self._scale
            arc = np.float32([(R.dot(ap * scale) + t) * self._scale for ap in _arc])

            _ddd = arc[1] - arc[2]
            ddd = (_ddd[0][0] ** 2 + _ddd[2][0] ** 2) ** .5

            if dd < max_d:
                self._obstacles = cv2.circle(self._obstacles, (int(p[0][0]), int(p[2][0])), int(max(1, ddd / 2)), (alpha), -1)
            __arc = np.int32([[pt[0][0], pt[2][0]] for pt in arc])
            self._sensor_model = cv2.fillPoly(self._sensor_model, [__arc], (beta))

        self._map = self._map + self._obstacles + self._sensor_model
        if self.display_results:
            self.draw()

        return pose

    def draw(self, path=None):
        """Helper method to draw the map"""
        if not len(self._poses):
            return

        disp = self._map
        #disp = cv2.inRange(self._map, (0), (255))
        disp = cv2.normalize(disp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

        tl = self._poses[0].translation * self._scale
        for p in self._poses[1:]:
            t = p.translation * self._scale
            cv2.line(disp, (int(tl[0][0]), int(tl[2][0])), (int(t[0][0]), int(t[2][0])), (0, 0, 255))
            tl = t

        if path is not None:
            tl = path[0]
            for t in path[1:]:
                cv2.line(disp, (int(tl[0]), int(tl[1])), (int(t[0]), int(t[1])), (0, 255, 0))
                tl = t

        cv2.imshow(self.name, disp)

    def plan(self, target, radius, **kwargs):
        """Finds the shortest path in the map between current and target poses.
        Only uses translation part of the Pose.

        :param target: Pose with valid coordinates. Will be translated to map coordinates using scale.
        :param radius: map blurring radius in world coordinates. Will be scaled. Specifies how far the path should be away from obstacles.
        :returns: an iterable of (x, y) scaled map coordinates.
        """

        start = (int(self.pose.translation[0][0] * self._scale), int(self.pose.translation[2][0] * self._scale))
        goal = (int(target.translation[0][0] * self._scale), int(target.translation[2][0] * self._scale))
        bs = int(radius * self._scale + .5)

        grid = cv2.blur(self._map, (bs, bs))
        grid += self._map
        grid = cv2.normalize(grid, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32FC1)

        grid = grid.tolist()

        h, w = self._map.shape

        def heuristic(_a, _b):
            c = grid[_a[1]][_a[0]] + 2.0
            return c * ((_a[0] - _b[0]) ** 2 + (_a[1] - _b[1]) ** 2) ** .5

        deltas = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1))

        def neighbors(current):
            for delta in deltas:
                x, y = current[0] + delta[0], current[1] + delta[1]
                if x < 0 or y < 0:
                    continue
                if x >= w or y >= h:
                    continue
                if grid[y][x] > 0:
                    continue

                yield x, y

        path = np.float32(tuple(self.astar(start, goal, neighbors, heuristic))) / self._scale

        return path[::-1]
