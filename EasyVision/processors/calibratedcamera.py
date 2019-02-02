# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *


class PinholeCamera(namedtuple('PinholeCamera', ['size', 'matrix', 'distortion', 'rectify', 'projection'])):
    """Pinhole Camera model for calibrated camera processor.

    Contains these fields:
        size - (width, height)
        matrix - camera matrix 3x3
        distortion - camera distortion coefficients
        rectify - rectification transform matrix 3x3
        projection - projection matrix after rectification 3x3

    """

    def __new__(cls, size, matrix, distortion, rectify=None, projection=None):
        if isinstance(size, list):
            size = tuple(size)
        if not isinstance(size, tuple) or len(size) != 2 or not all((isinstance(i, int) or isinstance(i, long)) and i > 0 for i in size):
            raise TypeError('Frame size must be a tuple consisting of two positive integers')
        matrix = np.float32(matrix) if not isinstance(matrix, np.ndarray) and matrix is not None else matrix
        distortion = np.float32(distortion) if not isinstance(distortion, np.ndarray) and distortion is not None else distortion
        rectify = np.float32(rectify) if not isinstance(rectify, np.ndarray) and rectify is not None else rectify
        projection = np.float32(projection) if not isinstance(projection, np.ndarray) and projection is not None else projection

        return super(PinholeCamera, cls).__new__(cls, size, matrix, distortion, rectify, projection)

    @property
    def width(self):
        return self.size[0]

    @property
    def height(self):
        return self.size[1]

    @property
    def focal_point(self):
        return (self.matrix[0, 0], self.matrix[1, 1])

    @property
    def center(self):
        return (self.matrix[0, 2], self.matrix[1, 2])

    @staticmethod
    def fromdict(as_dict):
        return PinholeCamera(**as_dict)

    def todict(self):
        d = {
            "size": self.size,
            "matrix": self.matrix.tolist(),
            "distortion": self.distortion.tolist(),
            "rectify": self.rectify.tolist() if self.rectify is not None else None,
            "projection": self.projection.tolist() if self.projection is not None else None
        }
        return d

    @staticmethod
    def from_parameters(frame_size, focal_point, center, distortion, rectify=None, projection=None):
        if len(distortion) != 5:
            raise ValueError("distortion must be vector of length 5")
        if len(frame_size) != 2:
            raise ValueError("frame size must be vector of length 2")
        if len(focal_point) != 2:
            raise ValueError("focal point must be vector of length 2")
        if len(center) != 2:
            raise ValueError("center must be vector of length 2")
        matrix = np.zeros((3, 3), np.float32)
        matrix[0, 0] = focal_point[0]
        matrix[1, 1] = focal_point[1]
        matrix[0, 2] = center[0]
        matrix[1, 2] = center[1]
        matrix[2, 2] = 1
        d = np.zeros((1, 5), np.float32)
        d[0] = distortion
        return PinholeCamera(frame_size, matrix, d, rectify, projection)


class CalibratedCamera(ProcessorBase):
    def __init__(self, vision, camera, grid_shape=(7, 6), square_size=20, max_samples=20, frame_delay=1, *args, **kwargs):
        calibrate = camera is None
        if not calibrate:
            if not isinstance(camera, PinholeCamera) and not (isinstance(camera, tuple) and len(camera) == 3):
                raise TypeError("Camera must be either PinholeCamera or tuple with (frame_size, camera_matrix, distortion)")
            self._camera = PinholeCamera._make(camera)
        else:
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self._camera = None
            self._grid_shape = grid_shape
            self._square_size = square_size
            self._max_samples = max_samples
            self._frame_delay = frame_delay
            self._last_timestamp = None

        self._calibrate = calibrate
        super(CalibratedCamera, self).__init__(vision, *args, **kwargs)

    def setup(self):
        if self._calibrate:
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self.objp = np.zeros((np.prod(self._grid_shape), 3), np.float32)
            self.objp[:, :2] = np.indices(self._grid_shape).T.reshape(-1, 2)
            self.objp *= self._square_size

            # Arrays to store object points and image points from all the images.
            self.objpoints = []  # 3d point in real world space
            self.imgpoints = []  # 2d points in image plane.
            self.calibration_samples = 0
        else:
            self._mapx, self._mapy = cv2.initUndistortRectifyMap(
                    self.camera.matrix,
                    self.camera.distortion,
                    self.camera.rectify,
                    self.camera.projection,
                    self.camera.size,
                    cv2.CV_32FC1)
        super(CalibratedCamera, self).setup()

    @property
    def description(self):
        return "Pinhole camera undistort processor"

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        if not isinstance(value, PinholeCamera):
            raise TypeError("Must be PinholeCamera")
        self._camera = value

    def process(self, image):
        if self._calibrate:
            img = image.image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self._grid_shape, None)
            if ret is True:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.criteria)

            # Draw and display the corners
            if self.display_results:
                img = cv2.drawChessboardCorners(img, self._grid_shape, corners, ret)
                cv2.putText(img, "Samples added: {}/{}".format(self.calibration_samples, self._max_samples),
                            (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
                cv2.imshow(self.name, img)

            return Image(self, gray, features=(ret, corners), feature_type='corners')
        else:
            mapped = cv2.remap(image.image, self._mapx, self._mapy, cv2.INTER_NEAREST)

            if self.display_results:
                cv2.imshow(self.name, mapped)

            return image._replace(image=mapped)

    def calibrate(self):
        if not self._calibrate:
            raise ValueError("calibrate parameter must be set")

        if self.calibration_samples >= self._max_samples:
            return self._camera

        frame = self.capture()

        if self._last_timestamp is None:
            self._last_timestamp = frame.timestamp

        if (frame.timestamp - self._last_timestamp).total_seconds() > self._frame_delay:
            ret, corners = frame.images[0].features
            if ret is True:
                self.objpoints.append(self.objp)
                self.imgpoints.append(corners)

                self.calibration_samples += 1
                self._last_timestamp = frame.timestamp

        if self.calibration_samples >= self._max_samples:
            img = frame.images[0].image
            shape = img.shape[::-1]
            self._camera = self._finish_calibration(self.objpoints, self.imgpoints, shape)
            return self._camera

    def _finish_calibration(self, objpoints, imgpoints, shape):
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

            return PinholeCamera(shape, mtx, dist)
