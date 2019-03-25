# -*- coding: utf-8 -*-
"""Implements Calibrated Camera processor. Uses camera intrinsic parameters transforms the image.

"""

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
        matrix = np.float64(matrix) if not isinstance(matrix, np.ndarray) and matrix is not None else matrix
        distortion = np.float64(distortion) if not isinstance(distortion, np.ndarray) and distortion is not None else distortion
        rectify = np.float64(rectify) if not isinstance(rectify, np.ndarray) and rectify is not None else rectify
        projection = np.float64(projection) if not isinstance(projection, np.ndarray) and projection is not None else projection

        return super(PinholeCamera, cls).__new__(cls, size, matrix, distortion, rectify, projection)

    @property
    def width(self):
        """Width of the camera sensor in pixels"""
        return self.size[0]

    @property
    def height(self):
        """Height of the camera sensor in pixels"""
        return self.size[1]

    @property
    def focal_point(self):
        """Focal point tuple in pixels"""
        return (self.matrix[0, 0], self.matrix[1, 1])

    @property
    def center(self):
        """Image center in pixels"""
        return (self.matrix[0, 2], self.matrix[1, 2])

    @staticmethod
    def fromdict(as_dict):
        """Creates camera object from dict"""
        return PinholeCamera(**as_dict)

    def todict(self):
        """Converts camera object to dict"""
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
        """Creates camera object from parameters

        :param frame_size: tuple containing (frame_width, frame_height)
        :param focal_point: tuple containing (focal_x, focal_y)
        :param center: tuple containing (center_x, center_y)
        :param distortion: distortion coefficients
        :param rectify: rectification 3x3 matrix
        :param projection: projection 3x3 matrix
        :return: PinholeCamera object
        """
        if len(distortion) != 5:
            raise ValueError("distortion must be vector of length 5")
        if len(frame_size) != 2:
            raise ValueError("frame size must be vector of length 2")
        if len(focal_point) != 2:
            raise ValueError("focal point must be vector of length 2")
        if len(center) != 2:
            raise ValueError("center must be vector of length 2")
        matrix = np.zeros((3, 3), np.float64)
        matrix[0, 0] = focal_point[0]
        matrix[1, 1] = focal_point[1]
        matrix[0, 2] = center[0]
        matrix[1, 2] = center[1]
        matrix[2, 2] = 1
        d = np.zeros((1, 5), np.float64)
        d[0] = distortion
        return PinholeCamera(frame_size, matrix, d, rectify, projection)


class CalibratedCamera(ProcessorBase):
    """Class implementing Calibrated Camera undistort/rectify and calibration using rectangular calibration pattern.

    """

    def __init__(self, vision, camera, grid_shape=(7, 6), square_size=20, max_samples=20, frame_delay=1, *args, **kwargs):
        """CalibratedCamera instance initialization

        :param vision: source vision object
        :param camera: either PinholeCamera object or None. If None is specified - then it will enter calibration mode.
        :param grid_shape: shape of the calibration pattern.
        :param square_size: size of the calibration pattern element e.g. in mm.
        :param max_samples: number of samples to collect for calibration
        :param frame_delay: how many frames to skip. Useful online calibration using camera.
        """

        calibrate = camera is None
        if not calibrate:
            if not isinstance(camera, PinholeCamera) and not (isinstance(camera, tuple) and len(camera) == 3):
                raise TypeError("Camera must be either PinholeCamera or tuple with (frame_size, camera_matrix, distortion)")
            self._camera = camera
        else:
            self._criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            self._camera = None
            self._grid_shape = grid_shape
            self._square_size = square_size
            self._max_samples = max_samples
            self._frame_delay = frame_delay
            self._last_timestamp = None

        self._cache_mapped = None
        self._calibrate = calibrate
        self.__setup_called = False
        super(CalibratedCamera, self).__init__(vision, *args, **kwargs)

    def __setup(self):
        if self._calibrate:
            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self._objp = np.zeros((np.prod(self._grid_shape), 3), np.float32)
            self._objp[:, :2] = np.indices(self._grid_shape).T.reshape(-1, 2)
            self._objp *= self._square_size

            # Arrays to store object points and image points from all the images.
            self._objpoints = []  # 3d point in real world space
            self._imgpoints = []  # 2d points in image plane.
            self._calibration_samples = 0
        else:
            self._mapx, self._mapy = cv2.initUndistortRectifyMap(
                    self.camera.matrix,
                    self.camera.distortion,
                    self.camera.rectify,
                    self.camera.projection,
                    self.camera.size,
                    cv2.CV_32FC1)

    def setup(self):
        self.__setup()
        self.__setup_called = True
        super(CalibratedCamera, self).setup()

    @property
    def description(self):
        return "Pinhole camera undistort processor"

    @property
    def camera(self):
        """Get/Set PinholeCamera object. Setting camera will disable calibration mode."""
        return self._camera

    @camera.setter
    def camera(self, value):
        if not isinstance(value, PinholeCamera):
            raise TypeError("Must be PinholeCamera")
        self._camera = value
        self._calibrate = False
        if self.__setup_called:
            self.__setup()

    def process(self, image):
        if self._calibrate:
            img = image.image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self._grid_shape, None)
            if ret is True:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), self._criteria)

            # Draw and display the corners
            if self.display_results:
                img = cv2.drawChessboardCorners(img, self._grid_shape, corners, ret)
                cv2.putText(img, "Samples added: {}/{}".format(self._calibration_samples, self._max_samples),
                            (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
                cv2.imshow(self.name, img)

            return Image(self, gray, features=(ret, corners), feature_type='corners')
        else:
            mapped = cv2.remap(image.image, self._mapx, self._mapy, cv2.INTER_NEAREST, dst=self._cache_mapped)

            if self.display_results:
                cv2.imshow(self.name, mapped)

            self._cache_mapped = image.image
            return image._replace(image=mapped)

    def calibrate(self):
        """Calibration method to be used instead of ``capture`` used for camera calibration."""
        if not self._calibrate:
            raise ValueError("calibrate parameter must be set")

        if self._calibration_samples >= self._max_samples:
            return self._camera

        frame = self.capture()

        if self._last_timestamp is None:
            self._last_timestamp = frame.timestamp

        if (frame.timestamp - self._last_timestamp).total_seconds() > self._frame_delay:
            ret, corners = frame.images[0].features
            if ret is True:
                self._objpoints.append(self._objp)
                self._imgpoints.append(corners)

                self._calibration_samples += 1
                self._last_timestamp = frame.timestamp

        if self._calibration_samples >= self._max_samples:
            img = frame.images[0].image
            shape = img.shape[::-1]
            self._camera = self._finish_calibration(self._objpoints, self._imgpoints, shape)
            return self._camera

    def _finish_calibration(self, objpoints, imgpoints, shape):
        """Helper method that executes camera calibration algorithm. Factored out specifically for ``CalibratedStereoCamera``"""
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, shape, None, None)

        return PinholeCamera(shape, mtx, dist)
