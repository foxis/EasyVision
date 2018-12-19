# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *


class PinholeCamera(namedtuple('PinholeCamera', ['size', 'matrix', 'distortion'])):

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
    def from_parameters(frame_size, focal_point, center, distortion):
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
        d = np.zeros((1,5), np.float32)
        d[0] = distortion
        return PinholeCamera(frame_size, matrix, d)


class CalibratedCamera(ProcessorBase):
    def __init__(self, vision, camera, calibrate=False, max_samples=20, debug=False, display_results=False, enabled=True, *args, **kwargs):
        if not calibrate:
            if not isinstance(camera, PinholeCamera) and not (isinstance(camera, tuple) and len(camera) == 3):
                raise TypeError("Camera must be either Camera or tuple with (frame_size, camera_matrix, distortion)")
            self._camera = PinholeCamera._make(camera)
        else:
            self._camera = None
            self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

            # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self.objp = np.zeros((6*7,3), np.float32)
            self.objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

            # Arrays to store object points and image points from all the images.
            self.objpoints = [] # 3d point in real world space
            self.imgpoints = [] # 2d points in image plane.
            self.calibration_samples = 0
            self._max_samples = max_samples

        self._calibrate = calibrate
        super(CalibratedCamera, self).__init__(vision, debug=debug, display_results=display_results, enabled=enabled, *args, **kwargs)

    @property
    def description(self):
        return "Pinhole camera undistort processor"

    @property
    def camera(self):
        return self._camera

    def process(self, image):
        undistorted = cv2.undistort(image.image, self._camera.matrix, self._camera.distortion, None, self._camera.matrix)

        if self.display_results:
            cv2.imshow(self.name, undistorted)

        return image._replace(image=undistorted)

    def calibrate(self):
        if not self._calibrate:
            raise ValueError("calibrate parameter must be set")

        if self.calibration_samples >= self._max_samples:
            return self._camera

        frame = self.source.capture()
        img = frame.images[0].image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            self.objpoints.append(self.objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), self.criteria)
            self.imgpoints.append(corners2)

            # Draw and display the corners
            if self.display_results:
                img = cv2.drawChessboardCorners(img, (7, 6), corners2,ret)
                cv2.imshow(self.name, img)
            self.calibration_samples += 1

        if self.calibration_samples >= self._max_samples:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None)

            self._camera = PinholeCamera((gray.shape[1], gray.shape[0]), mtx, dist)
            return self._camera
