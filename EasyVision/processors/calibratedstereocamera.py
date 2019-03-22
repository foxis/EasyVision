# -*- coding: utf-8 -*-
"""Implements stereo camera calibration and rectify/undistort with a pair of CalibratedCamera objects.

"""

import cv2
import numpy as np
from .base import *
from EasyVision.exceptions import TimeoutError
from .calibratedcamera import PinholeCamera
from EasyVision.vision import PyroCapture
import threading as mt


class StereoCamera(namedtuple('StereoCamera', 'left right R T E F Q')):
    """Stereo camera model that contains two pinhole cameras and transformation matrices between them.

    Has these properties:
        left - left camera intrinsics
        right - right camera intrinsics
        R - rotation matrix
        T - translation vector
        E - essential matrix
        F - fundamental matrix
        Q - disparity matrix

    """
    def __new__(cls, left, right, R, T, E, F, Q):
        if not isinstance(left, PinholeCamera):
            raise ValueError("Left camera must be PinholeCamera")
        if not isinstance(right, PinholeCamera):
            raise ValueError("Right camera must be PinholeCamera")
        if left.size != right.size:
            raise ValueError("Left and Right camera width/height must match")

        R = np.float64(R) if not isinstance(R, np.ndarray) and R is not None else R
        T = np.float64(T) if not isinstance(T, np.ndarray) and T is not None else T
        E = np.float64(E) if not isinstance(E, np.ndarray) and E is not None else E
        F = np.float64(F) if not isinstance(F, np.ndarray) and F is not None else F
        Q = np.float64(Q) if not isinstance(Q, np.ndarray) and Q is not None else Q
        return super(StereoCamera, cls).__new__(cls, left, right, R, T, E, F, Q)

    @staticmethod
    def fromdict(as_dict):
        """Creates StereoCamera from a dict"""
        left = PinholeCamera.fromdict(as_dict.pop('left'))
        right = PinholeCamera.fromdict(as_dict.pop('right'))
        return StereoCamera(left, right, **as_dict)

    def todict(self):
        """Converts StereoCamera into a dict"""
        d = {
            "left": self.left.todict(),
            "right": self.right.todict(),
            "R": self.R.tolist(),
            "T": self.T.tolist(),
            "E": self.E.tolist(),
            "F": self.F.tolist(),
            "Q": self.Q.tolist(),
        }
        return d

    @staticmethod
    def from_parameters(size, M1, d1, R1, P1, M2, d2, R2, P2, R, T, E, F, Q):
        """Creates StereoCamera from parameters

        :param size: Frame size tuple (width, height)
        :param M1: Left camera matrix
        :param d1: Left camera distortion coefficients
        :param R1: Left camera rectification matrix
        :param P1: Left camera projection matrix
        :param M2: Right camera matrix
        :param d2: Right camera distortion coefficients
        :param R2: Right camera rectification matrix
        :param P2: Right camera Projection matrix
        :param R: Right camera rotation matrix
        :param T: Right camera translation vector
        :param E: Essential matrix
        :param F: Fundamental matrix
        :param Q: Disparity matrix
        :return: StereoCamera
        """
        return StereoCamera(
            PinholeCamera(size, M1, d1, R1, P1),
            PinholeCamera(size, M2, d2, R2, P2),
            R, T, E, F, Q)


class CaptureThread(mt.Thread):
    """Capturing thread for a camera pair

    TODO: Synchronization
    """

    def __init__(self, vision):
        super(CaptureThread, self).__init__()
        self._vision = vision
        self._running = False
        self._capture = mt.Event()
        self._ready = mt.Event()
        self.frame = None
        self._ready.clear()
        self._capture.clear()

    def capture_prepare(self):
        self._ready.clear()
        self._capture.set()

    def capture_finish(self):
        if not self._running:
            return None
        self._ready.wait()
        return self.frame

    def __getattr__(self, name):
        return getattr(self._vision, name)

    def run(self):
        self._running = True

        try:
            self._ready.set()

            while True:
                if self._capture.wait():
                    if not self._running:
                        break
                    self.frame = self._vision.capture()
                    self._capture.clear()
                    self._ready.set()
        except:
            raise
        finally:
            self._running = False
            self.frame = None
            self._ready.set()


class CameraPairProxy(VisionBase):
    """Capturing proxy class

    """

    def __init__(self, _self, left, right):
        self._left = CaptureThread(left)
        self._right = CaptureThread(right)
        self._self = _self
        super(CameraPairProxy, self).__init__()

    def setup(self):
        self._left._vision.setup()
        self._right._vision.setup()
        self._left.start()
        self._right.start()
        if not self._left._ready.wait():
            raise TimeoutError()
        if not self._right._ready.wait():
            raise TimeoutError()
        super(CameraPairProxy, self).setup()

    def release(self):
        self._left._running = False
        self._right._running = False
        self._left._capture.set()
        self._right._capture.set()
        self._left.join()
        self._right.join()
        self._left._vision.release()
        self._right._vision.release()
        super(CameraPairProxy, self).release()

    def capture(self):
        super(CameraPairProxy, self).capture()
        self._left.capture_prepare()
        self._right.capture_prepare()
        left, right = self._left.capture_finish(), self._right.capture_finish()

        if left is None or right is None:
            return None

        return left._replace(images=left.images + right.images)

    def get_source(self, source):
        return self._left.get_source(source), self._right.get_source(source)

    def __getattr__(self, name):
        return getattr(self._left, name), getattr(self._right, name)

    @property
    def is_open(self):
        return self._left.is_open and self._right.is_open

    @property
    def frame_size(self):
        return self._left.frame_size

    @property
    def fps(self):
        return self._left.fps

    @property
    def name(self):
        return "({} : {})".format(self._left._vision.name, self._right._vision.name)

    @property
    def frame_count(self):
        return self._left.frame_count

    @property
    def path(self):
        return "{} : {}".format(self._left.path, self._right.path)

    @property
    def description(self):
        return "Stereo Pair Vision Proxy"

    @property
    def devices(self):
        return self._left.devices

    @property
    def autoexposure(self):
        return self._left.autoexposure, self._right.autoexposure

    @property
    def autofocus(self):
        return self._left.autofocus, self._right.autofocus

    @property
    def autowhitebalance(self):
        return self._left.autowhitebalance, self._right.autowhitebalance

    @property
    def autogain(self):
        return self._left.autogain, self._right.autogain

    @property
    def exposure(self):
        return self._left.exposure, self._right.exposure

    @property
    def focus(self):
        return self._left.focus, self._right.focus

    @property
    def whitebalance(self):
        return self._left.whitebalance, self._right.whitebalance

    @property
    def gain(self):
        return self._left.gain, self._right.gain

    @autoexposure.setter
    def autoexposure(self, value):
        self._left.autoexposure = value
        self._right.autoexposure = value

    @autofocus.setter
    def autofocus(self, value):
        self._left.autofocus = value
        self._right.autofocus = value

    @autowhitebalance.setter
    def autowhitebalance(self, value):
        self._left.autowhitebalance = value
        self._right.autowhitebalance = value

    @autogain.setter
    def autogain(self, value):
        self._left.autogain = value
        self._right.autogain = value

    @exposure.setter
    def exposure(self, value):
        self._left.exposure = value
        self._right.exposure = value

    @focus.setter
    def focus(self, value):
        self._left.focus = value
        self._right.focus = value

    @whitebalance.setter
    def whitebalance(self, value):
        self._left.whitebalance = value
        self._right.whitebalance = value

    @gain.setter
    def gain(self, value):
        self._left.gain = value
        self._right.gain = value


class CalibratedStereoCamera(ProcessorBase):
    """Implements calibrated stereo camera calibration, rectification/undistort in conjuction with CalibratedCamera"""

    def __init__(self, left, right, camera=None, calculate_disparity=False, num_disparities=255, block_size=15,
                 grid_shape=(9, 6), square_size=20, max_samples=20, frame_delay=1, *args, **kwargs):
        """CalibratedStereoCamera instance initialization

        :param left: Left camera capturing source
        :param right: Right camera capturing source
        :param camera: StereoCamera object
        :param calculate_disparity: flag indicating whether to calculate disparity map from stereo
        :param num_disparities: Disparity map calculation parameter
        :param block_size: Disparity map calculation parameter
        :param grid_shape: Calibration grid shape
        :param square_size: Calibration grid element size e.g. in mm.
        :param max_samples: number of samples to capture for calibration
        :param frame_delay: number of frames to skip
        """

        calibrate = camera is None
        if (not isinstance(left, ProcessorBase) and not isinstance(left, PyroCapture))or \
                (not isinstance(right, ProcessorBase) and not isinstance(right, PyroCapture)) or \
                left.get_source('CalibratedCamera') is None or right.get_source('CalibratedCamera') is None:
            raise TypeError("Left/Right must have CalibratedCamera")
        if not calibrate:
            if not isinstance(camera, StereoCamera) and not (isinstance(camera, tuple) and len(camera) == 6):
                raise TypeError("Camera must be either StereoCamera or tuple with (frame_size, camera_matrix, distortion)")
            self._camera = camera

            if not isinstance(left, PyroCapture):
                left.get_source('CalibratedCamera').camera = camera.left
            else:
                left.remote_set('camera', camera.left)
            if not isinstance(right, PyroCapture):
                right.get_source('CalibratedCamera').camera = camera.right
            else:
                right.remote_set('camera', camera.right)

            if left._calibrate or right._calibrate:
                raise ValueError("Left and Right cameras must NOT be set to calibrate mode")
        else:
            if not left._calibrate or not right._calibrate:
                raise ValueError("Left and Right cameras must be set to calibrate mode")

            if left.get_source('FeatureExtraction'):
                if not isinstance(left, PyroCapture):
                    left.get_source('FeatureExtraction').enabled = False
                else:
                    left.remote_set('enabled', False)
                if not isinstance(right, PyroCapture):
                    right.get_source('FeatureExtraction').enabled = False
                else:
                    right.remote_set('enabled', False)

            if not isinstance(left, PyroCapture):
                left._grid_shape = grid_shape
                left._square_size = square_size
            else:
                left.remote_set('_grid_shape', grid_shape)
                left.remote_set('_square_size', square_size)

            if not isinstance(right, PyroCapture):
                right._grid_shape = grid_shape
                right._square_size = square_size
            else:
                right.remote_set('_grid_shape', grid_shape)
                right.remote_set('_square_size', square_size)

            self._frame_delay = frame_delay
            self._grid_shape = grid_shape
            self._square_size = square_size
            self._camera = None
            self._stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            self._flags = 0
            #self.flags |= cv2.CALIB_FIX_INTRINSIC
            self._flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            #self.flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            self._flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            self._flags |= cv2.CALIB_FIX_ASPECT_RATIO
            self._flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # self.flags |= cv2.CALIB_RATIONAL_MODEL
            self._flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            self._flags |= cv2.CALIB_FIX_K3
            self._flags |= cv2.CALIB_FIX_K4
            self._flags |= cv2.CALIB_FIX_K5
            self._max_samples = max_samples
            self._last_timestamp = None

        vision = CameraPairProxy(self, left, right)

        self._calibrate = calibrate
        self._calculate_disparity = calculate_disparity
        self._num_disparities = num_disparities
        self._block_size = block_size
        super(CalibratedStereoCamera, self).__init__(vision, *args, **kwargs)

    def setup(self):
        if self._calibrate:
            self.objp = np.zeros((np.prod(self._grid_shape), 3), np.float32)
            self.objp[:, :2] = np.indices(self._grid_shape).T.reshape(-1, 2)
            self.objp *= self._square_size

            self.objpoints = []
            self.imgpoints_l = []
            self.imgpoints_r = []
            self.calibration_samples = 0
        if self._calculate_disparity:
            self._stereoBM = cv2.StereoBM_create(self._num_disparities, self._block_size)
        super(CalibratedStereoCamera, self).setup()

    @property
    def description(self):
        return "Stereo Camera rectify processor"

    @property
    def camera(self):
        return self._camera

    @camera.setter
    def camera(self, value):
        if not isinstance(value, StereoCamera):
            raise TypeError("Must be StereoCamera")
        self._camera = value
        self.source._left.camera = value.left
        self.source._right.camera = value.right

    def capture(self):
        frame = super(CalibratedStereoCamera, self).capture()
        if frame and self._calculate_disparity and not self._calibrate:
            try:
                left = cv2.cvtColor(frame.images[0].image, cv2.COLOR_BGR2GRAY)
                right = cv2.cvtColor(frame.images[1].image, cv2.COLOR_BGR2GRAY)
            except cv2.error:
                left = frame.images[0].image
                right = frame.images[1].image
            disparity = self._stereoBM.compute(left, right)
            if self.display_results:
                disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imshow("Disparity", disp)
            img = Image(self, disparity)
            frame = frame._replace(images=frame.images + (img,), processor_mask="110")
        return frame

    def process(self, image):
        if self._calibrate:
            # Draw and display the corners
            ret, corners = image.features
            if self.display_results:
                img = cv2.drawChessboardCorners(image.image, self._grid_shape, corners, ret)
                cv2.putText(img, "Samples added: {}/{}".format(self.calibration_samples, self._max_samples),
                            (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, 8)
                cv2.imshow("Left" if image.source is self._vision._left._vision else "Right", img)
        else:
            # TODO: rectified images
            img = image.image
            if self.display_results:
                cv2.imshow("Left" if image.source is self._vision._left._vision else "Right", img)
            print (image.source, self._vision._left._vision)

        return image

    def calibrate(self):
        """Use this method for calibration instead of ``capture``. See ``CalibratedCamera`` as the usage is the same."""
        if not self._calibrate:
            raise ValueError("calibrate parameter must be set")

        if self.calibration_samples >= self._max_samples:
            return self._camera

        frame = self.capture()
        left = frame.images[0]
        right = frame.images[1]

        ret_l, corners_l = left.features
        ret_r, corners_r = right.features

        if self._last_timestamp is None:
            self._last_timestamp = frame.timestamp

        if ret_l is True and ret_r is True and (frame.timestamp - self._last_timestamp).total_seconds() > self._frame_delay:
            self.objpoints.append(self.objp)
            self.imgpoints_l.append(corners_l)
            self.imgpoints_r.append(corners_r)

            self.calibration_samples += 1
            self._last_timestamp = frame.timestamp

        if self.calibration_samples >= self._max_samples:
            img_shape = left.image.shape[::-1]
            self._camera = self._finish_calibration(self.objpoints, self.imgpoints_l, self.imgpoints_r, img_shape)
            return self._camera

    def _finish_calibration(self, objpoints, imgpoints_l, imgpoints_r, shape):
        """Helper method that is factored out in the same spirit as in ``CalibratedCamera``"""
        left_camera = self.source._left._finish_calibration(objpoints, imgpoints_l, shape)
        right_camera = self.source._right._finish_calibration(objpoints, imgpoints_r, shape)

        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l, imgpoints_r,
            left_camera.matrix, left_camera.distortion,
            right_camera.matrix, right_camera.distortion,
            shape,
            criteria=self._stereocalib_criteria, flags=self._flags)

        R1, R2, P1, P2, Q, vb1, vb2 = cv2.stereoRectify(
            M1,
            d1,
            M2,
            d2,
            shape,
            R,
            T,
            flags=cv2.CALIB_ZERO_DISPARITY)

        left_camera = PinholeCamera(shape, M1, d1, R1, P1)
        right_camera = PinholeCamera(shape, M2, d2, R2, P2)

        return StereoCamera(left_camera, right_camera, R, T, E, F, Q)
