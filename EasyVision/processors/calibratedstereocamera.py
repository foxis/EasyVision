# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *
from EasyVision.exceptions import TimeoutError
from .calibratedcamera import PinholeCamera
import threading as mt


class StereoCamera(namedtuple('StereoCamera', ['left', 'right', 'R', 'T', 'E', 'F', 'Q'])):
    """

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

        R = np.array(R) if isinstance(R, list) else R
        T = np.array(T) if isinstance(T, list) else T
        E = np.array(E) if isinstance(E, list) else E
        F = np.array(F) if isinstance(F, list) else F
        Q = np.array(Q) if isinstance(Q, list) else Q
        return super(StereoCamera, cls).__new__(cls, left, right, R, T, E, F, Q)

    @staticmethod
    def fromdict(as_dict):
        left = PinholeCamera.fromdict(as_dict.pop('left'))
        right = PinholeCamera.fromdict(as_dict.pop('right'))
        return StereoCamera(left, right, **as_dict)

    def todict(self):
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
        return StereoCamera(
            PinholeCamera(size, M1, d1, R1, P1),
            PinholeCamera(size, M2, d2, R2, P2),
            R, T, E, F, Q)


class CaptureThread(mt.Thread):

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

    def __init__(self, left, right, camera, calculate_disparity=True, num_disparities=32, block_size=15,
                 grid_shape=(9, 6), max_samples=20, *args, **kwargs):
        calibrate = camera is None
        if not isinstance(left, ProcessorBase) or not isinstance(right, ProcessorBase) or \
           left.get_source('CalibratedCamera') is None or right.get_source('CalibratedCamera') is None:
            raise TypeError("Left/Right must have CalibratedCamera")
        if not calibrate:
            if not isinstance(camera, StereoCamera) and not (isinstance(camera, tuple) and len(camera) == 6):
                raise TypeError("Camera must be either StereoCamera or tuple with (frame_size, camera_matrix, distortion)")
            self._camera = StereoCamera._make(camera)
            if left.camera != camera.left or right.camera != camera.right:
                raise ValueError("Respective CalibratedCamera.camera must equal Camera.left/Camera.right")
            if left._calibrate or right._calibrate:
                raise ValueError("Left and Right cameras must NOT be set to calibrate mode")
        else:
            if not left._calibrate or not right._calibrate:
                raise ValueError("Left and Right cameras must be set to calibrate mode")
            left._grid_shape = grid_shape
            right._grid_shape = grid_shape
            self._grid_shape = grid_shape
            self._camera = None
            self.stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            self.flags = 0
            #self.flags |= cv2.CALIB_FIX_INTRINSIC
            #self.flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
            #self.flags |= cv2.CALIB_USE_INTRINSIC_GUESS
            #self.flags |= cv2.CALIB_FIX_FOCAL_LENGTH
            self.flags |= cv2.CALIB_FIX_ASPECT_RATIO
            self.flags |= cv2.CALIB_ZERO_TANGENT_DIST
            # self.flags |= cv2.CALIB_RATIONAL_MODEL
            self.flags |= cv2.CALIB_SAME_FOCAL_LENGTH
            # self.flags |= cv2.CALIB_FIX_K3
            # self.flags |= cv2.CALIB_FIX_K4
            # self.flags |= cv2.CALIB_FIX_K5
            self._max_samples = max_samples

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

    def capture(self):
        frame = super(CalibratedStereoCamera, self).capture()
        if frame and self._calculate_disparity:
            disparity = self._stereoBM.compute(frame.images[0].image, frame.images[1].image)
            img = Image(self, disparity)
            frame = frame._replace(images=frame.images + (img,), processor_mask="110")
        return frame

    def process(self, image):
        if self._calibrate:
            # Draw and display the corners
            ret, corners = image.features
            if self.display_results:
                img = cv2.drawChessboardCorners(image.image, self._grid_shape, corners, ret)
                cv2.imshow("Left" if image.source is self._vision._left._vision else "Right", img)
        else:
            # TODO: rectified images
            img = image.image
            if self.display_results:
                cv2.imshow("Left" if image.source is self._vision._left._vision else "Right", img)

        return image

    def calibrate(self):
        if not self._calibrate:
            raise ValueError("calibrate parameter must be set")

        if self.calibration_samples >= self._max_samples:
            return self._camera

        frame = self.capture()
        left = frame.images[0]
        right = frame.images[1]

        ret_l, corners_l = left.features
        ret_r, corners_r = right.features

        if ret_l is True and ret_r is True:
            self.objpoints.append(self.objp)
            self.imgpoints_l.append(corners_l)
            self.imgpoints_r.append(corners_r)

            self.calibration_samples += 1

        if self.calibration_samples >= self._max_samples:
            img_shape = left.image.shape[::-1]
            self._camera = self._finish_calibration(self.objpoints, self.imgpoints_l, self.imgpoints_r, img_shape)
            return self._camera

    def _finish_calibration(self, objpoints, imgpoints_l, imgpoints_r, shape):
        left_camera = self.source._left._finish_calibration(objpoints, imgpoints_l, shape)
        right_camera = self.source._right._finish_calibration(objpoints, imgpoints_r, shape)

        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_l, imgpoints_r,
            left_camera.matrix, left_camera.distortion,
            right_camera.matrix, right_camera.distortion,
            shape,
            criteria=self.stereocalib_criteria, flags=self.flags)

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
