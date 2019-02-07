#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2
import json
import numpy as np
from tests.common import *


@mark.main
def test_stereo_camera():
    camera = StereoCamera(
        left_camera,
        right_camera,
        R,
        T,
        E,
        F,
        Q)
    assert_stereo_camera(camera)


@mark.main
def test_stereo_camera_todict():
    camera = StereoCamera(
        left_camera,
        right_camera,
        R,
        T,
        E,
        F,
        Q)
    assert(as_dict_stereo == camera.todict())


@mark.main
def test_stereo_camera_fromdict():
    camera = StereoCamera.fromdict(as_dict_stereo)
    assert_stereo_camera(camera)


@mark.slow
def test_stereo_calibrate():
    left = CalibratedCamera(ImagesReader(images_left), None)
    right = CalibratedCamera(ImagesReader(images_right), None)
    with CalibratedStereoCamera(left, right, None, max_samples=8, frame_delay=0) as vision:

        for i in range(15):
            cam = vision.calibrate()
            if cam:
                print 'M_left = ', cam.left.matrix.tolist()
                print 'd_left = ', cam.left.distortion.tolist()
                print 'R_left = ', cam.left.rectify.tolist()
                print 'P_left = ', cam.left.projection.tolist()

                print 'M_right = ', cam.right.matrix.tolist()
                print 'd_right = ', cam.right.distortion.tolist()
                print 'R_right = ', cam.right.rectify.tolist()
                print 'P_right = ', cam.right.projection.tolist()

                print 'R = ', cam.R.tolist()
                print 'T = ', cam.T.tolist()
                print 'E = ', cam.E.tolist()
                print 'F = ', cam.F.tolist()
                print 'Q = ', cam.Q.tolist()

                assert(isinstance(cam, StereoCamera))
                assert(isinstance(cam.left, PinholeCamera))
                assert(isinstance(cam.right, PinholeCamera))

                assert(cam.left.size == (640, 480))
                assert(cam.left.focal_point[0] == approx(left_camera.focal_point[0], rel=2))
                assert(cam.left.focal_point[1] == approx(left_camera.focal_point[1], rel=2))
                assert(cam.left.center[0] == approx(left_camera.center[0], rel=2))
                assert(cam.left.center[1] == approx(left_camera.center[1], rel=2))
                for i in range(5):
                    assert(cam.left.distortion[0][i] == approx(left_camera.distortion[0][i], rel=5e-1, abs=5e-1))

                assert(cam.right.size == (640, 480))
                assert(cam.right.focal_point[0] == approx(right_camera.focal_point[0], rel=2))
                assert(cam.right.focal_point[1] == approx(right_camera.focal_point[1], rel=2))
                assert(cam.right.center[0] == approx(right_camera.center[0], rel=2))
                assert(cam.right.center[1] == approx(right_camera.center[1], rel=2))
                for i in range(5):
                    assert(cam.right.distortion[0][i] == approx(right_camera.distortion[0][i], rel=5e-1, abs=5e-1))

                assert(True)
                break
        else:
            assert(False)


@mark.complex
def test_stereo_calibrated():
    from datetime import datetime

    camera = StereoCamera(left_camera, right_camera, R, T, E, F, Q)

    #left = CalibratedCamera(ImagesReader(images_left), None)
    #right = CalibratedCamera(ImagesReader(images_right), None)
    #with CalibratedStereoCamera(left, right, None, max_samples=8, display_results=False) as vision:

    #    for i in range(15):
    #        cam = vision.calibrate()
    #        if cam:
    #            camera = cam
    #            break
    #        if vision.display_results:
    #            cv2.waitKey(0)

    left = CalibratedCamera(ImagesReader(images_left), camera.left)
    right = CalibratedCamera(ImagesReader(images_right), camera.right)
    with CalibratedStereoCamera(left, right, camera, display_results=True) as vision:

        for frame in vision:
            print (datetime.now() - frame.timestamp).total_seconds()
            cv2.waitKey(0)


@mark.main
def test_stereo_properties():
    camera = StereoCamera(left_camera, right_camera, R, T, E, F, Q)
    left = CalibratedCamera(VisionSubclass(), None)
    right = CalibratedCamera(VisionSubclass(), None)
    with CalibratedStereoCamera(left, right, camera) as s:
        assert(left.camera is left_camera)
        assert(right.camera is right_camera)

        assert(not left._calibrate)
        assert(not right._calibrate)

        assert(s.autoexposure == (None, None))
        assert(s.autofocus == (None, None))
        assert(s.autowhitebalance == (None, None))
        assert(s.autogain == (None, None))
        assert(s.exposure == (None, None))
        assert(s.focus == (None, None))
        assert(s.whitebalance == (None, None))

        s.autoexposure = 1
        s.autofocus = 2
        s.autowhitebalance = 3
        s.autogain = 4
        s.exposure = 5
        s.focus = 6
        s.whitebalance = 7

        assert(s.autoexposure == (1, 1))
        assert(s.autofocus == (2, 2))
        assert(s.autowhitebalance == (3, 3))
        assert(s.autogain == (4, 4))
        assert(s.exposure == (5, 5))
        assert(s.focus == (6, 6))
        assert(s.whitebalance == (7, 7))
