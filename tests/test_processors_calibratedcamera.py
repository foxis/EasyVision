#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2
import numpy as np
from .common import *


@mark.main
def test_camera_from_parameters():
    camera = PinholeCamera.from_parameters((640, 480), fp, cp, d)
    assert_camera(camera)


@mark.main
def test_camera():
    camera = PinholeCamera((640, 480), M_left, d_left)
    assert_camera(camera)


@mark.main
def test_camera_fromdict():
    camera = PinholeCamera.fromdict(as_dict_left)
    assert_camera(camera)


@mark.main
def test_camera_todict():
    camera = PinholeCamera((640, 480), M_left, d_left)
    as_d = camera.todict()

    test_d = {"rectify": None, "projection": None}
    test_d.update(as_dict_left)

    assert(as_d == test_d)


@mark.slow
def test_calibrate():
    vision = ImagesReader(images_left)
    with CalibratedCamera(vision, None, max_samples=len(images_left) - 3, frame_delay=0) as vision:

        for i in range(15):
            cam = vision.calibrate()
            if cam:
                assert(isinstance(cam, PinholeCamera))
                assert(cam.size == (640, 480))
                assert(cam.focal_point[0] == approx(fp[0], rel=2))
                assert(cam.focal_point[1] == approx(fp[1], rel=2))
                assert(cam.center[0] == approx(cp[0], rel=2))
                assert(cam.center[1] == approx(cp[1], rel=2))
                print cam.distortion[0]
                for i in range(5):
                    assert(cam.distortion[0][i] == approx(d[i], rel=5e-1, abs=5e-1))
                break
        else:
            assert(False)


@mark.complex
def test_calibrated():
    vision = ImagesReader(images_left)
    with CalibratedCamera(vision, left_camera, display_results=True) as vision:

        for frame in vision:
            pass
            cv2.waitKey(0)