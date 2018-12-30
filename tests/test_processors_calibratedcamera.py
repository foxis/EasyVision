#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2
import numpy as np


images = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]
fp = (535.5289137817749, 535.3112518132406)
cp = (333.9556024187135, 241.22736353333593)
d = [-0.29426670830681295, 0.11409183502487143, 0.0, 0.0, -0.023222122638183858]

M_left =  [[535.5289137817749, 0.0, 333.9556024187135], [0.0, 535.3112518132406, 241.22736353333593], [0.0, 0.0, 1.0]]
d_left =  [[-0.29426670830681295, 0.11409183502487143, 0.0, 0.0, -0.023222122638183858]]

as_dict = {
    "size": (640, 480),
    "matrix": M_left,
    "distortion": d_left
}


def _assert_camera(camera):
    assert(camera.size == (640, 480))
    assert(camera.focal_point[0] == approx(fp[0]))
    assert(camera.center[0] == approx(cp[0]))
    assert(camera.distortion[0][0] == approx(d[0]))
    assert(isinstance(camera.matrix, np.ndarray))
    assert(isinstance(camera.distortion, np.ndarray))


@pytest.mark.main
def test_camera_from_parameters():
    camera = PinholeCamera.from_parameters((640, 480), fp, cp, d)
    _assert_camera(camera)


@pytest.mark.main
def test_camera():
    camera = PinholeCamera((640, 480), M_left, d_left)
    _assert_camera(camera)


@pytest.mark.main
def test_camera_fromdict():
    camera = PinholeCamera.fromdict(as_dict)
    _assert_camera(camera)


@pytest.mark.main
def test_camera_todict():
    camera = PinholeCamera((640, 480), M_left, d_left)
    as_d = camera.todict()

    test_d = {"rectify": None, "projection": None}
    test_d.update(as_dict)

    assert(as_d == test_d)


@mark.slow
def test_calibrate():
    vision = ImagesReader(images)
    with CalibratedCamera(vision, None, max_samples=len(images) - 3) as vision:

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
    camera = PinholeCamera.from_parameters((640, 480), fp, cp, d)

    vision = ImagesReader(images)
    with CalibratedCamera(vision, camera, display_results=True) as vision:

        for frame in vision:
            pass
            cv2.waitKey(0)