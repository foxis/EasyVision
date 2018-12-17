#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2


images = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]
fp = (5.3591573396163199e+02, 5.3591573396163199e+02)
cp = (3.4228315473308373e+02, 2.3557082909788173e+02)
d = [-2.96548154e-01, 1.13868006e-01, 1.44476302e-03, 2.94839956e-04, 4.73668193e-02]


def test_camera():
    camera = PinholeCamera.from_parameters(
        (640, 480),
        fp,
        cp,
        d)
    assert(camera.size[0] == 640)
    assert(camera.focal_point[0] == approx(fp[0]))
    assert(camera.center[0] == approx(cp[0]))
    assert(camera.distortion[0][0] == approx(d[0]))


@mark.slow
def test_calibrate():
    vision = ImagesVision(images)
    with CalibratedCamera(vision, None, calibrate=True, max_samples=len(images) - 3) as vision:

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


@mark.long
def test_calibrated():
    camera = PinholeCamera.from_parameters((640, 480), fp, cp, d)

    vision = ImagesVision(images)
    with CalibratedCamera(vision, camera, display_results=True) as vision:

        for frame in vision:
            pass
            cv2.waitKey(0)

    assert(False)