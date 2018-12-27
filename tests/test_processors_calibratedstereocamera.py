#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2


images_left = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]
images_right = ["test_data/right{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]

fp_left = (5.3591573396163199e+02, 5.3591573396163199e+02)
cp_left = (3.4228315473308373e+02, 2.3557082909788173e+02)
d_left = [-2.96548154e-01, 1.13868006e-01, 1.44476302e-03, 2.94839956e-04, 4.73668193e-02]

fp_right = (5.3591573396163199e+02, 5.3591573396163199e+02)
cp_right = (3.4228315473308373e+02, 2.3557082909788173e+02)
d_right = [-2.96548154e-01, 1.13868006e-01, 1.44476302e-03, 2.94839956e-04, 4.73668193e-02]

R = []
T = []
E = []
F = []

left_camera = PinholeCamera.from_parameters((640, 480), fp_left, cp_left, d_left)
right_camera = PinholeCamera.from_parameters((640, 480), fp_right, cp_right, d_right)


def test_stereo_camera():
    camera = StereoCamera.from_parameters(
        left_camera,
        right_camera,
        R,
        T,
        E,
        F)
    #assert(camera.size[0] == 640)
    #assert(camera.focal_point[0] == approx(fp[0]))
    #assert(camera.center[0] == approx(cp[0]))
    #assert(camera.distortion[0][0] == approx(d[0]))


@mark.slow
def test_stereo_calibrate():
    left = CalibratedCamera(ImagesReader(images_left), None)
    right = CalibratedCamera(ImagesReader(images_right), None)
    with CalibratedStereoCamera(left, right, None, max_samples=len(images_left) - 3) as vision:

        for i in range(15):
            cam = vision.calibrate()
            if cam:
                print cam
                assert(isinstance(cam, StereoCamera))
                assert(cam.left.size == (640, 480))
                assert(cam.left.focal_point[0] == approx(fp_left[0], rel=2))
                assert(cam.left.focal_point[1] == approx(fp_left[1], rel=2))
                assert(cam.left.center[0] == approx(cp_left[0], rel=2))
                assert(cam.left.center[1] == approx(cp_left[1], rel=2))
                print cam.left.distortion[0]
                for i in range(5):
                    assert(cam.left.distortion[0][i] == approx(d_left[i], rel=5e-1, abs=5e-1))

                assert(cam.right.size == (640, 480))
                assert(cam.right.focal_point[0] == approx(fp_right[0], rel=2))
                assert(cam.right.focal_point[1] == approx(fp_right[1], rel=2))
                assert(cam.right.center[0] == approx(cp_right[0], rel=2))
                assert(cam.right.center[1] == approx(cp_right[1], rel=2))
                print cam.right.distortion[0]
                for i in range(5):
                    assert(cam.right.distortion[0][i] == approx(d_right[i], rel=5e-1, abs=5e-1))
                break
        else:
            assert(False)


@mark.complex
def test_calibrated():
    camera = StereoCamera.from_parameters(left_camera, right_camera, R, T, E, F)

    left = CalibratedCamera(ImagesReader(images_left), camera.left)
    right = CalibratedCamera(ImagesReader(images_right), camera.right)
    with CalibratedStereoCamera(left, right, camera, display_results=True) as vision:

        for frame in vision:
            cv2.waitKey(0)