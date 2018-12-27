#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2
import json


images_left = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]
images_right = ["test_data/right{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]

fp_left = (5.3591573396163199e+02, 5.3591573396163199e+02)
cp_left = (3.4228315473308373e+02, 2.3557082909788173e+02)
d_left = [-2.96548154e-01, 1.13868006e-01, 1.44476302e-03, 2.94839956e-04, 4.73668193e-02]
R1 = [[-0.82862177,  0.55401763,  0.08031456],
       [ 0.5547615 ,  0.79343379,  0.25040468],
       [ 0.07500432,  0.25204619, -0.96480416]]
P1 = [[659.55229104,   0.        , 396.0413723 ,   0.        ],
       [  0.        , 659.55229104, 454.6717453 ,   0.        ],
       [  0.        ,   0.        ,   1.        ,   0.        ]]

fp_right = (5.3591573396163199e+02, 5.3591573396163199e+02)
cp_right = (3.4228315473308373e+02, 2.3557082909788173e+02)
d_right = [-2.96548154e-01, 1.13868006e-01, 1.44476302e-03, 2.94839956e-04, 4.73668193e-02]
R2 = [[-0.83438477,  0.54530701, -0.0802641 ],
       [ 0.51647215,  0.82436576,  0.23168431],
       [ 0.19250605,  0.15185968, -0.96947411]]
P2 = [[ 659.55229104,    0.        ,  396.0413723 ,    0.        ],
       [   0.        ,  659.55229104,  321.73318481, -707.27720521],
       [   0.        ,    0.        ,    1.        ,    0.        ]]

R = [[ 0.97823634,  0.00326694, -0.20746804],
       [ 0.01009763,  0.99794182,  0.06332583],
       [ 0.20724792, -0.06404256,  0.97618997]]
T = [[-0.59230284], [-0.93823973], [ 0.32113015]]
E = [[-0.19769089, -0.26038193, -0.93623604],
       [ 0.43689471, -0.03688348,  0.51157584],
       [ 0.91183934, -0.5880186 , -0.23216283]]
F = [[ 2.76340601e-06,  3.63988036e-06,  5.21131734e-03],
       [-6.11597931e-06,  5.16344218e-07, -1.86772066e-03],
       [-6.26364026e-03,  3.11199783e-03,  1.00000000e+00]]
Q = [[   1.        ,    0.        ,    0.        , -362.92362785],
       [   0.        ,    1.        ,    0.        ,   41.49035454],
       [   0.        ,    0.        ,    0.        ,  536.66794941],
       [   0.        ,    0.        ,    0.86573078,   26.4431621 ]]

left_camera = PinholeCamera.from_parameters((640, 480), fp_left, cp_left, d_left, R1, P1)
right_camera = PinholeCamera.from_parameters((640, 480), fp_right, cp_right, d_right, R2, P2)


def test_stereo_camera():
    camera = StereoCamera(
        left_camera,
        right_camera,
        R,
        T,
        E,
        F,
        Q)
    #assert(camera.size[0] == 640)
    #assert(camera.focal_point[0] == approx(fp[0]))
    #assert(camera.center[0] == approx(cp[0]))
    #assert(camera.distortion[0][0] == approx(d[0]))


@mark.slow
def test_stereo_calibrate():
    left = CalibratedCamera(ImagesReader(images_left), None)
    right = CalibratedCamera(ImagesReader(images_right), None)
    with CalibratedStereoCamera(left, right, None, max_samples=8) as vision:

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
                for i in range(5):
                    assert(cam.left.distortion[0][i] == approx(d_left[i], rel=5e-1, abs=5e-1))

                assert(cam.right.size == (640, 480))
                assert(cam.right.focal_point[0] == approx(fp_right[0], rel=2))
                assert(cam.right.focal_point[1] == approx(fp_right[1], rel=2))
                assert(cam.right.center[0] == approx(cp_right[0], rel=2))
                assert(cam.right.center[1] == approx(cp_right[1], rel=2))
                for i in range(5):
                    assert(cam.right.distortion[0][i] == approx(d_right[i], rel=5e-1, abs=5e-1))

                assert(False)
                break
        else:
            assert(False)


@mark.complex
def test_stereo_calibrated():
    camera = StereoCamera(left_camera, right_camera, R, T, E, F, Q)

    left = CalibratedCamera(ImagesReader(images_left), None)
    right = CalibratedCamera(ImagesReader(images_right), None)
    with CalibratedStereoCamera(left, right, None, max_samples=8, display_results=False) as vision:

        for i in range(15):
            cam = vision.calibrate()
            if cam:
                camera = cam
                break
            if vision.display_results:
                cv2.waitKey(0)

    left = CalibratedCamera(ImagesReader(images_left), camera.left)
    right = CalibratedCamera(ImagesReader(images_right), camera.right)
    with CalibratedStereoCamera(left, right, camera, display_results=True) as vision:

        for frame in vision:
            cv2.waitKey(0)