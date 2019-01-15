#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.exceptions import *
from EasyVision.engine import *
from EasyVision.processors import *
from EasyVision.vision import *
import cv2
import numpy as np
from .common import *


@mark.main
def test_visual_odometry_3d2d():
    cam = CalibratedCamera(VideoCapture(dataset_note9), camera_note9, display_results=False, enabled=False)
    with VisualOdometry3D2DEngine(cam, display_results=False, debug=False, feature_type='ORB') as engine:
        for frame, pose in engine:
            break


@mark.complex
def test_visual_odometry_3d2d_kitti():
    common_test_visual_odometry_kitti('FREAK', mp=False, ocl=True, debug=False, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine)


@mark.complex
def test_visual_odometry_3d2d_debug_kitti():
    common_test_visual_odometry_kitti('FREAK', mp=True, ocl=False, debug=True, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine, pose="00")


if __name__ == "__main__":
    common_test_visual_odometry_kitti('FREAK', mp=False, ocl=True, debug=False, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine)
