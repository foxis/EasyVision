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
from tests.common import *


@mark.main
def test_visual_odometry_3d2d():
    FEATURE_TYPE = 'ORB'

    images_kitti_l = ['test_data/kitti00/image_0/{}.png'.format(str(i).zfill(6)) for i in range(3)]

    cam_left = CalibratedCamera(ImageTransform(ImagesReader(images_kitti_l), ocl=False, color=cv2.COLOR_BGR2GRAY), camera_kitti)
    with VisualOdometry3D2DEngine(cam_left, display_results=False, debug=False, feature_type=FEATURE_TYPE) as engine:
        for i, _ in enumerate(engine):
            if i > 1:
                break


@mark.complex
def test_visual_odometry_3d2d_kitti():
    common_test_visual_odometry_kitti('FREAK', mp=False, ocl=True, debug=False, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine)


@mark.complex
def test_visual_odometry_3d2d_debug_kitti():
    common_test_visual_odometry_kitti('FREAK', mp=True, ocl=False, debug=True, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine, pose="00")


if __name__ == "__main__":
    common_test_visual_odometry_kitti('FREAK', mp=False, ocl=True, debug=False, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine)
