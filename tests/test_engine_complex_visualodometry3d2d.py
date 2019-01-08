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


@mark.complex
def test_visual_odometry_3d2d_kitti():
    common_test_visual_odometry_kitti('FREAK', mp=False, ocl=False, debug=False, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine)


@mark.complex
def test_visual_odometry_3d2d_kitti_debug():
    common_test_visual_odometry_kitti('FREAK', mp=True, ocl=False, debug=True, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine, pose="00")
