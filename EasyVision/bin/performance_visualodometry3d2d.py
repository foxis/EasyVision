#!/usr/bin/env python
# -*- coding: utf-8 -*-

from EasyVision.exceptions import *
from EasyVision.engine import *
from EasyVision.processors import *
from EasyVision.vision import *
import cv2
import numpy as np
from tests.common import *

if __name__ == "__main__":
    common_test_visual_odometry_kitti('ORB', mp=False, ocl=True, debug=True, color=cv2.COLOR_BGR2GRAY, odometry_class=VisualOdometry3D2DEngine)
