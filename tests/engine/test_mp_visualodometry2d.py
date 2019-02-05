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


@mark.complex
def test_mp_visual_odometry_kitti():
    common_test_visual_odometry_kitti('ORB', mp=True, ocl=True, color=cv2.COLOR_BGR2GRAY)

