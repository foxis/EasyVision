#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2


images = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]


def test_calibrate():
    vision = ImagesReader(images)
    with ImageTransform(vision, ocl=True, color=cv2.COLOR_BGR2GRAY, operator=None) as vision:
        for i in vision:
            pass
