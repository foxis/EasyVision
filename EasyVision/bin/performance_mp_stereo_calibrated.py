#!/usr/bin/env python
# -*- coding: utf-8 -*-

from EasyVision.exceptions import *
from EasyVision.engine import *
from EasyVision.processors import *
from EasyVision.vision import *
import cv2
import numpy as np
from tests.processors.test_mp_calibratedstereocamera import test_stereo_calibrated_mp, test_stereo_calibrated_mp_dummy

if __name__ == "__main__":
    test_stereo_calibrated_mp_dummy()
    test_stereo_calibrated_mp()
