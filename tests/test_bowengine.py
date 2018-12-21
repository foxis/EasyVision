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


camera = PinholeCamera.from_parameters((1241.0, 376.0), (718.8560, 718.8560), (607.1928, 185.2157), [0.0, 0.0, 0.0, 0.0, 0.0])
camera1 = PinholeCamera.from_parameters((1920, 1080), (1920/2, 1080/2), (1920/2, 1080/2), [0.0, 0.0, 0.0, 0.0, 0.0])
camera2 = PinholeCamera.from_parameters((1280, 1024),
    (1280 * 0.535719308086809, 1024 * 0.669566858850269),
    (1280 * 0.493248545285398, 1024 * 0.500408664348414),
    [0.897966326944875 , 0.0, 0.0, 0.0, 0.0])

NUM_IMAGES = 159 # 1
pose = "00"
images = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in xrange(NUM_IMAGES)]
gt_path = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)


@mark.complex
def test_build_vocabulary():
    with CalibratedCamera(
        ImageTransform(
            ImagesVision(images, img_args=()),
            ocl=False, color=cv2.COLOR_BGR2GRAY, enabled=True),
        camera, display_results=True, enabled=True) as cam:
        with BOWVocabularyBuilderEngine(cam, clusters=70, display_results=True, debug=False, feature_type='SIFT') as engine:
            for frame in engine:
                cv2.waitKey(1)
            voc = engine.vocabulary

            cv2.waitKey(0)