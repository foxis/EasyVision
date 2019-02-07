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
import os


camera = PinholeCamera.from_parameters((1241, 376), (718.8560, 718.8560), (607.1928, 185.2157), [0.0, 0.0, 0.0, 0.0, 0.0])
camera1 = PinholeCamera.from_parameters((1920, 1080), (1920/2, 1080/2), (1920/2, 1080/2), [0.0, 0.0, 0.0, 0.0, 0.0])
camera2 = PinholeCamera.from_parameters((1280, 1024),
    (1280 * 0.535719308086809, 1024 * 0.669566858850269),
    (1280 * 0.493248545285398, 1024 * 0.500408664348414),
    [0.897966326944875 , 0.0, 0.0, 0.0, 0.0])

NUM_IMAGES = 159 # 1
pose = "00"
images = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in range(NUM_IMAGES)]
gt_path = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)


def build_vocabulary(path, dbow3, feature_type):
    cam = CalibratedCamera(ImageTransform(ImagesReader(images), ocl=False, color=cv2.COLOR_BGR2GRAY, enabled=True), camera, enabled=True)
    with BOWVocabularyBuilderEngine(cam, clusters=70, feature_type=feature_type, dbow3_trainer=dbow3) as engine:
        for frame in engine:
            pass
        engine.create_vocabulary()
        if path:
            engine.save(path)
            assert(os.path.isfile(path))
            engine.load(path)


@mark.long
def test_build_vocabulary_kmeans():
    build_vocabulary(None, False, 'SIFT')


@mark.long
def test_build_vocabulary_kmeans_orb():
    with raises(NotImplementedError):
        build_vocabulary(None, False, 'ORB')


@mark.long
def test_build_vocabulary_kmeans_save():
    build_vocabulary("test.bow", False, 'SIFT')


@mark.long
def test_build_vocabulary_dbow3():
    build_vocabulary("test.dbow3", True, 'ORB')


@mark.long
def test_dbow3_matching_mixin():
    cam = CalibratedCamera(ImageTransform(ImagesReader(images), ocl=False, color=cv2.COLOR_BGR2GRAY, enabled=True), camera, display_results=True, enabled=True)
    with FeatureExtraction(cam, 'ORB') as vision:
        frame_count = 0

        mixin = BOWMatchingMixin(None, None, "EasyVision/engine/orbvoc.dbow3", vision.feature_type)

        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            assert(frame.images[0].image is not None)

        assert(frame_count == 3)
