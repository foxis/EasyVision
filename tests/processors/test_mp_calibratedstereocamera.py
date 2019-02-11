#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2
import json
import numpy as np
import os
from tests.common import *

try:
    from future_builtins import zip
except:
    pass


class DummySubclass(VisionSubclass):
    def capture(self):
        from datetime import datetime
        from time import sleep
        self.frame += 1
        if self.frame > 10:
            return None
        f = Frame(datetime.now(), self.frame - 1, (Image(self, np.zeros((3 * 1024, 3 * 1024, 3), dtype=np.uint8)),))
        sleep(.01)
        return f


@mark.xfail
def test_stereo_calibrated_mp():
    from datetime import datetime
    from itertools import izip

    camera = StereoCamera(left_camera, right_camera, R, T, E, F, Q)
    feature_type = 'SURF'

    total_single = 0
    left = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type)
    with left as left_:
            print('Single')
            for frame in left_:
                if frame.index == 0:
                    continue
                delta = (datetime.now() - frame.timestamp).total_seconds()
                total_single += delta

    total_sum = 0

    left = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type)
    right = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_right), ocl=True), camera.right), feature_type=feature_type)
    with left as left_:
        with right as right_:
            print('Sequential')
            for framea, frameb in izip(left_, right_):
                if framea.index == 0:
                    continue
                delta = (datetime.now() - framea.timestamp).total_seconds()
                total_sum += delta

    total_th = 0

    left = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type)
    right = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_right), ocl=True), camera.right), feature_type=feature_type)
    with CalibratedStereoCamera(left, right, camera, display_results=False) as vision:
        print('Threaded')
        for frame in vision:
            if frame.index == 0:
                continue
            delta = (datetime.now() - frame.timestamp).total_seconds()
            total_th += delta

    total_mp = 0

    left = MultiProcessing(FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type), freerun=False)
    right = MultiProcessing(FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_right), ocl=True), camera.right), feature_type=feature_type), freerun=False)
    with CalibratedStereoCamera(left, right, camera, display_results=False) as vision:
        print('Multiprocessing')
        for frame in vision:
            if frame.index == 0:
                continue
            delta = (datetime.now() - frame.timestamp).total_seconds()
            total_mp += delta

    print('Single total = ', total_single)
    print('Sequential total = ', total_sum)
    print('Threaded total = ', total_th)
    print('Multiprocessing total = ', total_mp)

    assert(total_sum > total_single)
    #assert(total_single < total_th < total_sum)
    assert(total_single < total_mp < total_sum)


@mark.slow
def test_stereo_calibrated_mp_dummy():
    from datetime import datetime

    camera = StereoCamera(left_camera, right_camera, R, T, E, F, Q)

    total_single = 0
    left = CalibratedCamera(DummySubclass(), camera.left)
    with left as left_:
            for frame in left_:
                delta = (datetime.now() - frame.timestamp).total_seconds()
                total_single += delta

    total_sum = 0
    left = CalibratedCamera(DummySubclass(), camera.left)
    right = CalibratedCamera(DummySubclass(), camera.right)
    with left as left_:
        with right as right_:
            for framea, frameb in zip(left_, right_):
                delta = (datetime.now() - framea.timestamp).total_seconds()
                total_sum += delta

    total_th = 0
    left = CalibratedCamera(DummySubclass(), camera.left)
    right = CalibratedCamera(DummySubclass(), camera.right)
    with CalibratedStereoCamera(left, right, camera, display_results=False) as vision:
        for frame in vision:
            delta = (datetime.now() - frame.timestamp).total_seconds()
            total_th += delta

    total_mp = 0
    left = MultiProcessing(CalibratedCamera(DummySubclass(), camera.left), freerun=False)
    right = MultiProcessing(CalibratedCamera(DummySubclass(), camera.right), freerun=False)
    with CalibratedStereoCamera(left, right, camera, display_results=False) as vision:
        for frame in vision:
            delta = (datetime.now() - frame.timestamp).total_seconds()
            total_mp += delta

    print('Single total = ', total_single)
    print('Sequential total = ', total_sum)
    print('Threaded total = ', total_th)
    print('Multiprocessing total = ', total_mp)

    assert(total_sum > total_single)
    assert(total_single <= total_th < total_sum)
    assert(total_single <= total_mp < total_sum)
