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
from .common import *


@mark.long
def test_stereo_calibrated_mp():
    from datetime import datetime
    from itertools import izip

    camera = StereoCamera(left_camera, right_camera, R, T, E, F, Q)
    feature_type = 'SURF'

    total = 0
    left = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type)
    with left as left_:
            print 'Single'
            for frame in left_:
                delta = (datetime.now() - frame.timestamp).total_seconds()
                print delta
                total += delta

    print 'total = ', total
    total = 0

    left = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type)
    right = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_right), ocl=True), camera.right), feature_type=feature_type)
    with left as left_:
        with right as right_:
            print 'Sequential'
            for framea, frameb in izip(left_, right_):
                delta = (datetime.now() - framea.timestamp).total_seconds()
                print delta
                total += delta

    print 'total = ', total
    total = 0

    left = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type)
    right = FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_right), ocl=True), camera.right), feature_type=feature_type)
    with CalibratedStereoCamera(left, right, camera, display_results=False) as vision:
        print 'Threaded'
        for frame in vision:
            delta = (datetime.now() - frame.timestamp).total_seconds()
            print delta
            total += delta
            if vision.display_results:
                cv2.waitKey(0)

    print 'total = ', total
    total = 0

    left = MultiProcessing(FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_left), ocl=True), camera.left), feature_type=feature_type), freerun=False)
    right = MultiProcessing(FeatureExtraction(CalibratedCamera(ImageTransform(ImagesReader(images_right), ocl=True), camera.right), feature_type=feature_type), freerun=False)
    with CalibratedStereoCamera(left, right, camera, display_results=False) as vision:
        print 'Multiprocessing'
        for frame in vision:
            delta = (datetime.now() - frame.timestamp).total_seconds()
            print delta
            total += delta
            if vision.display_results:
                cv2.waitKey(0)

    print 'total = ', total
