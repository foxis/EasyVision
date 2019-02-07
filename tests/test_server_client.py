#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from common import *
from EasyVision.server import Server
import threading as mt
import time


def create_server(images, camera, name):
    """This test requires Pyro NameServer started"""

    vision = ImagesReader(images)
    cam = CalibratedCamera(vision, camera)
    features = FeatureExtraction(cam, feature_type='ORB')

    server = Server(name, features)

    t = mt.Thread(target=server.run)
    t.daemon = True
    t.start()
    time.sleep(1)

    return server


def __test_helper(callable):
    server = create_server(images_left, left_camera, 'LeftCamera')

    try:
        cap = PyroCapture('LeftCamera')

        callable(cap)
    except:
        raise
    finally:
        server.stop()


@mark.slow
def test_server_client():
    def callable(cap):
        with cap as vision:
            assert(vision.name == "LeftCamera @ FeatureExtraction <- CalibratedCamera <- images")
            assert(isinstance(vision.camera, PinholeCamera))
            assert(vision.get_source('ImagesReader') == "ImagesReader")
            assert(vision.get_source('CalibratedCamera') == "CalibratedCamera")
            assert(vision.feature_type == "ORB")

            with raises(AttributeError):
                vision.bla_bla_bla

            assert(vision.process(ImagesReader.load_image(images_left[0])) is not None)

            for idx, img in enumerate(vision):
                assert(isinstance(img, Frame))
                assert(idx < len(images_left) + 1)
                #assert(idx < 10)
    __test_helper(callable)


@mark.slow
def test_server_client_vo2d():
    def callable(cap):
        with VisualOdometry2DEngine(cap, display_results=False, debug=False, feature_type=None) as engine:
            for idx, framepose in enumerate(engine):
                assert(idx < len(images_left) + 1)
    __test_helper(callable)


@mark.slow
def test_server_client_vo3d2d():
    def callable(cap):
        with VisualOdometry3D2DEngine(cap, display_results=False, debug=False, feature_type=None) as engine:
            for idx, framepose in enumerate(engine):
                assert(idx < len(images_left) + 1)
    __test_helper(callable)


@mark.slow
def test_server_client_vostereo():
    images_kitti_l = ['test_data/kitti00/image_0/{}.png'.format(str(i).zfill(6)) for i in xrange(3)]
    images_kitti_r = ['test_data/kitti00/image_1/{}.png'.format(str(i).zfill(6)) for i in xrange(3)]

    server_left = create_server(images_kitti_l, None, 'LeftCamera')
    server_right = create_server(images_kitti_r, None, 'RightCamera')

    try:
        left = PyroCapture('LeftCamera')
        right = PyroCapture('RightCamera')

        camera = StereoCamera(camera_kitti, camera_kitti_right, R_kitti, T_kitti, None, None, None)
        cap = CalibratedStereoCamera(left, right, camera)
        with VisualOdometryStereoEngine(cap, display_results=False, debug=False, feature_type=None) as engine:
            for idx, framepose in enumerate(engine):
                assert (idx < len(images_left) + 1)
    except:
        raise
    finally:
        server_left.stop()
        server_right.stop()



