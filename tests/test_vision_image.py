#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
import cv2
import cPickle


@mark.slow
def test_load_images():
    with ImagesReader(["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)

        assert(frame_count == 3)


@mark.slow
def test_load_image():
    image = ImagesReader.load_image("test_data/34838518832_fd00147042_k.jpg")
    assert(image.source is None)
    assert(image.image is not None)


@mark.slow
def test_load_image_fail():
    with raises(IOError):
        ImagesReader.load_image("no-such-file.jpg")


@mark.long
def test_load_and_display_images():
    images = ["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]
    with ImagesReader(images, display_results=True) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)
            cv2.waitKey(0)

        assert(frame_count == 3)


def test_pickle_image():
    _image = ImagesReader.load_image("test_data/34838518832_fd00147042_k.jpg")
    tmp = cPickle.dumps(_image, -1)
    image = cPickle.loads(tmp)

    assert(image.source is None)
    assert(image.image is not None)
