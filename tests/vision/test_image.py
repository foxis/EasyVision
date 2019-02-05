#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
import cv2
import cPickle
import numpy as np


@pytest.mark.main
def test_load_images():
    with ImagesReader(["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]) as vision:
        frame_count = 0
        for frame in vision:
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)
            assert(frame.index == frame_count)
            frame_count += 1

        assert(frame_count == 3)


@pytest.mark.main
def test_load_image():
    image = ImagesReader.load_image("test_data/34838518832_fd00147042_k.jpg")
    assert(image.source is None)
    assert(image.image is not None)


@pytest.mark.main
def test_load_image_fail():
    with raises(IOError):
        ImagesReader.load_image("no-such-file.jpg")


@mark.complex
def test_load_and_display_images():
    images = ["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]
    with ImagesReader(images, display_results=True) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(isinstance(frame.images[0].image, np.ndarray))
            cv2.waitKey(0)

        assert(frame_count == 3)


@pytest.mark.main
def test_images_capture_debug(mocker):
    mocker.patch('cv2.namedWindow', autospec=True)
    mocker.patch('cv2.destroyWindow', autospec=True)
    mocker.patch('cv2.imshow', autospec=True)
    images = ["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]
    image = None
    with ImagesReader(images) as vision:
        vision.display_results = True
        frame = vision.capture()
        assert(isinstance(frame, Frame))
        assert(frame.images[0].source is vision)
        assert(isinstance(frame.images[0].image, np.ndarray))
        image = frame.images[0].image

    name = "images"
    cv2.namedWindow.assert_called_with(name, cv2.WINDOW_NORMAL)
    #cv2.destroyWindow.assert_called_with(name)
    cv2.imshow.assert_called_with(name, image)


@pytest.mark.main
def test_pickle_image():
    _image = ImagesReader.load_image("test_data/34838518832_fd00147042_k.jpg")
    tmp = cPickle.dumps(_image, -1)
    image = cPickle.loads(tmp)

    assert(image.source is None)
    assert(image.image is not None)
