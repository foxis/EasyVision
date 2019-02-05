#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2


images = ["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]


@mark.slow
def test_load_images():
    vision = ImagesReader(images)
    with FeatureExtraction(vision, 'ORB') as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)

        assert(frame_count == 3)


@mark.slow
def test_load_image():
    image = ImagesReader.load_image(images[0])
    vision = ImagesReader(images)
    with FeatureExtraction(vision, 'ORB') as vision:
        result = vision.process(image)
        assert(isinstance(result, Image))
        assert(result.image is image.image)
        assert(hasattr(result.features, 'points'))
        assert(hasattr(result.features, 'descriptors'))


@mark.complex
def test_load_and_display_images():
    vision = ImagesReader(images, display_results=True)
    with FeatureExtraction(vision, 'ORB', display_results=True) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)
            cv2.waitKey(0)

        assert(frame_count == 3)
