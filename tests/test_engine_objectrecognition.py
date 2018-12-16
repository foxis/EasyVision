#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.exceptions import *
from EasyVision.engine import *
from EasyVision.processors import FeatureExtraction
from EasyVision.vision import ImagesVision, Frame, Image
import cv2

images = [
    "test_data/34838518832_fd00147042_k.jpg",
    "test_data/2732011028_f0f033e678_b.jpg",
    "test_data/4472701625_6b23da9a23_b.jpg",
    "test_data/13669656965_86a0146858_k.jpg",
    "test_data/21503937795_a9f41f428a_k.jpg"
]
obj1 = "test_data/4472701625_6b23da9a23_b_crop1.jpg"
obj2 = "test_data/4472701625_6b23da9a23_b_crop2.jpg"

@mark.long
def test_match_images():
    with FeatureExtraction(ImagesVision(images), 'ORB', display_results=True) as extractor:
        with ObjectRecognitionEngine(extractor, display_results=True) as engine:
            frame_count = 0

            assert(engine.enroll("obj1", ImagesVision.load_image(obj1), add=True, display_results=True) is not None)
            assert(engine.enroll("obj2", ImagesVision.load_image(obj2), add=True, display_results=True) is not None)
            assert(len(engine.models) == 2)

            for frame, matches in engine:
                frame_count += 1
                assert(isinstance(frame, Frame))
                assert(frame.images[0].image is not None)
                cv2.waitKey(0)

            assert(frame_count == len(images))


@mark.slow
def test_match_images_default_extractor():
    with FeatureExtraction(ImagesVision(images), 'ORB', display_results=False) as extractor:
        with ObjectRecognitionEngine(extractor, display_results=False) as engine:
            frame_count = 0

            assert(engine.enroll("obj1", ImagesVision.load_image(obj1), add=True, display_results=False) is not None)
            assert(engine.enroll("obj2", ImagesVision.load_image(obj2), add=True, display_results=False) is not None)
            assert(len(engine.models) == 2)

            for frame, matches in engine:
                frame_count += 1
                assert(isinstance(frame, Frame))
                assert(frame.images[0].image is not None)

                if frame.index == 3:
                    assert(len(matches) == 2)
                    assert(matches[0].model.name == 'obj1' or matches[0].model.name == 'obj2')
                    assert(matches[1].model.name == 'obj1' or matches[1].model.name == 'obj2')
                else:
                    assert(len(matches) == 0)

            assert(frame_count == len(images))
