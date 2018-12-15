#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
import cv2

@mark.slow
def test_load_images():
    with ImagesVision(["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)

        assert(frame_count == 3)

@mark.long
def test_load_and_display_images():
    images = ["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]
    with ImagesVision(images, display_results=True) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(frame.images[0].image is not None)
            cv2.waitKey(0)

        assert(frame_count == 3)
