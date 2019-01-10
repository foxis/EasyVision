#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import HistogramBackprojection
import cv2
import numpy as np


@mark.complex
def test_histogram_backprojection():
    images = ["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"]

    subject = cv2.imread('test_data/4472701625_6b23da9a23_b_crop1_masked.jpg')
    histogram = HistogramBackprojection.calculate_histogram(subject)

    with HistogramBackprojection(ImagesReader(images), histogram, display_results=True) as vision:
        frame_count = 0
        for frame in vision:
            frame_count += 1
            assert(isinstance(frame, Frame))
            print frame.images[0].image.__class__
            assert(isinstance(frame.images[0].image, np.ndarray))
            assert(isinstance(frame.images[0].original, np.ndarray))
            assert(isinstance(frame.images[0].mask, np.ndarray))
            cv2.waitKey(0)

        assert(frame_count == 3)

