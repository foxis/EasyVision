#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from EasyVision.processors import *
import cv2


images = ["test_data/left{:02d}.jpg".format(i + 1) for i in range(14) if i != 9]


@pytest.mark.main
def test_image_transform():
    vision = ImagesReader(images)
    with ImageTransform(vision, ocl=True, color=cv2.COLOR_BGR2GRAY, operator=None) as vision:
        for i, img in enumerate(vision):
            assert(isinstance(img.images[0].image, cv2.UMat))
            if i > 3:
                break


@pytest.mark.main
def test_image_transform_pickle():
    vision = ImagesReader(images)
    with ImageTransform(vision, ocl=True, color=cv2.COLOR_BGR2GRAY, operator=None) as vision:
        for i, img in enumerate(vision):

            assert(isinstance(img.images[0].image, cv2.UMat))

            data = img.tobytes()

            new_img = Frame.frombytes(data)

            assert(not isinstance(new_img.images[0].image, cv2.UMat))

            if i > 3:
                break
