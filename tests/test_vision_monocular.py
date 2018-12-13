#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision import *

def test_monocular_vision():
    vision = MonocularVision(0)


def test_monocular_vision_devicenotfound():
    with raises(DeviceNotFound):
        _ = MonocularVision(999)


def test_monocular_vision_capture():
    vision = MonocularVision(0)

    img = vision.capture()
    assert(isinstance(img, Frame))


def test_monocular_vision_capture_debug():
    vision = MonocularVision(0)
    vision.debug = True
    img = vision.capture()
