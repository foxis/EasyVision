#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors import MultiConsumers
from EasyVision.vision import *
from tests.common import VisionSubclass

try:
    from future_builtins import zip
except:
    pass

@pytest.mark.main
def test_capture_multiconsumers():
    vision = VisionSubclass(0)
    processor = MultiConsumers(vision)
    with processor as p1:
        with processor as p2:
            for img1, img2 in zip(p1, p2):
                assert(isinstance(img1, Frame))
                assert(isinstance(img2, Frame))
                assert(img1 == img2)
                if img1.index > 10:
                    break


@pytest.mark.main
def test_capture_multiconsumers_fail():
    vision = VisionSubclass(0)
    processor = MultiConsumers(vision)

    processor.setup()
    processor.setup()

    processor.release()
    processor.release()

    with raises(AssertionError):
        processor.release()


@pytest.mark.main
def test_mc_properties():
    vision = VisionSubclass("Test")

    with MultiConsumers(vision) as s:
        assert(s.autoexposure is None)
        assert(s.autofocus is None)
        assert(s.autowhitebalance is None)
        assert(s.autogain is None)
        assert(s.exposure is None)
        assert(s.focus is None)
        assert(s.whitebalance is None)

        s.autoexposure = 1
        s.autofocus = 2
        s.autowhitebalance = 3
        s.autogain = 4
        s.exposure = 5
        s.focus = 6
        s.whitebalance = 7

        assert(s.autoexposure == 1)
        assert(s.autofocus == 2)
        assert(s.autowhitebalance == 3)
        assert(s.autogain == 4)
        assert(s.exposure == 5)
        assert(s.focus == 6)
        assert(s.whitebalance == 7)
