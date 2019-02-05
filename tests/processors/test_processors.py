#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors.base import *
from tests.common import VisionSubclass, ProcessorA, ProcessorB


@pytest.mark.main
def test_abstract():
    with raises(TypeError):
        ProcessorBase()


@pytest.mark.main
def test_implementation():
    source = VisionSubclass()
    pr = ProcessorA(source)
    assert(pr.source is source)


@pytest.mark.main
def test_capture():
    vision = VisionSubclass(0)

    with ProcessorA(vision) as processor:
        img = processor.capture()
        assert(isinstance(img, Frame))
        assert(img.images[0].source is processor)
        assert(img.images[0].image == "AN IMAGE")


@pytest.mark.main
def test_capture_incorrect():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)

    with raises(AssertionError):
        processor.capture()


@pytest.mark.main
def test_capture_stacked_incorrect():
    vision = VisionSubclass("Test")
    processorA = ProcessorA(vision)
    processorB = ProcessorB(processorA)

    assert(processorB.name == "ProcessorB <- ProcessorA <- Test")

    with raises(AssertionError):
        processorB.capture()


@pytest.mark.main
def test_capture_stacked():
    vision = VisionSubclass("Test")
    processorA = ProcessorA(vision)
    processorB = ProcessorB(processorA)

    assert(processorB.name == "ProcessorB <- ProcessorA <- Test")

    with processorB as processor:
        img = processor.capture()
        assert(isinstance(img, Frame))
        assert(img.images[0].source is processorB)
        assert(img.images[0].image == "An Image")
        assert(processorB.get_source('Test') is vision)
        assert(processorB.get_source('ProcessorA') is processorA)
        assert(processorB.get_source('ProcessorB') is processorB)
        assert(processorB.get_source('Test no') is None)


@pytest.mark.main
def test_method_resolution():
    vision = VisionSubclass("Test")
    processorA = ProcessorA(vision)
    processorB = ProcessorB(processorA)

    assert(processorB.name == "ProcessorB <- ProcessorA <- Test")

    assert(not vision.camera_called)
    assert(processorB.camera_())
    assert(processorB.camera_called)
    assert(vision.camera_called)

@pytest.mark.main
def test_processor_properties():
    vision = VisionSubclass("Test")
    processorA = ProcessorA(vision)
    processorB = ProcessorB(processorA)

    with processorB as s:
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
