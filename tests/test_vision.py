#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from .common import VisionSubclass


@pytest.mark.main
def test_abstract_vision_abstract():
    with raises(TypeError):
        VisionBase()


@pytest.mark.main
def test_abstract_vision_implementation():
    VisionSubclass()


@pytest.mark.main
def test_abstract_vision_implementation_nosetup():
    s = VisionSubclass()
    with raises(AssertionError):
        for _ in s:
            break


@pytest.mark.main
def test_abstract_vision_implementation_context():
    s = VisionSubclass()
    with s as ss:
        for _ in ss:
            break


@pytest.mark.main
def test_abstract_vision_implementation_setup():
    s = VisionSubclass()
    s.setup()
    for _ in s:
        break
    s.release()


@pytest.mark.main
def test_abstract_vision_implementation_properties():
    with VisionSubclass() as s:
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


@pytest.mark.main
def test_image():
    img = Image(VisionSubclass(), "Some frame")
    assert(isinstance(img.source, VisionSubclass))
    assert(img.image == "Some frame")


@pytest.mark.main
def test_image_make():
    img = Image._make([VisionSubclass(), "Some frame"])
    assert(isinstance(img.source, VisionSubclass))
    assert(img.image == "Some frame")


@pytest.mark.main
def test_image_replace():
    img = Image(VisionSubclass(), "Some frame")
    img1 = img._replace(image="Some other frame")
    assert(isinstance(img1.source, VisionSubclass))
    assert(img1.image == "Some other frame")


@pytest.mark.main
def test_image_no_source():
    img = Image(None, "Some frame")
    assert(img.source is None)
    assert(img.image == "Some frame")


@pytest.mark.main
def test_image_fail():
    with raises(TypeError):
        Image("fake object", "some frame")


@pytest.mark.main
def test_frame():
    frame = Frame(datetime.now(), 0, (Image(VisionSubclass(), "some frame"), ))
    assert(isinstance(frame.timestamp, datetime))
    assert(frame.index == 0)
    assert(len(frame.images) == 1)
    assert(isinstance(frame.images[0], Image))


@pytest.mark.main
def test_frame_fail_timestamp():
    with raises(TypeError):
        Frame("", 0, (Image(VisionSubclass(), "some frame"), ))


@pytest.mark.main
def test_frame_fail_index():
    with raises(TypeError):
        Frame(datetime.now(), "", (Image(VisionSubclass(), "some frame"), ))


@pytest.mark.main
def test_frame_fail_image():
    with raises(TypeError):
        Frame(datetime.now(), 0, "")


@pytest.mark.main
def test_frame_get_image():
    sourceA = VisionSubclass("A")
    sourceB = VisionSubclass("B")
    sourceC = VisionSubclass("C")

    imgA = Image(sourceA, "some image")
    imgB = Image(sourceB, "some other image")
    frame = Frame(datetime.now(), 0, (imgA, imgB))

    assert(frame.get_image("A") == imgA)
    assert(frame.get_image("B") == imgB)
    assert(frame.get_image("C") is None)

    assert(frame.get_image(sourceA) == imgA)
    assert(frame.get_image(sourceB) == imgB)
    assert(frame.get_image(sourceC) is None)

    with raises(TypeError):
        frame.get_image(1)