#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *


class Subclass(VisionBase):

    def __init__(self, name="", *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10
        self._name = name

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return (datetime.now, ('an image',))

    def release(self):
        pass

    @property
    def is_open(self):
        return self.frame < self.frames

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        pass

    @property
    def path(self):
        pass

    @property
    def frame_size(self):
        pass

    @property
    def fps(self):
        pass

    @property
    def frame_count(self):
        return self.frames

    @property
    def devices(self):
        """
        :return: [{name:, description:, path:, etc:}]
        """
        pass


def test_abstract_vision_abstract():
    with raises(TypeError):
        VisionBase()


def test_abstract_vision_implementation():
    Subclass()


def test_image():
    img = Image(Subclass(), "Some frame")
    assert(isinstance(img.source, Subclass))
    assert(img.image == "Some frame")


def test_image_mask():
    img = ImageWithMask(Subclass(), "Some frame", ("Some mask", ))
    assert(isinstance(img.source, Subclass))
    assert(img.image == "Some frame")
    assert(img.mask == ("Some mask", ))


def test_image_features():
    img = ImageWithFeatures(Subclass(), "Some frame", "features")
    assert(isinstance(img.source, Subclass))
    assert(img.image == "Some frame")
    assert(img.features == "features")


def test_image_features_mask():
    img = ImageWithFeaturesAndMask(Subclass(), "Some frame", ("Some mask", ), "features")
    assert(isinstance(img.source, Subclass))
    assert(img.image == "Some frame")
    assert(img.features == "features")
    assert(img.mask == ("Some mask", ))


def test_image_no_source():
    img = Image(None, "Some frame")
    assert(img.source is None)
    assert(img.image == "Some frame")


def test_image_fail():
    with raises(TypeError):
        Image("fake object", "some frame")


def test_frame():
    frame = Frame(datetime.now(), 0, (Image(Subclass(), "some frame"), ))
    assert(isinstance(frame.timestamp, datetime))
    assert(frame.index == 0)
    assert(len(frame.images) == 1)
    assert(isinstance(frame.images[0], Image))


def test_frame_fail_timestamp():
    with raises(TypeError):
        Frame("", 0, (Image(Subclass(), "some frame"), ))


def test_frame_fail_index():
    with raises(TypeError):
        Frame(datetime.now(), "", (Image(Subclass(), "some frame"), ))


def test_frame_fail_image():
    with raises(TypeError):
        Frame(datetime.now(), 0, "")


def test_frame_get_image():
    sourceA = Subclass("A")
    sourceB = Subclass("B")
    sourceC = Subclass("C")

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