#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *


class Subclass(VisionBase):

    def __init__(self, *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10

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
        pass

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
    Image(Subclass(), "Some frame")


def test_image_fail():
    with raises(TypeError):
        Image("fake object", "some frame")


def test_frame():
    Frame(datetime.now(), 0, (Image(Subclass(), "some frame"), ))


def test_frame_fail_timestamp():
    with raises(TypeError):
        Frame("", 0, (Image(Subclass(), "some frame"), ))


def test_frame_fail_index():
    with raises(TypeError):
        Frame(datetime.now(), "", (Image(Subclass(), "some frame"), ))


def test_frame_fail_image():
    with raises(TypeError):
        Frame(datetime.now(), 0, "")
