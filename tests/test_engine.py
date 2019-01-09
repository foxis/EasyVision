#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.exceptions import *
from EasyVision.engine.base import EngineBase
from EasyVision.vision.base import VisionBase


class Subclass(EngineBase):

    def __init__(self, vision, *args, **kwargs):
        super(Subclass, self).__init__(vision, *args, **kwargs)

    def setup(self):
        super(Subclass, self).setup()

    def compute(self):
        return self.vision.capture()

    @property
    def description(self):
        pass

    @property
    def capabilities(self):
        pass


class VisionSubclass(VisionBase):

    def __init__(self, *args, **kwargs):
        super(VisionSubclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return (datetime.now, ('an image',))

    def setup(self):
        super(VisionSubclass, self).setup()

    def release(self):
        super(VisionSubclass, self).release()

    @property
    def is_open(self):
        print self.frame, self.frames
        return self.frame < self.frames

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

    @property
    def autoexposure(self):
        pass

    @property
    def autofocus(self):
        pass

    @property
    def autowhitebalance(self):
        pass

    @property
    def autogain(self):
        pass

    @property
    def exposure(self):
        pass

    @property
    def focus(self):
        pass

    @property
    def whitebalance(self):
        pass

    @property
    def gain(self):
        pass


@pytest.mark.main
def test_abstract_vision_abstract():
    with raises(TypeError):
        _ = EngineBase(None)


@pytest.mark.main
def test_abstract_vision_implementation():
    _ = Subclass(VisionSubclass())


@pytest.mark.main
def test_abstract_vision_implementation_bar_arg():
    class BadVision(object):
        pass

    with raises(TypeError):
        _ = Subclass(BadVision())


@pytest.mark.main
def test_iterator():
    with Subclass(VisionSubclass()) as engine:
        count = 0
        for result in engine:
            count += 1
            if count > 13:
                break
        assert(count == 10)