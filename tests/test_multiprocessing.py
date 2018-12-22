#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors.base import *
from EasyVision.processors import MultiProcessing


class Subclass(VisionBase):

    def __init__(self, name="", *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10
        self._name = name
        self._camera_called = False

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return Frame(datetime.now(), self.frame - 1, (Image(self, 'an image'),))

    def release(self):
        pass

    def camera(self):
        self._camera_called = True
        return True

    @property
    def camera_called(self):
        return self._camera_called

    @property
    def is_open(self):
        return self.frame < self.frames

    @property
    def name(self):
        return 'Test'

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


class ProcessorA(ProcessorBase):

    def __init__(self, vision, *args, **kwargs):
        super(ProcessorA, self).__init__(vision, *args, **kwargs)

    @property
    def description(self):
        return "Simple processor"

    def process(self, image):
        new_image = image.image.upper()
        return image._replace(source=self, image=new_image)


def test_capture_mp():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        img = mp.capture()
        assert(isinstance(img, Frame))
        assert(img.images[0].image == "AN IMAGE")


def test_capture_mp_lazy():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor, freerun=False) as mp:
        img = mp.capture()
        assert(isinstance(img, Frame))
        assert(img.images[0].image == "AN IMAGE")
