#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors.base import *


class Subclass(VisionBase):

    def __init__(self, name="", *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10
        self._name = name

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return Frame(datetime.now(), self.frame - 1, (Image(self, 'an image'),))

    def release(self):
        pass

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


class ProcessorB(ProcessorBase):

    def __init__(self, vision, *args, **kwargs):
        super(ProcessorB, self).__init__(vision, *args, **kwargs)

    @property
    def description(self):
        return "Simple processor 2"

    def process(self, image):
        new_image = image.image.title()
        return image._replace(source=self, image=new_image)


def test_abstract():
    with raises(TypeError):
        ProcessorBase()


def test_implementation():
    source = Subclass()
    pr = ProcessorA(source)
    assert(pr.source is source)


def test_capture(mocker):
    vision = Subclass(0)
    processor = ProcessorA(vision)

    img = processor.capture()
    assert(isinstance(img, Frame))
    assert(img.images[0].source is processor)
    assert(img.images[0].image == "AN IMAGE")


def test_capture_stacked(mocker):
    vision = Subclass(0)
    processorA = ProcessorA(vision)
    processorB = ProcessorB(processorA)

    assert(processorB.name == "ProcessorB <- ProcessorA <- Test")

    img = processorB.capture()
    assert(isinstance(img, Frame))
    assert(img.images[0].source is processorB)
    assert(img.images[0].image == "An Image")
    assert(processorB.get_source('Test') is vision)
    assert(processorB.get_source('ProcessorA') is processorA)
    assert(processorB.get_source('ProcessorB') is processorB)
    assert(processorB.get_source('Test no') is None)
