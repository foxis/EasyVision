#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors.base import *
from EasyVision.processors import MultiProcessing
from EasyVision.vision import *
from collections import namedtuple
from .common import VisionSubclass, MyException

Payload = namedtuple('Payload', ('a', 'b'))


class ProcessorA(ProcessorBase):

    def __init__(self, vision, *args, **kwargs):
        super(ProcessorA, self).__init__(vision, *args, **kwargs)
        self._some_field = 0

    def test_payload1(self, payload):
        assert(payload.a == payload.b * 2)
        return Payload(payload.a + payload.b, 10)

    def test_payload2(self, payload):
        assert(payload.a == payload.b * 2)
        return Image(None, payload.a + payload.b)

    @property
    def description(self):
        return "Simple processor"

    def process(self, image):
        new_image = image.image.upper()
        return image._replace(source=self, image=new_image)


@pytest.mark.main
def test_capture_mp_freerun():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        for index, img in enumerate(mp):
            assert(isinstance(img, Frame))
            assert(img.images[0].image == "AN IMAGE")
            if index > 10:
                assert(img.index > index)
                break


@pytest.mark.main
def test_capture_mp_get():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        assert(mp.remote_get('test_remote_get') == 'success')
        assert(mp.test_remote_get == 'success')
        vision.test_remote_get = 'changing current process which will not change remote process'
        assert(mp.test_remote_get == 'success')


@pytest.mark.main
def test_capture_mp_set_getter():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        with raises(AttributeError):
            mp.remote_set('test_remote_getter_only', 0)


@pytest.mark.main
def test_capture_mp_set():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        mp.remote_set('test_remote_get', 1)
        assert(mp.remote_get('test_remote_get') == 1)
        assert(mp.test_remote_get == 1)
        assert(vision.test_remote_get == 'success')

        mp.remote_set('_some_field', 1)
        assert(mp._some_field == 1)
        assert(processor._some_field == 0)
        processor._some_field = 3
        assert(mp._some_field == 1)
        assert(processor._some_field == 3)


@pytest.mark.main
def test_capture_mp_call():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        assert(mp.remote_call('test_remote_call', 2, 5, kwarg_test=7) == ('success', 2, 5, 7))
        vision.test_remote_get = 1
        vision.test_remote_get = 3
        assert(mp.remote_call('test_remote_call', 2, 5, kwarg_test=7) == ('success', 2, 5, 7))
        assert(mp.test_remote_call(2, 5, kwarg_test=7) == ('success', 2, 5, 7))
        mp.remote_set('test_remote_get', 2)
        assert(mp.remote_get('test_remote_get') == 2)
        assert(mp.remote_call('test_remote_call', 2, 5, kwarg_test=7) == (2, 2, 5, 7))
        assert(mp.test_remote_call(2, 5, kwarg_test=7) == (2, 2, 5, 7))
        assert(mp.test_payload1(Payload(2, 1)).a == 3)
        assert(mp.test_payload2(Payload(2, 1)).image == 3)
        assert(mp.process(Image(None, 'testing')).image == 'TESTING')


@pytest.mark.main
def test_capture_mp_call_exception():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        with raises(MyException):
            mp.remote_call('test_remote_exception', 2, 5, kwarg_test=7)


@pytest.mark.main
def test_capture_mp_noattr():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        with raises(AttributeError):
            mp.remote_call('test_remote_exception_no_such_attr', 2, 5, kwarg_test=7)
        with raises(AttributeError):
            _ = mp.test_remote_exception_no_such_attr
        with raises(AttributeError):
            _ = mp.remote_get('test_remote_exception_no_such_attr')
        with raises(AttributeError):
            _ = mp.remote_set('test_remote_exception_no_such_attr', 0)


@pytest.mark.main
def test_capture_mp_lazy():
    vision = VisionSubclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor, freerun=False) as mp:
        for index, img in enumerate(mp):
            assert(isinstance(img, Frame))
            assert(img.index == index)
            assert(img.images[0].image == "AN IMAGE")
            if img.index > 10:
                break


@pytest.mark.slow
def test_capture_mp_images():
    vision = ImagesReader(["test_data/34838518832_fd00147042_k.jpg", "test_data/2732011028_f0f033e678_b.jpg", "test_data/4472701625_6b23da9a23_b.jpg"])
    with MultiProcessing(vision, freerun=False) as mp:
        frame_count = 0
        for frame in mp:
            frame_count += 1
            assert(isinstance(frame, Frame))
            assert(frame.images[0].image is not None)
            print(frame.images[0].image.shape)

        assert(frame_count == 3)


@pytest.mark.slow
def test_capture_mp_camera():
    vision = VideoCapture(0)
    with MultiProcessing(vision, freerun=False) as mp:
        for i, img in enumerate(mp):
            assert(isinstance(img, Frame))
            if i > 30:
                break

@pytest.mark.main
def test_mp_properties():
    vision = VisionSubclass("Test")

    with MultiProcessing(vision, freerun=False) as s:
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

        assert(vision.autoexposure is None)
        assert(vision.autofocus is None)
        assert(vision.autowhitebalance is None)
        assert(vision.autogain is None)
        assert(vision.exposure is None)
        assert(vision.focus is None)
        assert(vision.whitebalance is None)
