#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors.base import *
from EasyVision.processors import MultiProcessing
from EasyVision.vision import *


class MyException(Exception):
    pass


class Subclass(VisionBase):

    def __init__(self, name="", *args, **kwargs):
        super(Subclass, self).__init__(*args, **kwargs)
        self.frame = 0
        self.frames = 10
        self._name = name
        self._camera_called = False
        self._test_remote_get = 'success'

    def capture(self):
        from datetime import datetime
        self.frame += 1
        return Frame(datetime.now(), self.frame - 1, (Image(self, 'an image'),))

    def setup(self):
        super(Subclass, self).setup()

    def release(self):
        super(Subclass, self).release()

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

    @property
    def test_remote_getter_only(self):
        return 'some_constant'

    @property
    def test_remote_get(self):
        print 'remote get', self, self._test_remote_get
        return self._test_remote_get

    @test_remote_get.setter
    def test_remote_get(self, value):
        print 'remote set', self, value
        self._test_remote_get = value

    def test_remote_call(self, a, b, kwarg_test=0):
        print 'remote call', self, self._test_remote_get, a, b, kwarg_test
        return (self._test_remote_get, a, b, kwarg_test)

    def test_remote_exception(self, a, b, kwarg_test=0):
        raise MyException()


class ProcessorA(ProcessorBase):

    def __init__(self, vision, *args, **kwargs):
        super(ProcessorA, self).__init__(vision, *args, **kwargs)
        self._some_field = 0

    @property
    def description(self):
        return "Simple processor"

    def process(self, image):
        new_image = image.image.upper()
        return image._replace(source=self, image=new_image)


def test_capture_mp_freerun():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        for index, img in enumerate(mp):
            assert(isinstance(img, Frame))
            assert(img.images[0].image == "AN IMAGE")
            if index > 10:
                assert(img.index > index)
                break


def test_capture_mp_get():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        assert(mp.remote_get('test_remote_get') == 'success')
        assert(mp.test_remote_get == 'success')
        vision.test_remote_get = 'changing current process which will not change remote process'
        assert(mp.test_remote_get == 'success')


def test_capture_mp_set_getter():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        with raises(AttributeError):
            mp.remote_set('test_remote_getter_only', 0)


def test_capture_mp_set():
    vision = Subclass(0)
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


def test_capture_mp_call():
    vision = Subclass(0)
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


def test_capture_mp_call_exception():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor) as mp:
        with raises(MyException):
            mp.remote_call('test_remote_exception', 2, 5, kwarg_test=7)


def test_capture_mp_noattr():
    vision = Subclass(0)
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


def test_capture_mp_lazy():
    vision = Subclass(0)
    processor = ProcessorA(vision)
    with MultiProcessing(processor, freerun=False) as mp:
        for index, img in enumerate(mp):
            assert(isinstance(img, Frame))
            assert(img.index == index)
            assert(img.images[0].image == "AN IMAGE")
            if img.index > 10:
                break


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


def test_capture_mp_camera():
    vision = VideoCapture(0)
    with MultiProcessing(vision, freerun=False) as mp:
        for i, img in enumerate(mp):
            assert(isinstance(img, Frame))
            if i > 30:
                break