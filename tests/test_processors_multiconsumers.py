#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.vision.base import *
from EasyVision.processors import MultiConsumers
from EasyVision.vision import *
from itertools import izip


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


@pytest.mark.main
def test_capture_multiconsumers():
    vision = Subclass(0)
    processor = MultiConsumers(vision)
    with processor as p1:
        with processor as p2:
            for img1, img2 in izip(p1, p2):
                assert(isinstance(img1, Frame))
                assert(isinstance(img2, Frame))
                assert(img1 == img2)
                if img1.index > 10:
                    break


@pytest.mark.main
def test_capture_multiconsumers_fail():
    vision = Subclass(0)
    processor = MultiConsumers(vision)

    processor.setup()
    processor.setup()

    processor.release()
    processor.release()

    with raises(AssertionError):
        processor.release()


