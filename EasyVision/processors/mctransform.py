# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import cv2
from .base import *
from EasyVision.base import EasyVisionBase


class MultiConsumers(ProcessorBase):

    def __init__(self, *args, **kwargs):
        self._frame = None
        self._consumers = 0
        self._numcaptured = 0
        super(MultiConsumers, self).__init__(*args, **kwargs)

    def process(self, image):
        raise NotImplementedError()

    def capture(self):
        super(ProcessorBase, self).capture()

        if self._numcaptured == 0:
            self._frame = self._vision.capture()

        self._numcaptured = (self._numcaptured + 1) % self._consumers

        return self._frame

    def setup(self):
        self._numconsumed = 0
        if not self._consumers:
            super(MultiConsumers, self).setup()
        self._consumers += 1

    def release(self):
        self._consumers -= 1
        assert(self._consumers >= 0)
        if not self._consumers:
            super(MultiConsumers, self).release()

    @property
    def description(self):
        return "Allows to consume the same frame multiple times."