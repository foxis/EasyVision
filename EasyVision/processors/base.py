# -*- coding: utf-8 -*-
from EasyVision.vision.base import *


class ProcessorBase(VisionBase):

    def __init__(self, vision, enabled=True, *args, **kwargs):
        if not isinstance(vision, VisionBase):
            raise TypeError("Vision object must be of type VisionBase")
        super(ProcessorBase, self).__init__(*args, **kwargs)
        self._vision = vision
        self._enabled = enabled

    @abstractmethod
    def process(self, frame):
        pass

    def capture(self):
        return self.process(self._vision.capture()) if self._enabled else self._vision.capture()

    @property
    def enabled(self):     return self._enabled

    @property.setter
    def enabled(self, value):     self._enabled = value

    @property
    def is_open(self):     return self._vision.is_open

    @property
    def frame_size(self):     return self._vision.frame_size

    @property
    def fps(self):     return self._vision.fps

    @property
    def frame_count(self):     return self._vision.frame_count

    @property
    def path(self):    return self._vision.path

    @property
    def devices(self):    return self._vision.devices