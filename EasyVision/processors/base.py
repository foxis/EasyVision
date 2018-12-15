# -*- coding: utf-8 -*-
from EasyVision.vision.base import *


class ProcessorBase(VisionBase):

    def __init__(self, vision, enabled=True, *args, **kwargs):
        if not isinstance(vision, VisionBase):
            raise TypeError("Vision object must be of type VisionBase")
        self._vision = vision
        self._enabled = enabled
        super(ProcessorBase, self).__init__(*args, **kwargs)

    @abstractmethod
    def process(self, image):
        pass

    def capture(self):
        frame = self._vision.capture()
        if not self.enabled:
            return frame
        else:
            images = tuple(self.process(img) for img in frame.images)
            return frame._replace(images=images)

    def release(self):
        self._vision.release()

    @property
    def source(self):
        return self._vision

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    @property
    def is_open(self):
        return self._vision.is_open

    @property
    def frame_size(self):
        return self._vision.frame_size

    @property
    def fps(self):
        return self._vision.fps

    @property
    def name(self):
        return "{} <- {}".format(super(ProcessorBase, self).name, self._vision.name)

    @property
    def frame_count(self):
        return self._vision.frame_count

    @property
    def path(self):
        return self._vision.path

    @property
    def devices(self):
        return self._vision.devices