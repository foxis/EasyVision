# -*- coding: utf-8 -*-
from EasyVision.vision.base import *
import cv2


class ProcessorBase(VisionBase):

    def __init__(self, vision, processor_mask=None, enabled=True, *args, **kwargs):
        if not isinstance(vision, VisionBase):
            raise TypeError("Vision object must be of type VisionBase")
        self._vision = vision
        self._processor_mask = Frame.tidy_processor_mask(processor_mask)
        self._enabled = True
        self.enabled = enabled
        super(ProcessorBase, self).__init__(*args, **kwargs)

    @abstractmethod
    def process(self, image):
        pass

    def capture(self):
        super(ProcessorBase, self).capture()
        frame = self._vision.capture()
        if not self.enabled:
            return frame
        elif frame:
            processor_mask = frame.processor_mask if frame.processor_mask else self._processor_mask
            if not processor_mask:
                processor_mask = "1" * len(frame.images)
            images = tuple(m == "0" and img or self.process(img)._replace(source=self) for m, img in zip(processor_mask, frame.images))
            return frame._replace(images=images)

    def setup(self):
        self._vision.setup()
        super(ProcessorBase, self).setup()

    def release(self):
        self._vision.release()
        super(ProcessorBase, self).release()

    @property
    def source(self):
        return self._vision

    def get_source(self, name):
        if self.__class__.__name__ == name:
            return self
        elif isinstance(self._vision, ProcessorBase):
            return self._vision.get_source(name)
        elif self._vision.name == name:
            return self._vision

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super(ProcessorBase, self).__getattr__(name)
        return getattr(self._vision, name)

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        lastenabled, self._enabled = self._enabled, value
        if lastenabled != value and hasattr(self, 'enabled_changed'):
            self.enabled_changed(lastenabled, value)

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

    def display_results_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)