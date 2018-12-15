# -*- coding: utf-8 -*-
from .base import *
from .exceptions import DeviceNotFound
import cv2
from datetime import datetime


class ImagesVision(VisionBase):

    def __init__(self, image_paths, *args, **kwargs):
        self._name = 'images'
        self._images = [cv2.imread(path) for path in image_paths]
        self._frame_count = len(image_paths)
        self._frame_index = 0
        super(ImagesVision, self).__init__(*args, **kwargs)

    def release(self):
        if self._images:
            self._images = None
            if self.debug:
                cv2.destroyWindow(self.name)

    def capture(self):
        if not self.is_open:
            return None

        frame = self._images[self._frame_index]
        self._frame_index += 1
        timestamp = datetime.now()
        if self.display_results:
            cv2.imshow(self.name, frame)
        return Frame(timestamp, self._frame_index, (Image(self, frame), ))

    @property
    def frame_size(self):
        return (0, 0)

    @property
    def fps(self):
        return 0

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def is_open(self):
        return self._frame_index < self._frame_count

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Monocular image file reader"

    @property
    def path(self):
        return ""

    @property
    def devices(self):
        return ()

    def display_results_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)