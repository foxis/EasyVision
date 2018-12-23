# -*- coding: utf-8 -*-
from .base import *
from .exceptions import DeviceNotFound
import cv2
from datetime import datetime


class MonocularVision(VisionBase):

    def __init__(self, path, width=None, height=None, fps=None, name=None, *args, **kwargs):
        self._name = name
        self._path = path
        self._frame_index = 0
        self._width, self._height = width, height
        self._fps = fps
        self._capture = None
        super(MonocularVision, self).__init__(*args, **kwargs)

    def setup(self):
        super(MonocularVision, self).setup()
        self._capture = cv2.VideoCapture(self._path)
        if self._width and self._height:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps:
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)

        if not self._capture.isOpened():
            raise DeviceNotFound()

        self._frame_size = (self._capture.get(cv2.CAP_PROP_FRAME_WIDTH), self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frame_count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)

    def release(self):
        if self._capture:
            self._capture.release()
            self._is_open = False
            self._capture = None
            if self.debug:
                cv2.destroyWindow(self.name)

    def capture(self):
        if not self.is_open:
            return None

        self._is_open, frame = self._capture.read()
        timestamp = datetime.now()
        if self.debug:
            cv2.imshow(self.name, frame)
        return Frame(timestamp, self._frame_index, (Image(self, frame), ))

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def fps(self):
        return self._fps

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def is_open(self):
        assert(self._capture is not None)
        return self._capture.isOpened()

    @property
    def name(self):
        return self._name if self._name else "Capture {}".format(self._path)

    @property
    def description(self):
        return "Monocular Camera/Video file capturer"

    @property
    def path(self):
        return self._path

    @property
    def devices(self):
        return ()

    def debug_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)