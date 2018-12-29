# -*- coding: utf-8 -*-
from .base import *
from .exceptions import DeviceNotFound
import cv2
from datetime import datetime


class VideoCapture(VisionBase):

    def __init__(self, path, width=None, height=None, fps=None, name=None, *args, **kwargs):
        self._name = name
        self._path = path
        self._frame_index = 0
        self._width, self._height = width, height
        self._fps = fps
        self._frame_count = -1
        self._frame_size = None
        self._capture = None
        self._is_open = False
        super(VideoCapture, self).__init__(*args, **kwargs)

    def setup(self):
        super(VideoCapture, self).setup()
        self._capture = cv2.VideoCapture(self._path)
        if self._width and self._height:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        if self._fps:
            self._capture.set(cv2.CAP_PROP_FPS, self._fps)

        self._is_open = self._capture.isOpened()
        if not self._is_open:
            raise DeviceNotFound()

        self._frame_size = (self._capture.get(cv2.CAP_PROP_FRAME_WIDTH), self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._fps = self._capture.get(cv2.CAP_PROP_FPS)
        self._frame_count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)

        self._frame_index = 0

    def release(self):
        if self._capture:
            self._capture.release()
            self._is_open = False
            self._capture = None
        super(VideoCapture, self).release()

    def capture(self):
        super(VideoCapture, self).capture()
        if not self.is_open:
            return None

        self._is_open, image = self._capture.read()
        if not self._is_open:
            return None
        timestamp = datetime.now()
        if self.display_results:
            cv2.imshow(self.name, image)
        self._frame_index += 1
        return Frame(timestamp, self._frame_index - 1, (Image(self, image), ))

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
        return self._is_open

    @property
    def name(self):
        return self._name if self._name else "Capture {}".format(self._path)

    @property
    def description(self):
        return "Camera/Video file capturer"

    @property
    def path(self):
        return self._path

    @property
    def devices(self):
        return ()

    def display_results_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)