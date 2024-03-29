# -*- coding: utf-8 -*-
"""Uses multithreading to efficiently read frames from VideoCapture.
"""

import threading as mt
from .base import *
import cv2


class MultiThreading(ProcessorBase):
    """
    """

    def __init__(self, vision, timeout=10, *args, **kwargs):
        """MultiThreading instance initialization

        :param vision: capturing source object
        :param timeout: timeout for calls
        """
        self._timeout = timeout

        self._run_event = mt.Event()
        self._exit_event = mt.Event()
        self._frame_event = mt.Event()
        self._run_event.clear()
        self._exit_event.clear()
        self._frame_event.clear()
        self._frame = None
        self._thread = None
        self._lock = mt.Lock()

        super(MultiThreading, self).__init__(vision, *args, **kwargs)

    def next(self):
        frame = self.capture()

        if frame is None:
            raise StopIteration()
        return frame

    def setup(self):
        assert(not self._run_event.is_set())
        super(MultiThreading, self).setup()

        self._run_event.clear()
        self._exit_event.clear()
        self._frame_event.clear()
        self._frame = None

        self._thread = mt.Thread(target=self.run)
        self._thread.start()
        if not self._run_event.wait(self._timeout):
            raise TimeoutError()

    def release(self):
        self._run_event.clear()
        self._exit_event.wait(self._timeout)
        super(MultiThreading, self).release()

    @property
    def description(self):
        return "Allows multithreaded capturing"

    def process(self, image):
        return self.source.process(image)

    def capture(self):
        if not self._run_event.is_set():
            return None
        self.update_fps()
        if not self._frame_event.wait(self._timeout):
            raise TimeoutError()
        with self._lock:
            frame = self._frame
            self._frame_event.clear()

        return frame

    def run(self):
        self._run_event.set()
        for frame in self.source:
            if not self._run_event.is_set():
                break

            with self._lock:
                self._frame = frame
                self._frame_event.set()

            if self.display_results:
                cv2.waitKey(1)

        with self._lock:
            self._frame = None
            self._frame_event.set()

        self._run_event.clear()
        self._exit_event.set()
