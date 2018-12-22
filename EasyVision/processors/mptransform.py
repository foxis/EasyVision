# -*- coding: utf-8 -*-
import cv2
import numpy as np
import multiprocessing
from .base import *


class MultiProcessing(ProcessorBase, multiprocessing.Process):
    def __init__(self, vision, freerun=True, *args, **kwargs):
        self._freerun = freerun

        self.event = multiprocessing.Event()
        self.event.clear()
        self.running = multiprocessing.Value("i", 1)
        self._in, self._out = multiprocessing.Pipe()

        super(MultiProcessing, self).__init__(vision, *args, **kwargs)

        self.start()

    @property
    def description(self):
        return "Allows processors run on a separate process"

    def process(self, image):
        return self._vision.process(image)

    def capture(self):
        if not self.enabled:
            return self._vision.capture()

        self.event.wait(10)
        self.event.clear()
        frame = self._out.recv()
        return frame

    def release(self):
        self.running.value = 0
        time.sleep(.5)
        self._vision.release()

    def run(self):
        while self.running.value:
            frame = self._vision.capture()

            if not self.event.is_set():
                self._in.send(frame)
                self.event.set()

    def enabled_changed(self, last, current):
        pass