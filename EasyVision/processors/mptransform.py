# -*- coding: utf-8 -*-
import cv2
import numpy as np
import multiprocessing
from .base import *


class MultiProcessing(ProcessorBase, multiprocessing.Process):
    Attr = namedtuple("Attr", ['name', 'method', 'args', 'kwargs'])

    def __init__(self, vision, freerun=True, *args, **kwargs):
        self._freerun = freerun

        self._running = multiprocessing.Value("i", 0)
        self._frame_in, self._frame_out = multiprocessing.Pipe()
        self._ctrl_in, self._ctrl_out = multiprocessing.Pipe()
        self._res_in, self._res_out = multiprocessing.Pipe()

        self._ctrl_sem = multiprocessing.Semaphore()
        self._res_sem = multiprocessing.Semaphore()

        self._exit_event = multiprocessing.Event()
        self._frame_event = multiprocessing.Event()
        self._cap_event = multiprocessing.Event()
        self._exit_event.clear()
        self._frame_event.clear()
        self._cap_event.clear()

        super(MultiProcessing, self).__init__(vision, *args, **kwargs)

    def setup(self):
        assert(self._running.value == 0)
        self.start()

    def release(self):
        self._running.value = 0
        self._exit_event.wait(10)

    @property
    def description(self):
        return "Allows processors run on a separate process"

    def process(self, image):
        return self._remote_call('process', (image,))

    def capture(self):
        assert(self._running.value)

        if self._freerun:
            self._cap_event.set()

        self._frame_event.wait(10)
        print 'wait done'
        frame = self._out.recv()
        print 'received: ', frame
        self._frame_event.clear()
        if isinstance(frame, Exception):
            raise frame
        return frame

    def _send_ctrl(self, ctrl, lock):
        assert(self._running.value)
        self._ctrl_in.send(ctrl)
        self._ctrl_sem.release()
        self._res_sem.acquire(lock)
        res = self._res_out.recv()
        if isinstance(res, Exception):
            raise res
        return res

    def remote_get(self, name, lock=True):
        return self._send_ctrl(self.Attr(name, 'GET', None, None), lock)

    def remote_set(self, name, value, lock=True):
        self._send_ctrl(self.Attr(name, 'SET', value, None), lock)

    def remote_call(self, name, lock=True, *args, **kwargs):
        return self._send_ctrl(self.Attr(name, 'CALL', args, kwargs), lock)

    def _remote_call_handle(self):
        if self.ctrl_event.acquire(False):
            ctrl = self._ctrl_out.recv()
            try:
                attr = getattr(self._vision, ctrl.name)
                result = None
                if ctrl.method == 'SET':
                    attr = ctrl.args
                elif ctrl.method == 'GET':
                    result = attr
                elif ctrl.method == 'CALL':
                    result = attr(*ctrl.args, **ctrl.kwargs)
                else:
                    pass
                self._res_in.send(result)
            except Exception as e:
                self._res_in.send(e)
            finally:
                self._res_sem.release()

    def run(self):
        super(MultiProcessing, self).setup()
        self._running.value = 1
        self._lazy_frame = None
        while self._running.value:
            self._remote_call_handle()

            try:
                if self._freerun:
                    self._capture_freerun()
                else:
                    self._capture_lazy()
            except Exception as e:
                self._send_frame(e)

            self.exit_event.set()

        super(MultiProcessing, self).release()

    def _send_frame(self, frame):
        if not self._frame_event.is_set():
            print 'sending: ', frame
            self._frame_in.send(frame)
            self._frame_event.set()

    def _capture_freerun(self):
        frame = self._vision.capture()
        self._send_frame(frame)
        if not frame:
            self.running.value = 0

    def _capture_lazy(self):
        if self._cap_event.wait(.001):
            if self._lazy_frame:
                self._send_frame(self._lazy_frame)
                self._cap_event.clear()
            self._lazy_frame = self._vision.capture()
            if not self._lazy_frame:
                self._send_frame(None)
                self.running.value = 0

    def enabled_changed(self, last, current):
        self.remote_set('enabled', current)

    def debug_changed(self, last, current):
        self.remote_set('debug', current)

    def display_results_changed(self, last, current):
        self.remote_set('display_results', current)
