# -*- coding: utf-8 -*-
import cv2
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from .base import *
from EasyVision.exceptions import TimeoutError
import functools
import cPickle
import ctypes
#import os
#import affinity


Attr = namedtuple("Attr", 'name method args kwargs')
multiprocessing.connection.BUFSIZE = 64 * 1024 * 1024


class MultiProcessing(ProcessorBase, mp.Process):

    def __init__(self, vision, freerun=True, timeout=10, *args, **kwargs):
        self._freerun = freerun

        self._timeout = timeout
        self._running = mp.Value("b", 0)

        self._frame_in, self._frame_out = mp.Pipe(False)
        self._ctrl_in, self._ctrl_out = mp.Pipe(False)
        self._res_in, self._res_out = mp.Pipe(False)

        self._ctrl_sem = mp.Semaphore(0)
        self._res_sem = mp.Semaphore(0)

        self._run_event = mp.Event()
        self._frame_event = mp.Event()
        self._cap_event = mp.Event()
        self._run_event.clear()
        self._frame_event.clear()
        self._cap_event.clear()

        super(MultiProcessing, self).__init__(vision, *args, **kwargs)

    def __getattr__(self, name):
        def caller_proxy(_self, _name, _attr):
            @functools.wraps(_attr)
            def wrapper(*args, **kwargs):
                return _self.remote_call(_name, *args, **kwargs)
            return wrapper

        if (name.startswith('__') and name.endswith('__')) or not self._running.value:
            return super(MultiProcessing, self).__getattr__(name)
        attr = getattr(self._vision, name)
        if hasattr(attr, '__call__'):
            return caller_proxy(self, name, attr)
        else:
            return self.remote_get(name)

    def next(self):
        frame = self.capture()

        if frame is None:
            raise StopIteration()
        return frame

    def setup(self):
        assert(not self._running.value)
        self.start()
        if not self._run_event.wait(self._timeout):
            raise TimeoutError()

    def release(self):
        self._running.value = False
        self.join(self._timeout)

    @property
    def is_open(self):
        return self._running.value

    @property
    def description(self):
        return "Allows processors run on a separate process"

    def process(self, image):
        return self.remote_call('process', image)

    def capture(self):
        if not self._running.value:
            return None

        if not self._freerun:
            self._cap_event.set()
        if not self._frame_event.wait(self._timeout):
            raise TimeoutError()

        frame = Frame.frombytes(self._frame_in.recv_bytes())

        self._frame_event.clear()
        if isinstance(frame, Exception):
            raise frame
        if isinstance(frame, Frame):
            frame = frame._replace(images=tuple(i._replace(source=self) for i in frame.images))
        return frame

    def _send_ctrl(self, ctrl, lock=True):
        assert(self._running.value)
        self._ctrl_out.send_bytes(cPickle.dumps(ctrl, protocol=-1))
        self._ctrl_sem.release()
        self._res_sem.acquire(lock)
        res = cPickle.loads(self._res_in.recv_bytes())
        if isinstance(res, Exception):
            raise res
        return res

    def remote_get(self, name):
        return self._send_ctrl(Attr(name, 'GET', None, None))

    def remote_set(self, name, value):
        self._send_ctrl(Attr(name, 'SET', value, None))

    def remote_call(self, name, *args, **kwargs):
        return self._send_ctrl(Attr(name, 'CALL', args, kwargs))

    def _remote_call_handle(self):
        if self._ctrl_sem.acquire(False):
            ctrl = cPickle.loads(self._ctrl_in.recv_bytes())
            try:
                result = None
                if ctrl.method == 'SET':
                    cur_obj = self._vision
                    last_obj = None
                    while cur_obj is not None or last_obj is not None:
                        last_obj, cur_obj = cur_obj, getattr(cur_obj, '_vision', None)

                        if hasattr(last_obj, ctrl.name) and not hasattr(cur_obj, ctrl.name):
                            setattr(last_obj, ctrl.name, ctrl.args)
                            break
                    if cur_obj is None and last_obj is None:
                        raise AttributeError("can't set attribute")
                elif ctrl.method == 'GET':
                    result = getattr(self._vision, ctrl.name)
                elif ctrl.method == 'CALL':
                    result = getattr(self._vision, ctrl.name)(*ctrl.args, **ctrl.kwargs)
                else:
                    pass
                self._res_out.send(result)
            except Exception as e:
                self._res_out.send(e)
            finally:
                self._res_sem.release()

    def run(self):
        super(MultiProcessing, self).setup()
        self._running.value = True
        self._lazy_frame = None
        self._run_event.set()
        while self._running.value:
            self._remote_call_handle()

            try:
                if self._freerun:
                    self._capture_freerun()
                else:
                    self._capture_lazy()
            except Exception as e:
                self._send_frame(e)

            if self.debug:
                cv2.waitKey(1)
        self._running.value = 0
        if self.debug:
            cv2.waitKey(0)

        super(MultiProcessing, self).release()

    def _send_frame(self, frame):
        if not self._frame_event.is_set():
            data = frame.tobytes() if isinstance(frame, Frame) else cPickle.dumps(frame, protocol=-1)
            self._frame_out.send_bytes(data)
            self._frame_event.set()

    def _capture_freerun(self):
        frame = self._vision.capture()
        self._send_frame(frame)
        if not frame:
            self._running.value = False

    def _capture_lazy(self):
        if self._cap_event.wait(.001):
            if self._lazy_frame:
                self._send_frame(self._lazy_frame)
                self._cap_event.clear()
            self._lazy_frame = self._vision.capture()
            #print os.getpid(), affinity.get_process_affinity_mask(os.getpid())
            if not self._lazy_frame:
                self._send_frame(None)
                self._running.value = False

    @property
    def autoexposure(self):
        return self.remote_get("autoexposure")

    @property
    def autofocus(self):
        return self.remote_get("autofocus")

    @property
    def autowhitebalance(self):
        return self.remote_get("autowhitebalance")

    @property
    def autogain(self):
        return self.remote_get("autogain")

    @property
    def exposure(self):
        return self.remote_get("exposure")

    @property
    def focus(self):
        return self.remote_get("focus")

    @property
    def whitebalance(self):
        return self.remote_get("whitebalance")

    @property
    def gain(self):
        return self.remote_get("gain")

    @autoexposure.setter
    def autoexposure(self, value):
        self.remote_set("autoexposure", value)

    @autofocus.setter
    def autofocus(self, value):
        self.remote_set("autofocus", value)

    @autowhitebalance.setter
    def autowhitebalance(self, value):
        self.remote_set("autowhitebalance", value)

    @autogain.setter
    def autogain(self, value):
        self.remote_set("autogain", value)

    @exposure.setter
    def exposure(self, value):
        self.remote_set("exposure", value)

    @focus.setter
    def focus(self, value):
        self.remote_set("focus", value)

    @whitebalance.setter
    def whitebalance(self, value):
        self.remote_set("whitebalance", value)

    @gain.setter
    def gain(self, value):
        self.remote_set("gain", value)