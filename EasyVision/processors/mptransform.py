# -*- coding: utf-8 -*-
"""Uses multiprocessing to allow for distributed processing.

NOTE: will set ``multiprocessing.connection.BUFSIZE`` to 64Mb in order to increase frame transfer between processes.
This is needed as using Pipe is much faster than using e.g. RawArray or anything else. Although Pipe is still very slow.
By the way, Multiprocessing doesn't really work fine with e.g. feature extraction or any other openCV algorithms
and usually is a little bit slower than if processing sequentially. See tests for more details.
"""

import cv2
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from .base import *
from EasyVision.exceptions import TimeoutError
import functools

try:
    import cPickle as pickle
except ImportError:
    import pickle

Attr = namedtuple("Attr", 'name method args kwargs')
multiprocessing.connection.BUFSIZE = 64 * 1024 * 1024


class MultiProcessing(ProcessorBase, mp.Process):
    """Implements processor stack using multiprocessing
    Allows to set/get properties on the forked process as well as calling methods.
    This processor supports two modes: freerun and lazy.

    In freerun mode after a new process is forked a capturing loop will be started.
    The parent process in this case will capture the last captured frame and wait for any new frames.

    In lazy mode last captured frame will be returned and new capturing will be initiated. If no last captured
    frame is available, then capturing will be initiated and it's result will be returned.

    Usually for streaming devices freerun should be used. Lazy mode is used primarily for tests that use ImagesReader
    so that every frame will be processed.
    """

    def __init__(self, vision, freerun=True, timeout=10, *args, **kwargs):
        """MultiProcessing instance initialization

        :param vision: capturing source object
        :param freerun: indicates whether to execute capturing loop asynchronously
        :param timeout: timeout for calls
        """
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

        # this line is required for pickling/unpickling after fork
        if not hasattr(self, '_vision') or not hasattr(self, '_running'):
            raise AttributeError("")

        if not self._running.value:
            return getattr(self._vision, name)

        attr = getattr(self._vision, name)
        if hasattr(attr, '__call__'):
            return caller_proxy(self, name, attr)
        else:
            return self.remote_get(name)

    #def __setattr__(self, name, value):
    #    if name.startswith('_') or not hasattr(self._vision, name):
    #        super(MultiProcessing, self).__setattr__(name, value)
    #    elif self._running.value:
    #        self.remote_set(name, value)
    #    else:
    #        setattr(self._vision, name, value)

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
            if not self._running.value:
                return None
            raise TimeoutError()

        frame = Frame.frombytes(self._frame_in.recv_bytes())

        self._frame_event.clear()
        if isinstance(frame, Exception):
            raise frame
        if isinstance(frame, Frame):
            frame = frame._replace(images=tuple(i._replace(source=self) for i in frame.images))
        return frame

    def _send_ctrl(self, ctrl, lock=True):
        """Helper method to send control messages to a forked process

        :param ctrl: instance of Attr. Will be pickled before sending through a pipe.
        :param lock:
        :return: whatever the _remote_call_handle produced
        """
        assert(self._running.value)
        self._ctrl_out.send_bytes(pickle.dumps(ctrl, protocol=-1))
        self._ctrl_sem.release()
        self._res_sem.acquire(lock)
        res = pickle.loads(self._res_in.recv_bytes())
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
        """Remote call handle. Will receive pickled Attr instances through the pipe, decode them,
        call appropriate methods from source and send out pickled results."""
        if self._ctrl_sem.acquire(False):
            ctrl = pickle.loads(self._ctrl_in.recv_bytes())
            try:
                result = None
                if ctrl.method == 'SET':
                    """Setting of properties is rather interesting task, as we want to set the property
                    whenever we find it. and this code tries to find the right property.
                    Due to __getattr__ implementation we will find the property in processors even if they don't 
                    have it, but their source does.
                    """
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
        """Forked process loop"""
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
        """Helper method to send a frame. Will pickle None, Frame and exceptions."""
        if not self._frame_event.is_set():
            data = frame.tobytes() if isinstance(frame, Frame) else pickle.dumps(frame, protocol=-1)
            self._frame_out.send_bytes(data)
            self._frame_event.set()

    def _capture_freerun(self):
        """Helper method to capture frames in freerun mode"""
        frame = self._vision.capture()
        self._send_frame(frame)
        if not frame:
            self._running.value = False

    def _capture_lazy(self):
        """Helper method to capture frames in lazy mode."""
        if self._cap_event.wait(.001):
            if self._lazy_frame:
                self._send_frame(self._lazy_frame)
                self._cap_event.clear()
            self._lazy_frame = self._vision.capture()
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