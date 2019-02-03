# -*- coding: utf-8 -*-
from .base import *
import Pyro4
import functools
from EasyVision.server import Command


class PyroCapture(VisionBase):
    def __init__(self, name):
        self._name = name
        self._proxy = None

    def __getattr__(self, name):
        def caller_proxy(_self, _name, _attr):
            @functools.wraps(_attr)
            def wrapper(*args, **kwargs):
                return _self.remote_call(_name, *args, **kwargs)
            return wrapper

        if name.startswith('__') and name.endswith('__') or self._proxy is None:
            return super(PyroCapture, self).__getattr__(name)

        attr = getattr(self._vision, name)
        if hasattr(attr, '__call__'):
            return caller_proxy(self, name, attr)
        else:
            return self.remote_get(name)

    def remote_get(self, name):
        result = self._proxy.command(Command(name, 'GET', None, None))
        if isinstance(result, Exception):
            raise result
        else:
            return result

    def remote_set(self, name, value):
        result = self._proxy.command(Command(name, 'SET', value, None))
        if isinstance(result, Exception):
            raise result

    def remote_call(self, name, *args, **kwargs):
        result = self._proxy.command(Command(name, 'CALL', args, kwargs))
        if isinstance(result, Exception):
            raise result
        else:
            return result

    def setup(self):
        super(PyroCapture, self).setup()
        self._proxy = Pyro4.Proxy('PYRONAME:%s' % self._name)
        self.remote_call('setup')

    def release(self):
        self.remote_call('release')
        super(PyroCapture, self).release()

    def capture(self):
        super(PyroCapture, self).capture()
        return self.remote_call('capture')

    @property
    def frame_size(self):
        return self.remote_get('frame_size')

    @property
    def fps(self):
        return self.remote_get('fps')

    @property
    def frame_count(self):
        return self.remote_get('frame_count')

    @property
    def is_open(self):
        return self.remote_get('is_open')

    @property
    def name(self):
        return self._name if self._name else "Capture {}".format(self._path)

    @property
    def description(self):
        return "Remote capturer using Pyro"

    @property
    def path(self):
        return self.remote_get('path')

    @property
    def devices(self):
        return self.remote_get('devices')

    @property
    def autoexposure(self):
        return self.remote_get('autoexposure')

    @property
    def autofocus(self):
        return self.remote_get('autofocus')

    @property
    def autowhitebalance(self):
        return self.remote_get('autowhitebalance')

    @property
    def autogain(self):
        return self.remote_get('autogain')

    @property
    def exposure(self):
        return self.remote_get('exposure')

    @property
    def focus(self):
        return self.remote_get('focus')

    @property
    def whitebalance(self):
        return self.remote_get('whitebalance')

    @property
    def gain(self):
        return self.remote_get('gain')

    @autoexposure.setter
    def autoexposure(self, value):
        self.remote_set('autoexposure', value)

    @autofocus.setter
    def autofocus(self, value):
        self.remote_set('autofocus', value)

    @autowhitebalance.setter
    def autowhitebalance(self, value):
        self.remote_set('autowhitebalance', value)

    @autogain.setter
    def autogain(self, value):
        self.remote_set('autogain', value)

    @exposure.setter
    def exposure(self, value):
        self.remote_set('exposure', value)

    @focus.setter
    def focus(self, value):
        self.remote_set('focus', value)

    @whitebalance.setter
    def whitebalance(self, value):
        self.remote_set('whitebalance', value)

    @gain.setter
    def gain(self, value):
        self.remote_set('gain', value)

    # Calibrated camera specifics
    @property
    def camera(self):
        return self.remote_get('camera')

    @camera.setter
    def camera(self, value):
        self.remote_set('camera', value)
