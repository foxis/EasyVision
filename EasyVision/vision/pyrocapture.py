# -*- coding: utf-8 -*-
""" Implements capturing from Pyro4 server.

"""

from EasyVision.base import lru_cache
from .base import *
import Pyro4
import socket
from EasyVision.server import Command

try:
    import cPickle as pickle
except:
    import pickle


class PyroCapture(VisionBase):
    """Class for capturing frames from remote processor stack server.

    Uses Pyro4 remote object names. Should be used in conjunction with ``EasyVision.server``.
    Best results are achieved using EasyVision.base.server and Pyro4-ns.

    """

    def __init__(self, name, *args, **kwargs):
        self._name = name
        self._proxy = Pyro4.Proxy('PYRONAME:%s' % self._name)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect(tuple(self._proxy.getsockname()))
        super(PyroCapture, self).__init__(*args, **kwargs)

    def __delete__(self, instance):
        self._sock.close()

    def __getattr__(self, name):
        def caller_proxy(_self, _name):
            def wrapper(*args, **kwargs):
                """Remote object call wrapper"""
                return _self.remote_call(_name, *args, **kwargs)
            return wrapper

        if name.startswith('__') and name.endswith('__') or self._proxy is None:
            raise AttributeError("No such attribute could be found: %s" % name)

        if self._hascall(name):
            return caller_proxy(self, name)
        return self.remote_get(name)

    @lru_cache(maxsize=None)
    def _hascall(self, name):
        return self._proxy.hascall(name)

    @lru_cache(maxsize=None)
    def _hasattr(self, name):
        return self._proxy.hasattr(name)

    def __receive_blob(self, blob_id):
        if blob_id is not None:
            self._sock.sendall(blob_id.encode('utf-8'))
            len = int(self._sock.recv(16).decode())
            data = self._sock.recv(len)
            result = pickle.loads(data)
            return result

    def __command(self, cmd):
        assert(self._proxy is not None)
        data = pickle.dumps(cmd, protocol=-1)
        blob_id = self._proxy.command(data)
        return self.__receive_blob(blob_id)

    def remote_get(self, name):
        return self.__command(Command(name, 'GET', None, None))

    def remote_set(self, name, value):
        self.__command(Command(name, 'SET', value, None))

    def remote_call(self, name, *args, **kwargs):
        return self.__command(Command(name, 'CALL', args, kwargs))

    def setup(self):
        super(PyroCapture, self).setup()
        self._proxy.setup()

    def release(self):
        self._proxy.release()
        super(PyroCapture, self).release()

    def capture(self):
        super(PyroCapture, self).capture()
        blob_id = self._proxy.capture()
        return self.__receive_blob(blob_id)

    def compute(self):
        super(PyroCapture, self).capture()
        blob_id = self._proxy.compute()
        return self.__receive_blob(blob_id)

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
        return "{} @ {}".format(self._name, self.remote_get('name'))

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
        """Retrieves StereoCamera or PinholeCamera for stacks that contain CalibratedStereoCamera or CalibratedCamera"""
        return self.remote_get('camera')

    @camera.setter
    def camera(self, value):
        """Sets StereoCamera or PinholeCamera for stacks that contain CalibratedStereoCamera or CalibratedCamera"""
        self.remote_set('camera', value)
