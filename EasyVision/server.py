# -*- coding: utf-8 -*-
"""Processor stack server using Pyro4. Used in conjunction with vision.PyroCapture.

Uses Pyro4 for RPC and raw socket for return data transfer.

NOTE: Passing images to the server is very inefficient.
"""

from EasyVision.base import *
from collections import namedtuple
import threading
import Pyro4
import select
import uuid
import base64
from datetime import datetime

try:
    # Due to Multiprocessing incompatability
    from EasyVision.engine.base import EngineBase
except ImportError:
    pass

from EasyVision.vision.base import VisionBase

try:
    import cPickle as pickle
except ImportError:
    import pickle


Command = namedtuple('Command', 'name method args kwargs')


@Pyro4.behavior(instance_mode="single")
class ProxyVision(object):
    """Proxy object for accepting and handling remote commands.
    """

    def __init__(self, vision, freerun):
        if not isinstance(vision, VisionBase) and not isinstance(vision, EngineBase):
            raise TypeError("Vision must be VisionBase or EngineBase")

        self._vision = vision
        self._freerun = freerun
        self._result = None
        self._result_ready = False
        self._running = False
        self._result_lock = threading.Lock()
        self._event = threading.Event()
        self._exit_event = threading.Event()
        self._event.clear()
        self._exit_event.clear()

        super(ProxyVision, self).__init__()

    @Pyro4.expose
    def echo(self, data):
        """Used for testing. Will return whatever is passed to it."""
        return data

    @Pyro4.expose
    def command(self, data):
        """Will handle remote commands"""
        data = base64.b64decode(data['data'])
        ctrl = pickle.loads(data)
        if ctrl.method == 'SET':
            cur_obj = self._vision
            last_obj = None
            while cur_obj is not None or last_obj is not None:
                last_obj, cur_obj = cur_obj, getattr(cur_obj, '_vision', None)
                if hasattr(last_obj, ctrl.name) and not hasattr(cur_obj, ctrl.name):
                    setattr(last_obj, ctrl.name, ctrl.args)
                    return None
            if cur_obj is None and last_obj is None:
                raise AttributeError("can't set attribute")
        elif ctrl.method == 'GET':
            result = getattr(self._vision, ctrl.name)
        elif ctrl.method == 'CALL':
            result = getattr(self._vision, ctrl.name)(*ctrl.args, **ctrl.kwargs)
        else:
            return None

        if isinstance(result, EasyVisionBase):
            result = result.__class__.__name__

        return self.send_data(result)

    def send_data(self, data):
        if data is None:
            return None
        data = pickle.dumps(data, protocol=-1)
        data_id = str(uuid.uuid4())
        self._pyroDaemon.datablobs[data_id] = data
        return data_id

    def freerun(self):
        """Capturing loop for freerun configuration"""
        try:
            self._running = True
            self._event.clear()
            self._exit_event.clear()
            self._frames = 0
            self._framestart = datetime.now()

            print('thread started')

            for result in self._vision:
                with self._result_lock:
                    self._result = result
                    self._result_ready = True
                    self._event.set()
                    self._frames += 1
                if not self._running:
                    break
        except:
            print('thread except')
            raise
        finally:
            print('thread finally')
            with self._result_lock:
                 self._running = False
            self._exit_event.set()

    @Pyro4.expose
    def fps(self):
        """Will return actual FPS calculated in freerun configuration. Useful for estimating network throughput."""
        if self._freerun:
            with self._result_lock:
                return self._frames / (datetime.now() - self._framestart).total_seconds()
        else:
            return None

    @Pyro4.expose
    def setup(self):
        """Calls setup on processor stack."""
        self._vision.setup()
        if self._freerun:
            thread = threading.Thread(target=self.freerun)
            thread.daemon = True
            thread.start()

    @Pyro4.expose
    def release(self):
        """Calls release on processor stack."""
        if self._freerun:
            self._running = False
            self._exit_event.wait()
        self._vision.release()

    @Pyro4.expose
    def getsockname(self):
        """Returns socket name used for return data transfer."""
        return self._pyroDaemon.blobsocket.getsockname()

    def get_last_result(self):
        """Helper method to get last result calculated in freerun configuration"""
        with self._result_lock:
            result_ready = self._result_ready
            running = self._running

        if not result_ready:
            if not running:
                return None
            self._event.wait()

        with self._result_lock:
            result = self._result
            self._result_ready = False
            self._event.clear()

        return result

    @Pyro4.expose
    def capture(self):
        """Captures processor stack result"""
        if self._freerun:
            result = self.get_last_result()
        else:
            result = self._vision.capture()
        return self.send_data(result)

    @Pyro4.expose
    def compute(self):
        """Captures Engine computation result"""
        if self._freerun:
            result = self.get_last_result()
        else:
            result = self._vision.compute()
        return self.send_data(result)

    @Pyro4.expose
    def hascall(self, name):
        return hasattr(getattr(self._vision, name), '__call__')

    @Pyro4.expose
    def hasattr(self, name):
        return hasattr(self._vision, name)


class ServerDaemon(Pyro4.core.Daemon):
    """Custom  Server Daemon that supports raw sockets for large data transfer
    """

    def __init__(self, host=None, port=0):
        super(ServerDaemon, self).__init__(host, port)
        host, _ = self.transportServer.sock.getsockname()
        self.blobsocket = Pyro4.socketutil.createSocket(bind=(host, 0), timeout=Pyro4.config.COMMTIMEOUT, nodelay=False)
        self.datablobs = {}

    def close(self):
        self.blobsocket.close()
        super(ServerDaemon, self).close()

    def requestLoop(self, loopCondition=lambda: True):
        while loopCondition:
            rs = [self.blobsocket]
            rs.extend(self.sockets)
            rs, _, _ = select.select(rs, [], [], 3)
            daemon_events = []
            for sock in rs:
                if sock in self.sockets:
                    daemon_events.append(sock)
                elif sock is self.blobsocket:
                    self.handle_blob_connect(sock)
            if daemon_events:
                self.events(daemon_events)

    def handle_blob_connect(self, sock):
        csock, caddr = sock.accept()
        thread = threading.Thread(target=self.blob_client, args=(csock,))
        thread.daemon = True
        thread.start()

    def blob_client(self, csock):
        """Will send back a requested data blob referenced by UUID"""
        while True:
            try:
                file_id = Pyro4.socketutil.receiveData(csock, 36).decode()
                if file_id is None:
                    return
                data = self.datablobs.pop(file_id)
                csock.sendall("{:016}".format(len(data)).encode('utf-8'))
                csock.sendall(data)
            except:
                break
        csock.close()


class Server(object):
    """Server of the EasyVision objects or any other objects supplied to the server"""

    def __init__(self, name, vision, host='localhost', port=0, freerun=True, proxy_class=ProxyVision, objects=None):
        """

        :param name: Name of the EasyVision object
        :param vision: EasyVision object (One of vision/processors or engine
        :param host: Host name to serve on
        :param port: Port number to serve on
        :param freerun: Whether to start capturing loop
        :param proxy_class: Specify a custom vision object remote proxy class
        :param objects: misc objects to be remotely served as a dictionary {"name": Object_or_class, ...}
        """
        if not isinstance(vision, VisionBase):
            raise TypeError("Vision must be VisionBase")

        self._name = name
        self._running = False
        self._host = host
        self._port = port
        self._objects = objects if objects else {}

        self._proxy = proxy_class(vision, freerun)

    def run(self):
        """Starts a Pyro4 ServerDaemon request loop.
        The request loop can be stopped by calling stop()

        :return: None
        """
        with ServerDaemon(host=self._host, port=self._port) as daemon:
            ns = Pyro4.locateNS()

            uri = daemon.register(self._proxy)
            ns.register(self._name, uri)

            for name, obj in self._objects.items():
                uri = daemon.register(obj)
                ns.register(name, uri)

            self._running = True
            daemon.requestLoop(loopCondition=lambda: self._running)

    def stop(self):
        """Stops the Server if called from another thread.

        :return: None
        """
        self._running = False
