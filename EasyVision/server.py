# -*- coding: utf-8 -*-
from .base import EasyVisionBase
from .vision.base import VisionBase
from .engine.base import EngineBase
from collections import namedtuple
import threading
import Pyro4
import cPickle
import select
import uuid

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
    def command(self, data):
        """Will handle remote commands"""
        result = None
        ctrl = cPickle.loads(data)
        if ctrl.method == 'SET':
            cur_obj = self._vision
            last_obj = None
            while cur_obj or last_obj:
                last_obj, cur_obj = cur_obj, getattr(cur_obj, '_vision', None)

                if hasattr(last_obj, ctrl.name) and not hasattr(cur_obj, ctrl.name):
                    setattr(last_obj, ctrl.name, ctrl.args)
                    break
            if not cur_obj and not last_obj:
                raise AttributeError("can't set attribute")
            return None
        elif ctrl.method == 'GET':
            result = getattr(self._vision, ctrl.name)
        elif ctrl.method == 'CALL':
            result = getattr(self._vision, ctrl.name)(*ctrl.args, **ctrl.kwargs)
        else:
            pass

        if isinstance(result, EasyVisionBase):
            result = result.name

        return self.send_data(result)

    def send_data(self, data):
        data = cPickle.dumps(data, protocol=-1)
        data_id = str(uuid.uuid4())
        self._pyroDaemon.datablobs[data_id] = data
        return data_id, self._pyroDaemon.blobsocket.getsockname()

    def freerun(self):
        try:
            self._running = True
            self._event.clear()
            self._exit_event.clear()

            for result in self._vision:
                with self._result_lock:
                    self._result = result
                    self._result_ready = True
                    self._event.set()
                if not self._running:
                    break
        except:
            raise
        finally:
            self._running = False
            self._exit_event.set()

    @Pyro4.expose
    def setup(self):
        self._vision.setup()
        if self._freerun:
            thread = threading.Thread(target=self.freerun)
            thread.daemon = True
            thread.start()

    @Pyro4.expose
    def release(self):
        if self._freerun:
            self._running = False
            self._exit_event.wait()
        self._vision.release()

    @Pyro4.expose
    def getsockname(self):
        return self._pyroDaemon.blobsocket.getsockname()

    def get_last_result(self):
        with self._result_lock:
            result_ready = self._result_ready

        if not result_ready:
            self._event.wait()

        with self._result_lock:
            result = self._result
            self._result_ready = False
            self._event.clear()

        return result

    @Pyro4.expose
    def capture(self):
        if self._freerun:
            result = self.get_last_result()
        else:
            result = self._vision.capture()
        return self.send_data(result)

    @Pyro4.expose
    def compute(self):
        if self._freerun:
            result = self.get_last_result()
        else:
            result = self._vision.compute()
        return self.send_data(result)


class ServerDaemon(Pyro4.core.Daemon):
    """
    Custom  Server Daemon that supports raw sockets for large data transfer
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
        while True:
            file_id = Pyro4.socketutil.receiveData(csock, 36).decode()
            if file_id is None:
                return
            data = self.datablobs.pop(file_id)
            csock.sendall(data)
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
        if not isinstance(vision, EasyVisionBase):
            raise TypeError("Vision must be EasyVisionBase")

        self._name = name
        self._running = False
        self._host = host
        self._port = port
        self._objects = objects if objects else {}

        self._proxy = proxy_class(vision, freerun)

    def run(self):
        """

        :return:
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
        """

        :return:
        """
        self._running = False
