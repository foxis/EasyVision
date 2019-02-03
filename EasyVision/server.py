# -*- coding: utf-8 -*-
from .base import EasyVisionBase
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
        if not isinstance(vision, EasyVisionBase):
            raise TypeError("Vision must be EasyVisionBase")

        self._vision = vision
        self._freerun = freerun

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

        data = cPickle.dumps(result, protocol=-1)
        data_id = str(uuid.uuid4())
        self._pyroDaemon.datablobs[data_id] = data
        return data_id, self._pyroDaemon.blobsocket.getsockname()

    @Pyro4.expose
    def getsockname(self):
        return self._pyroDaemon.blobsocket.getsockname()

    def setup(self):
        """In freerun configuration will run the computation/capturing loop"""
        # TODO implement threading loop
        pass


class ServerDaemon(Pyro4.core.Daemon):
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
    """Server of the """

    def __init__(self, name, vision, host='localhost', port=0, freerun=False, proxy_class=ProxyVision):
        if not isinstance(vision, EasyVisionBase):
            raise TypeError("Vision must be EasyVisionBase")

        self._name = name
        self._running = False
        self._host = host
        self._port = port

        self._proxy = proxy_class(vision, freerun)

    def run(self):
        with ServerDaemon(host=self._host, port=self._port) as daemon:
            ns = Pyro4.locateNS()

            uri = daemon.register(self._proxy)
            ns.register(self._name, uri)

            self._proxy.setup()

            self._running = True
            daemon.requestLoop(loopCondition=lambda: self._running)

    def stop(self):
        self._running = False
