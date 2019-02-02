# -*- coding: utf-8 -*-
from .base import EasyVisionBase
from collections import namedtuple
import Pyro.core
import Pyro.naming
from Pyro.errors import PyroError,NamingError

Command = namedtuple('Command', 'name method args kwargs')


class ProxyVision(Pyro.core.ObjBase):
    """Proxy object for accepting and handling remote commands.
    """

    def __init__(self, vision, freerun):
        if not isinstance(vision, EasyVisionBase):
            raise TypeError("Vision must be EasyVisionBase")

        self._vision = vision
        self._freerun = freerun

        super(ProxyVision, self).__init__()

    def command(self, ctrl):
        """Will handle remote commands"""
        try:
            result = None
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
            elif ctrl.method == 'GET':
                result = getattr(self._vision, ctrl.name)
            elif ctrl.method == 'CALL':
                result = getattr(self._vision, ctrl.name)(*ctrl.args, **ctrl.kwargs)
            else:
                pass

            if isinstance(result, EasyVisionBase):
                result = result.name

            return result
        except Exception as e:
            return e

    def setup(self):
        """In freerun configuration will run the computation/capturing loop"""
        # TODO implement threading loop
        pass


class Server(object):
    """Server of the """

    def __init__(self, name, vision, freerun=False, proxy_class=ProxyVision):
        if not isinstance(vision, EasyVisionBase):
            raise TypeError("Vision must be EasyVisionBase")

        self._name = name

        self._proxy = proxy_class(vision, freerun)

    def run(self):
        Pyro.core.initServer()
        ns = Pyro.naming.NameServerLocator().getNS()
        daemon = Pyro.core.Daemon()
        daemon.useNameServer(ns)

        try:
            ns.unregister(self._name)
        except NamingError:
            pass

        uri = daemon.connect(self._proxy, self._name)
        self._proxy.setup()

        try:
            daemon.requestLoop()
        except KeyboardInterrupt:
            pass

        daemon.disconnect(self._proxy)
        daemon.shutdown()