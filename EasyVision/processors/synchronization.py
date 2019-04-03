# -*- coding: utf-8 -*-
"""Allows to synchronize capture using a callback.
"""

from .base import *
import Pyro4
import multiprocessing as mp


class Synchronize(ProcessorBase):
    """ Synchronization processor that allows to synchronize two processing streams,
    that possibly run on different machines.

    Basically it accepts a callable which will be called before capturing a frame
    and it should block until all processing streams are in sync.
    """

    def __init__(self, vision, sync, *args, **kwargs):
        """Synchronize instance initialization

        :param vision: capturing source object
        :param sync: a blocking callable responsible for synchronization
        """

        self._sync = sync

        super(Synchronize, self).__init__(vision, *args, **kwargs)

    @property
    def description(self):
        return "Processing stream synchronizing processor"

    def process(self, image):
        return self.source.process(image)

    def capture(self):
        super(ProcessorBase, self).capture()
        if self.enabled:
            self._sync()
        return self.source.capture()


class SimpleSynchronizer(object):
    """Simple Multiprocessing synchronizer that uses events and locks to synchronize two processes together.
    """

    def __init__(self, num_streams, timeout=30, *args, **kwargs):
        """SimpleSynchronizer object initialization.

        :param num_streams: Number of streams to synchronize
        :param timeout: timeout for the event waiting.
        """
        self._events = tuple(mp.Event() for _ in range(num_streams))
        self._lock = mp.Lock()
        self._timeout = timeout
        for i in self._events:
            i.clear()

    def __call__(self):
        """Will find an unset event, set it and wait for all events.
        After all events are set, will clear the one event that it set previously.
        In result every caller(up to num_streams) will wait for every other caller to call this
        synchronizer and release control to the stream afterwards.

        Will raise a TimeoutError exception if timeout occurs waiting for an event.
        """
        with self._lock:
            my_event = -1
            for i, ev in enumerate(self._events):
                if not ev.is_set():
                    ev.set()
                    my_event = i
                    break

        for i in self._events:
            if i.wait(self._timeout):
                raise TimeoutError

        with self._lock:
            self._events[my_event].clear()


class PyroSynchronizer(object):
    """Pyro synchronizer proxy object. Will connect to a PyroSynchronizerObject
    specified by name and will relay synchronization task to it."""

    def __init__(self, name, nameserver=None):
        with Pyro4.locateNS(host=nameserver) as ns:
            uri = ns.lookup(name)
        self._proxy = Pyro4.Proxy(uri)

    def __call__(self):
        self._proxy.sync()


@Pyro4.behavior(instance_mode="single")
class PyroSynchronizerObject(SimpleSynchronizer):
    """A Pyro synchronizer object wrapper over SimpleSynchronizer to provide syncing functionality over Pyro"""

    def __init__(self, num_streams, *args, **kwargs):
        super(PyroSynchronizerObject, self).__init__(num_streams, *args, **kwargs)

    @Pyro4.expose
    def sync(self):
        self()

