# -*- coding: utf-8 -*-
"""Implements Pyro4 proxy object for remote map
"""
from EasyVision.engine.base import MapBase
from EasyVision.engine.base import Pose
import Pyro4


class PyroMap(MapBase):
    """Class implementing Proxy for remote Pyro object representing a map.
    Should be used with conjunction with VisualOdometry engine.
    Uses 3d feature points of the Pose. Z coordinate denotes forward, and Y coordinate - up.

    """

    def __init__(self, name, nameserver=None, *args, **kwargs):
        """Instance initialization.

        :param name: Name of the remote pyro object representing the map
        :param nameserver: hostname of the Pyro NameServer
        """
        with Pyro4.locateNS(host=nameserver) as ns:
            uri = ns.lookup(self._name)

        self._proxy = Pyro4.Proxy(uri)

        super(PyroMap, self).__init__(*args, **kwargs)

    @property
    def map_raw(self):
        return self._proxy.map_raw()

    @property
    def pose(self):
        return self._proxy.pose()

    @property
    def path(self):
        return self._proxy.path()

    def __iter__(self):
        return self._proxy.poses()

    def __len__(self):
        return self._proxy.poses_len()

    def setup(self):
        super(PyroMap, self).setup()
        self._proxy.setup()

    def release(self):
        super(PyroMap, self).release()
        self._proxy.release()

    @property
    def description(self):
        return "Remote Map proxy using Pyro"

    def update(self, pose, **kwargs):
        return self._proxy.update(pose, **kwargs)

    def plan(self, target, radius, **kwargs):
        return self._proxy.plan(target, radius, **kwargs)
