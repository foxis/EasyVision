# -*- coding: utf-8 -*-
from .base import VisionBase
from .exceptions import DeviceNotFound


class StereoPairVision(VisionBase):

    def __init__(self, pathleft, pathright):
        super(StereoPairVision, self).__init__()
