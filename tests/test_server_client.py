#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from common import *
from EasyVision.server import Server
import threading as mt


@mark.main
def test_server_client():
    """This test requires Pyro NameServer started"""
    
    vision = ImagesReader(images_left)
    cam = CalibratedCamera(vision, left_camera)

    server = Server('Left Camera', cam)

    t = mt.Thread(target=server.run)
    t.start()

    cap = PyroCapture('Left Camera')
    with cap as vision:
        for idx, img in enumerate(vision):
            assert(idx > len(images_left))

    server.stop()
