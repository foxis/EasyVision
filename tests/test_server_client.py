#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.vision import *
from common import *
from EasyVision.server import Server
import threading as mt
import time


@mark.main
def test_server_client():
    """This test requires Pyro NameServer started"""

    vision = ImagesReader(images_left)
    cam = CalibratedCamera(vision, left_camera)

    server = Server('LeftCamera', cam)

    t = mt.Thread(target=server.run)
    t.daemon = True
    t.start()
    time.sleep(1)

    try:
        cap = PyroCapture('LeftCamera')
        with cap as vision:
            for idx, img in enumerate(vision):
                assert(isinstance(img, Frame))
                assert(idx < len(images_left) + 1)
                #assert(idx < 10)
    except:
        raise
    finally:
        server.stop()
