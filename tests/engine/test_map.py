#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx
from EasyVision.engine.base import EngineBase, MapBase, Pose
from EasyVision.engine import OccupancyGridMap
import cv2


@pytest.mark.main
def test_map_base():
    with raises(TypeError):
        MapBase()


@pytest.mark.main
def test_map_load_plan():
    m = cv2.cvtColor(cv2.imread("test_data/gridmap.bmp"), cv2.COLOR_BGR2GRAY)

    start = Pose(0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[10], [0], [5]])
    target = Pose(0, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[325//2], [0], [5]])

    _map = OccupancyGridMap(m, scale=1, poses=[start])

    path = _map.plan(target, 11)

    #_map.draw(path)
    #cv2.waitKey(0)

    assert(path[0] == (10, 5))
    assert(path[-1] == (325 // 2, 5))
    assert(len(path) > 10)

