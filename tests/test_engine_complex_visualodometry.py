#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from pytest import raises, approx, mark
from EasyVision.exceptions import *
from EasyVision.engine import *
from EasyVision.processors import *
from EasyVision.vision import *
import cv2
import numpy as np
from .common import *


@mark.complex
def test_visual_odometry_kitti():
    common_test_visual_odometry_kitti('ORB', mp=False, ocl=True, color=cv2.COLOR_BGR2GRAY)


@mark.complex
def test_visual_odometry_indoor():
    traj = np.zeros((600, 600, 3), dtype=np.uint8)
    cam = CalibratedCamera(VideoCapture(dataset_note9), camera_note9, display_results=False, enabled=False)
    with VisualOdometryEngine(cam, display_results=True, debug=False, feature_type='ORB') as engine:
        for frame, pose in engine:
            if not pose:
                continue

            t = pose.translation
            draw_x, draw_y = int(t[0]) + 300, int(t[2]) + 300

            cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
            cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
            cv2.putText(traj, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Trajectory', traj)
            cv2.waitKey(1)

    cv2.waitKey(0)


@mark.complex
def test_visual_odometry_dataset():
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    with open(sequence_tum + "times.txt") as f:
        images = ['{}images/{}.jpg'.format(sequence_tum, line.split()[0]) for line in f.readlines()]

    cam = CalibratedCamera(ImagesReader(images), camera_tum, display_results=False, enabled=False)
    with VisualOdometryEngine(cam, display_results=True, debug=False, feature_type='GFTT') as engine:
        for frame, pose in engine:
            if not pose:
                continue

            t = pose.translation
            draw_x, draw_y = int(t[0]) + 300, int(t[2]) + 300

            cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
            cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
            text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
            cv2.putText(traj, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Trajectory', traj)
            cv2.waitKey(1)

    cv2.waitKey(0)