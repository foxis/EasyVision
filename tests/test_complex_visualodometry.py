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


camera = PinholeCamera.from_parameters((1241.0, 376.0), (718.8560, 718.8560), (607.1928, 185.2157), [0.0, 0.0, 0.0, 0.0, 0.0])
camera1 = PinholeCamera.from_parameters((1920, 1080), (1920/2, 1080/2), (1920/2, 1080/2), [0.0, 0.0, 0.0, 0.0, 0.0])
camera2 = PinholeCamera.from_parameters((1280, 1024),
    (1280 * 0.535719308086809, 1024 * 0.669566858850269),
    (1280 * 0.493248545285398, 1024 * 0.500408664348414),
    [0.897966326944875 , 0.0, 0.0, 0.0, 0.0])


@mark.complex
def test_visual_odometry_kitti():
    traj = np.zeros((600,600,3), dtype=np.uint8)
    NUM_IMAGES = 1591
    pose = "00"
    images = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in xrange(NUM_IMAGES)]
    gt_path = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)

    with open(gt_path) as f:
        ground_truth = [[float(i) for i in line.split()] for line in f.readlines()]

    error = 0
    with CalibratedCamera(
        ImageTransform(
            ImagesVision(images, img_args=()),
            ocl=True, color=cv2.COLOR_BGR2GRAY, enabled=True),
        camera, display_results=False, enabled=False) as cam:
        with VisualOdometryEngine(cam, display_results=True, debug=False, feature_type='ORB') as engine:
            for img_id, _ in enumerate(images):
                true_x = ground_truth[img_id][3]
                true_y = ground_truth[img_id][7]
                true_z = ground_truth[img_id][11]

                if img_id > 0:
                    scale = np.sqrt((true_x - x_prev) ** 2 + (true_y - y_prev) ** 2 + (true_z - z_prev) ** 2)
                    x_prev, y_prev, z_prev = true_x, true_y, true_z
                else:
                    scale = 1.0
                    x_prev, y_prev, z_prev = true_x, true_y, true_z

                frame, pose = engine.compute(absolute_scale=scale)
                if not pose:
                    continue

                t = pose.translation

                error += np.sqrt((true_x - t[0]) ** 2 + (true_y - t[1]) ** 2 + (true_z - t[2]) ** 2)

                draw_x, draw_y = int(t[0])+290, int(t[2])+90
                dtrue_x, dtrue_y = int(true_x)+290, int(true_z)+90

                cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0))
                cv2.circle(traj, (dtrue_x, dtrue_y), 1, (0, 0, 255))
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "pose: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "true: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
                cv2.putText(traj, text, (20, 22), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "scale: %2f" % scale
                cv2.putText(traj, text, (20, 33), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "cumulative error: %2f " % error
                cv2.putText(traj, text, (20, 44), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                cv2.imshow('Trajectory', traj)
                cv2.waitKey(1)

    cv2.waitKey(0)


@mark.complex
def test_visual_odometry_indoor():
    traj = np.zeros((600,600,3), dtype=np.uint8)
    NUM_IMAGES = 1591

    with CalibratedCamera(MonocularVision("d:\datasets\VID_20181217_163202.mp4"), camera1, display_results=False, enabled=False) as cam:
        with VisualOdometryEngine(cam, display_results=True, debug=False, feature_type='GFTT') as engine:
            for frame, pose in engine:
                if not pose:
                    continue

                t = pose.translation
                draw_x, draw_y = int(t[0])+300, int(t[2])+300

                cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                cv2.imshow('Trajectory', traj)
                cv2.waitKey(1)

    cv2.waitKey(0)


@mark.complex
def test_visual_odometry_dataset():
    traj = np.zeros((600,600,3), dtype=np.uint8)

    sequence = 'd:/datasets/vision.in.tum.de/sequence_50/'
    with open(sequence + "times.txt") as f:
        images = ['{}images/{}.jpg'.format(sequence, line.split()[0])  for line in f.readlines()]

    with CalibratedCamera(ImagesVision(images), camera2, display_results=False, enabled=False) as cam:
        with VisualOdometryEngine(cam, display_results=True, debug=False, feature_type='GFTT') as engine:
            for frame, pose in engine:
                if not pose:
                    continue

                t = pose.translation
                draw_x, draw_y = int(t[0])+300, int(t[2])+300

                cv2.circle(traj, (draw_x, draw_y), 1, (0, 255, 0), 1)
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                cv2.imshow('Trajectory', traj)
                cv2.waitKey(1)

    cv2.waitKey(0)