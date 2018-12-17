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


@mark.complex
def test_visual_odometry():
    traj = np.zeros((600,600,3), dtype=np.uint8)
    NUM_IMAGES = 1591
    pose = "09"
    images = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in xrange(NUM_IMAGES)]
    gt_path = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)

    with open(gt_path) as f:
        ground_truth = [[float(i) for i in line.split()] for line in f.readlines()]

    with CalibratedCamera(ImagesVision(images, img_args=(0,)), camera, display_results=False, enabled=False) as cam:
        with VisualOdometryEngine(cam, display_results=True, debug=False, feature_type='GFTT') as engine:
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

                draw_x, draw_y = int(t[0])+290, int(t[2])+90
                dtrue_x, dtrue_y = int(true_x)+290, int(true_z)+90

                cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0), 1)
                cv2.circle(traj, (dtrue_x, dtrue_y), 1, (0, 0, 255), 2)
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
                cv2.putText(traj, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                text = "scale: %2fm " % scale
                cv2.putText(traj, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

                cv2.imshow('Trajectory', traj)
                cv2.waitKey(1)

    cv2.waitKey(0)
