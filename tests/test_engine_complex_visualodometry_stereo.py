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


@mark.main
def test_visual_odometry_stereo():
    camera = StereoCamera(camera_kitti, camera_kitti_right, R_kitti, T_kitti, None, None, None)
    FEATURE_TYPE = 'ORB'

    images_kitti_l = ['test_data/kitti00/image_0/{}.png'.format(str(i).zfill(6)) for i in xrange(3)]
    images_kitti_r = ['test_data/kitti00/image_1/{}.png'.format(str(i).zfill(6)) for i in xrange(3)]

    cam_left = CalibratedCamera(ImageTransform(ImagesReader(images_kitti_l), ocl=False, color=cv2.COLOR_BGR2GRAY), camera.left)
    cam_right = CalibratedCamera(ImageTransform(ImagesReader(images_kitti_r), ocl=False, color=cv2.COLOR_BGR2GRAY), camera.right)
    cam = CalibratedStereoCamera(cam_left, cam_right, camera)
    cam = CalibratedStereoCamera(
            FeatureExtraction(cam_left, FEATURE_TYPE),
            FeatureExtraction(cam_right, FEATURE_TYPE),
            camera)
    with VisualOdometryStereoEngine(cam, display_results=False, debug=False) as engine:
        for i, _ in enumerate(engine):
            if i > 1:
                break


@mark.complex
def test_visual_odometry_kitti_stereo():
    traj = np.zeros((600, 600, 3), dtype=np.uint8)

    pose = "00"
    images_kitti_l = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_0/{}.png'.format(pose, str(i).zfill(6)) for i in xrange(NUM_IMAGES)]
    images_kitti_r = ['d:/datasets/data_odometry_gray/dataset/sequences/{}/image_1/{}.png'.format(pose, str(i).zfill(6)) for i in xrange(NUM_IMAGES)]
    gt_path_kitti = "d:/datasets/data_odometry_gray/dataset/poses/{}.txt".format(pose)

    with open(gt_path_kitti) as f:
        ground_truth = [[float(i) for i in line.split()] for line in f.readlines()]

    error = 0
    camera = StereoCamera(camera_kitti, camera_kitti_right, R_kitti, T_kitti, None, None, None)

    FEATURE_TYPE = 'ORB'

    cam_left = CalibratedCamera(ImageTransform(ImagesReader(images_kitti_l), ocl=False, _color=cv2.COLOR_BGR2GRAY), camera.left)
    cam_right = CalibratedCamera(ImageTransform(ImagesReader(images_kitti_r), ocl=False, _color=cv2.COLOR_BGR2GRAY), camera.right)
    cam = CalibratedStereoCamera(
            FeatureExtraction(cam_left, FEATURE_TYPE),
            FeatureExtraction(cam_right, FEATURE_TYPE),
            camera)
    with VisualOdometryStereoEngine(cam, display_results=True, debug=True, feature_type=FEATURE_TYPE) as engine:
        for img_id, _ in enumerate(images_kitti_l):
            true_x = ground_truth[img_id][3]
            true_y = ground_truth[img_id][7]
            true_z = ground_truth[img_id][11]

            x_prev, y_prev, z_prev = true_x, true_y, true_z

            frame, pose = engine.compute()
            if pose:
                t = pose.translation / 1000

                error += np.sqrt((true_x - t[0]) ** 2 + 0 * (true_y - t[1]) ** 2 + (true_z - t[2]) ** 2)

                draw_x, draw_y = int(t[0]) + 290, int(t[2]) + 90
                dtrue_x, dtrue_y = int(true_x) + 290, int(true_z) + 90

                cv2.circle(traj, (draw_x, draw_y), 1, (img_id * 255 / 4540, 255 - img_id * 255 / 4540, 0))
                cv2.circle(traj, (dtrue_x, dtrue_y), 1, (0, 0, 255))
                cv2.rectangle(traj, (0, 0), (600, 60), (0, 0, 0), -1)
                text = "pose: x=%2fm y=%2fm z=%2fm" % (t[0], t[1], t[2])
                cv2.putText(traj, text, (20, 11), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "true: x=%2fm y=%2fm z=%2fm" % (true_x, true_y, true_z)
                cv2.putText(traj, text, (20, 22), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                text = "cumulative error: %2f " % error
                cv2.putText(traj, text, (20, 44), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            cv2.imshow('Trajectory', traj)
            if cv2.waitKey(1) == 27:
                break

    cv2.waitKey(0)
