# -*- coding: utf-8 -*-
from .base import *
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction, CalibratedCamera, FeatureMatchingMixin, Features
import cv2
import numpy as np


class VisualOdometry3D2DEngine(FeatureMatchingMixin, OdometryBase):

    def __init__(self, vision, feature_type=None, pose=None, num_features=10000, reproj_thresh=0.3, *args, **kwargs):
        feature_extractor_provided = False
        if not isinstance(vision, ProcessorBase) and not isinstance(vision, VisionBase):
            raise TypeError("Vision must be either VisionBase or ProcessorBase")
        if isinstance(vision, ProcessorBase):
            if vision.get_source('CalibratedCamera') is None:
                raise TypeError("Vision must contain CalibratedCamera")

            if vision.get_source('FeatureExtraction') is not None:
                feature_type = vision.feature_type
                feature_extractor_provided = True
            elif not feature_type:
                raise TypeError("Feature type must be provided")

        if feature_type in ['FAST', 'GFTT']:
            raise ValueError('FAST and GFTT features not supported')

        defaults = {}
        if feature_type == 'ORB':
            defaults['nfeatures'] = num_features
            #defaults['scoreType'] = cv2.ORB_FAST_SCORE
            defaults['nlevels'] = 16
            defaults.update(kwargs)
            defaults.pop('enabled', None)
            defaults.pop('debug', None)
            defaults.pop('display_results', None)

        self._feature_type = feature_type
        _vision = FeatureExtraction(vision, feature_type=feature_type, **defaults) if not feature_extractor_provided else vision

        self._camera = _vision.camera
        self._last_image = None
        self._last_kps = None
        self._last_features = None
        self._pose = pose
        self._last_pose = None
        self._images = []

        self._ratio = 0.7
        self._distance_thresh = 30
        self._min_matches = 100
        self._reproj_thresh = reproj_thresh

        super(VisualOdometry3D2DEngine, self).__init__(_vision, *args, **kwargs)

    def compute(self, absolute_scale=1.0):
        frame = self.vision.capture()
        if not frame:
            return None

        self._images += [[frame.images[0].features, None, frame.images[0].image]]
        self._images = self._images[-3:]

        if len(self._images) == 1:
            return frame, self._pose

        featuresA, featuresB, features3D = self._calculate_3d(self._images[-2][0], self._images[-1][0], absolute_scale)
        if featuresA is not None and featuresB is not None and features3D is not None:
            #self._images[-2][0] = featuresA
            self._images[-2][1] = features3D
            #self._images[-1][0] = featuresB

        if len(self._images) == 3 and self._images[-3][1] is not None:
            # TODO filter those features that have similar distance from -3 and -2
            matches = self._match_features(self._images[-3][1].descriptors, self._images[-1][0].descriptors, self._feature_type, self._ratio, self._distance_thresh, self._min_matches)
            if matches is None or not matches:
                return frame, self._pose
            points_3d = np.float32([self._images[-3][1].points[m.queryIdx] for m in matches])
            points_2d = np.float32([self._images[-1][0].points[m.trainIdx].pt for m in matches])
            ret, r, t, _ = cv2.solvePnPRansac(points_3d, points_2d, self._camera.matrix, None, reprojectionError=5)
            R, _ = cv2.Rodrigues(r * -1)
            t *= -1
            if ret:
                if self._pose:
                    self._pose = self._pose._replace(translation=self._pose.translation + absolute_scale * self._pose.rotation.dot(t),
                                                     rotation=R.dot(self._pose.rotation))
                else:
                    self._pose = Pose(R, t)
                self._last_pose = Pose(R, t)

        if self.display_results and featuresA is not None and featuresB is not None:
            for i, img in enumerate(self._images):
                if i == 0 and self.debug:
                    last = [kp.pt for kp in featuresA.points]
                    current = [kp.pt for kp in featuresB.points]
                    for l, c in zip(last, current):
                        cv2.circle(img[2], (int(l[0]), int(l[1])), 1, (0, 0, 255))
                        cv2.line(img[2], (int(l[0]), int(l[1])), (int(c[0]), int(c[1])), (0, 0, 255))
                if i == 0:
                    cv2.imshow("stack %i" %i, img[2])

        return frame, self._pose

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def camera(self):
        return self._camera

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value):
        if not isinstance(value, Pose) and value is not None:
            raise TypeError("Pose must be of type Pose")
        self._pose = value

    @property
    def relative_pose(self):
        return self._last_pose

    @property
    def camera_orientation(self):
        pass

    @camera_orientation.setter
    def camera_orientation(self, value):
        pass

    @property
    def description(self):
        return "Monocular Visual Odometry (3D-2D)"

    @property
    def capabilities(self):
        return EngineCapabilities(
                (ProcessorBase, FeatureExtraction),
                (Frame, Pose),
                {'feature_type': ('FREAK', 'SURF', 'SIFT', 'ORB', 'KAZE', 'AKAZE')}
            )

    def _calculate_3d(self, featuresA, featuresB, scale):
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB

        matches = self._match_features(descriptorsA, descriptorsB, self._feature_type, self._ratio, self._distance_thresh, self._min_matches)

        if matches is None or not matches:
            return None, None, None

        umat_descriptors = isinstance(descriptorsA, cv2.UMat)
        if umat_descriptors:
            descriptorsA = descriptorsA.get()
            descriptorsB = descriptorsB.get()

        kpsA = [kpsA[m.queryIdx] for m in matches]
        kpsB = [kpsB[m.trainIdx] for m in matches]

        dA = [descriptorsA[m.queryIdx] for m in matches]
        dB = [descriptorsB[m.trainIdx] for m in matches]

        current = np.float32([kp.pt for kp in kpsB])
        last = np.float32([kp.pt for kp in kpsA])

        E, mask = cv2.findEssentialMat(current, last, focal=self._camera.focal_point[0], pp=self._camera.center,
                                       method=cv2.RANSAC, prob=0.999, threshold=self._reproj_thresh)

        kpsA = [kp for m, kp in zip(mask, kpsA) if m]
        kpsB = [kp for m, kp in zip(mask, kpsB) if m]
        dA = np.array([d for m, d in zip(mask, dA) if m])
        dB = np.array([d for m, d in zip(mask, dB) if m])

        last = np.float32([kp.pt for kp in kpsA])
        current = np.float32([kp.pt for kp in kpsB])

        ret, R, t, mask = cv2.recoverPose(E, current, last, focal=self._camera.focal_point[0], pp=self._camera.center)

        if not ret:
            return None, None, None

        kpsA = [kp for m, kp in zip(mask, kpsA) if m]
        kpsB = [kp for m, kp in zip(mask, kpsB) if m]
        dA = np.array([d for m, d in zip(mask, dA) if m], dtype=descriptorsA.dtype)
        dB = np.array([d for m, d in zip(mask, dB) if m], dtype=descriptorsB.dtype)

        last = np.float32([kp.pt for kp in kpsA])
        current = np.float32([kp.pt for kp in kpsB])

        P1 = np.dot(self._camera.matrix, np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
        P2 = np.dot(self._camera.matrix, np.hstack((R, t)))

        point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(current, axis=1), np.expand_dims(last, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        if self.debug:
            cv2.rectangle(self.features, (0, 0), (600, 600), (0, 0, 0), -1)

            scale = max(max(p[0], p[2]) for p in point_3d)
            yscale = max(p[1] for p in point_3d)
            scale = 50
            for p in point_3d:
                cv2.circle(self.features, (300 + int(300 * p[0] / scale), 600 - int(600 * p[2] / scale)), 1, (0, 0, 255 - int(200 * p[1] / yscale)))
            cv2.imshow("3D points", self.features)

        if umat_descriptors:
            dA = cv2.UMat(dA)
            dB = cv2.UMat(dB)

        return Features(kpsA, dA), Features(kpsB, dB), Features(point_3d, dB)

    def debug_changed(self, last, current):
        if current:
            cv2.namedWindow("3D points")
            self.features = np.zeros((600, 600, 3), dtype=np.uint8)
