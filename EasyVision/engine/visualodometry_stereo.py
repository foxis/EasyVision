# -*- coding: utf-8 -*-
from .base import *
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction, StereoCamera, CalibratedStereoCamera, FeatureMatchingMixin
import cv2
import numpy as np


class VisualOdometryStereoEngine(FeatureMatchingMixin, OdometryBase):

    def __init__(self, vision, feature_type=None, pose=None, num_features=10000, ratio=None, distance_thresh=None, reproj_thresh=None, *args, **kwargs):
        feature_extractor_provided = False
        if not isinstance(vision, ProcessorBase) and not isinstance(vision, CalibratedStereoCamera):
            raise TypeError("Vision must be either CalibratedStereoCamera or ProcessorBase")

        if isinstance(vision, ProcessorBase):
            if vision.get_source('CalibratedStereoCamera') is None:
                raise TypeError("Vision must contain CalibratedStereoCamera")

            if vision.get_source('FeatureExtraction') is not None:
                fe = vision.get_source('FeatureExtraction')
                assert(fe[0].feature_type == fe[1].feature_type)

                feature_type = fe[0].feature_type
                feature_extractor_provided = True
            elif not feature_type:
                raise TypeError("Feature type must be provided")
        else:
            raise TypeError("Vision must contain CalibratedStereoCamera")

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
        assert(isinstance(self.camera, StereoCamera))
        self._last_frame = None
        self._pose = pose
        self._last_pose = None
        self._last_3dfeatures = None

        self._ratio = 0.7
        self._distance_thresh = 100
        self._min_matches = 10
        self._reproj_thresh = 0.3

        if feature_type == 'ORB':
            self._distance_thresh = 80
        elif feature_type == 'FREAK':
            self._distance_thresh = 90
        elif feature_type == 'SIFT':
            self._distance_thresh = 200
        elif feature_type == 'BRISK':
            self._distance_thresh = 80

        if ratio is not None:
            self._ratio = ratio
        if distance_thresh is not None:
            self._distance_thresh = distance_thresh
        if reproj_thresh is not None:
            self._reproj_thresh = reproj_thresh

        super(VisualOdometryStereoEngine, self).__init__(_vision, *args, **kwargs)

    def compute(self):
        frame = self.vision.capture()
        if not frame:
            return None

        stereo_features = self._calculate_3d(frame.images[0].features, frame.images[1].features, frame.images[0].image)

        if self._last_3dfeatures is not None:
            matches = self._match_features(self._last_3dfeatures[1], frame.images[0].features.descriptors,
                    self._feature_type, self._ratio, self._distance_thresh/3, self._min_matches)

            if self.debug:
                text = "Number of 3d features: %i" % len(self._last_3dfeatures[0])
                cv2.putText(self.features, text, (20, 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            if matches is None or not matches:
                self._last_3dfeatures = stereo_features
                if self.debug:
                    cv2.imshow("3D points", self.features)

                return frame, self._pose

            if self.debug:
                text = "Number of matching features: %i" % len(matches)
                cv2.putText(self.features, text, (20, 25), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)

            points_3d = np.float32([self._last_3dfeatures[0][m.queryIdx] for m in matches])
            points_2d = np.float32([frame.images[0].features.points[m.trainIdx].pt for m in matches])
            ret, r, t, _ = cv2.solvePnPRansac(points_3d, points_2d, self._camera.left.matrix, None, reprojectionError=5)
            R, _ = cv2.Rodrigues(r * -1)
            t *= -1.4
            if ret:
                if self._pose:
                    self._pose = self._pose._replace(translation=self._pose.translation + self._pose.rotation.dot(t),
                                                     rotation=R.dot(self._pose.rotation))
                else:
                    self._pose = Pose(R, t)
                self._last_pose = Pose(R, t)

        if self.debug:
            cv2.imshow("3D points", self.features)

        self._last_3dfeatures = stereo_features
        return frame, self._pose

    def _calculate_3d(self, featuresA, featuresB, img):
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB
        matches = self._match_features(descriptorsA, descriptorsB, self._feature_type,
                        self._ratio, self._distance_thresh, self._min_matches)

        if matches is None or not matches:
            return None

        umat_descriptors = isinstance(descriptorsA, cv2.UMat)
        if umat_descriptors:
            descriptorsA = descriptorsA.get()
            descriptorsB = descriptorsB.get()

        kpsA = [kpsA[m.queryIdx] for m in matches]
        kpsB = [kpsB[m.trainIdx] for m in matches]
        mask = [abs(a.pt[1] - b.pt[1]) < 2 and a.pt[0] > b.pt[0] for a, b in zip(kpsA, kpsB)]

        dA = np.array([descriptorsA[m.queryIdx] for M, m in zip(mask, matches) if M], dtype=descriptorsA.dtype)
        dB = np.array([descriptorsB[m.trainIdx] for M, m in zip(mask, matches) if M], dtype=descriptorsB.dtype)

        kpsA = [p for M, p in zip(mask, kpsA) if M]
        kpsB = [p for M, p in zip(mask, kpsB) if M]

        left = np.float32([kp.pt for kp in kpsA])
        right = np.float32([kp.pt for kp in kpsB])

        #E, mask = cv2.findEssentialMat(left, right, focal=self._camera.left.focal_point[0], pp=self._camera.left.center,
        #                               method=cv2.RANSAC, prob=0.999, threshold=self._reproj_thresh)

        #kpsA = [kp for m, kp in zip(mask, kpsA) if m]
        #kpsB = [kp for m, kp in zip(mask, kpsB) if m]
        #dA = np.array([d for m, d in zip(mask, dA) if m])
        #dB = np.array([d for m, d in zip(mask, dB) if m])

        #left = np.float32([kp.pt for kp in kpsA])
        #right = np.float32([kp.pt for kp in kpsB])

        if self.debug:
            img = img.get() if isinstance(img, cv2.UMat) else img
            for l, r in zip(left, right):
                cv2.line(img, (int(l[0]), int(l[1])), (int(r[0]), int(r[1])), (0, 0, 255))
            cv2.imshow("Left feature matches to right", img)

        if self._camera.left.projection is None or self._camera.left.projection is None:
            P1 = np.dot(self._camera.left.matrix, np.hstack((np.eye(3, 3), np.zeros((3, 1)))))
            P2 = np.dot(self._camera.right.matrix, np.hstack((self._camera.R, self._camera.T)))
        else:
            P1 = self._camera.left.projection
            P2 = self._camera.right.projection

        point_4d_hom = cv2.triangulatePoints(P1, P2, np.expand_dims(left, axis=1), np.expand_dims(right, axis=1))
        point_4d = point_4d_hom / np.tile(point_4d_hom[-1, :], (4, 1))
        point_3d = point_4d[:3, :].T

        if self.debug:
            cv2.rectangle(self.features, (0, 0), (600, 600), (0, 0, 0), -1)

            scale = max(max(p[0], p[2]) for p in point_3d)
            yscale = max(p[1] for p in point_3d)
            scale = 50000
            yscale = 1000
            for p in point_3d:
                c = max(0, 255 - int(200 * p[1] / yscale))
                cv2.circle(self.features, (300 + int(300 * p[0] / scale), 600 - int(600 * p[2] / scale)), 1, (0, 0, c))

        if umat_descriptors:
            dA = cv2.UMat(dA)
            dB = cv2.UMat(dB)

        return point_3d, dA

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
        return "Stereo Visual Odometry"

    @property
    def capabilities(self):
        return EngineCapabilities(
                (ProcessorBase, CalibratedStereoCamera, FeatureExtraction),
                (Frame, Pose),
                {'feature_type': ('FREAK', 'SURF', 'SIFT', 'ORB', 'KAZE', 'AKAZE')}
            )

    def debug_changed(self, last, current):
        if current:
            cv2.namedWindow("3D points")
            self.features = np.zeros((600, 600, 3), dtype=np.uint8)