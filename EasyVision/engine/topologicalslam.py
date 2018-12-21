# -*- coding: utf-8 -*-
from .base import EngineBase
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction, CalibratedCamera, FeatureMatchingMixin
from .bowvocabulary import BOWMatchingMixin
import cv2
import numpy as np


class TopologicalSLAMEngine(FeatureMatchingMixin, BOWMatchingMixin, EngineBase):
    Keyframe = namedtuple('Keyframe', ['image', 'points', 'descriptors', 'bow'])
    Pose = namedtuple('Pose', ['rotation', 'translation'])
    Node = namedtuple('Node', ['keyframe', 'transitions'])
    Transition = namedtuple('transition', ['current', 'target', 'pose', 'control'])

    def __init__(self, vision, vocabulary, feature_type, pose=None, min_features=5000, debug=False, display_results=False, *args, **kwargs):
        if not isinstance(vision, CalibratedCamera):
            raise TypeError("Vision must be CalibratedCamera")

        defaults = dict()

        if feature_type == 'ORB':
            defaults['nfeatures'] = 10000
            #defaults['scoreType'] = cv2.ORB_FAST_SCORE
            defaults['nlevels'] = 16
            defaults.update(kwargs)

        self._feature_type = feature_type
        _vision = FeatureExtraction(vision, feature_type=feature_type, extract=True, **defaults)

        if _vision.get_source("CalibratedCamera") is None:
            raise ValueError("vision processor stack must contain CalibratedCamera")

        self.camera = _vision.get_source("CalibratedCamera").camera
        self._last_image = None
        self._last_kps = None
        self._last_features = None
        self.pose = pose

        super(TopologicalSLAMEngine, self).__init__(_vision, debug=debug, display_results=display_results, *args, **kwargs)

        self.initBOW(None, self._matcher_h, vocabulary)

    def compute(self, absolute_scale=1.0):
        frame = self.vision.capture()
        current_image = frame.images[0]

        pose = self._compute_match(current_image, absolute_scale) if self._extract else self._compute_track(current_image, absolute_scale)

        self._last_image = current_image

        if self.debug:
            img = current_image.image
            for x, y in self._last_kps:
                cv2.circle(img, (x, y), 5, (0, 255, 0))
        if self.display_results or self.debug:
            cv2.imshow(self.name, current_image.image)

        return frame, pose

    def _compute_match(self, current_image, absolute_scale):
        if not self._last_image:
            self._last_kps = np.array([x.pt for x in current_image.features.points], dtype=np.float32)
        else:
            M = self._match_features(self._last_features, current_image.features)

            if M is None:
                return self._pose
            last, current = M
            #if len(current) < self._min_features:
            #    return self._pose

            self._last_kps = current

            if self.debug:
                for a, b in zip(last, current):
                    cv2.line(current_image.image, (a[0], a[1]), (b[0], b[1]), (0, 0, 255))

            E, mask = cv2.findEssentialMat(current, last,
                                           focal=self.camera.focal_point[0], pp=self.camera.center,
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, current, last,
                                            focal=self.camera.focal_point[0], pp=self.camera.center)
            if self._pose:
                self._pose = self._pose._replace(translation=self._pose.translation + absolute_scale * self._pose.rotation.dot(t),
                                                rotation=R.dot(self._pose.rotation))
            else:
                self._pose = self.Pose(R, t)

        self._last_features = current_image.features


        return self._pose

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value):
        if not isinstance(value, self.Pose) and value is not None:
            raise TypeError("Pose must be of type VisualOdometryEngine.Pose")
        self._pose = value

    def release(self):
        super(VisualOdometryEngine, self).release()

    @property
    def description(self):
        return "Visual Odometry inspired by https://github.com/uoip/monoVO-python"

    @property
    def capabilities(self):
        return {}

    def _match_features(self, featuresA, featuresB, ratio=0.7, reprojThresh=5.0, distance_thresh=30, min_matches=5):
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB

        matches = super(TopologicalSLAMEngine, self)._match_features(descriptorsA, descriptorsB, self._feature_type, ratio, distance_thresh, min_matches)

        if matches is None or not matches:
            return None

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

        return ptsA, ptsB