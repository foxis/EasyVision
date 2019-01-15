# -*- coding: utf-8 -*-
from .base import *
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction, CalibratedCamera, FeatureMatchingMixin
import cv2
import numpy as np


class VisualOdometry2DEngine(FeatureMatchingMixin, OdometryBase):

    def __init__(self, vision, feature_type=None, pose=None, num_features=10000, min_features=1000, reproj_thresh=0.3, debug=False, display_results=False, *args, **kwargs):
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

        if feature_type == 'FAST':
            defaults = dict(threshold=25, nonmaxSuppression=True)
            self._extract = False
        elif feature_type == 'GFTT':
            defaults = dict(maxCorners=3000, qualityLevel=0.01, blockSize=3, minDistance=1)
            self._extract = False
        else:
            self._extract = True
            defaults = dict()

        if feature_type == 'ORB':
            defaults['nfeatures'] = num_features
            #defaults['scoreType'] = cv2.ORB_FAST_SCORE
            defaults['nlevels'] = 16
            defaults.update(kwargs)

        self._feature_type = feature_type
        _vision = FeatureExtraction(vision, feature_type=feature_type, extract=self._extract, **defaults) if not feature_extractor_provided else vision

        self._min_features = min_features
        self._reproj_thresh = reproj_thresh

        self._lk_params = dict(
            winSize=(21, 21),
            #maxLevel = 3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.camera = _vision.camera
        self._last_image = None
        self._last_kps = None
        self._last_features = None
        self._last_pose = None
        self._pose = pose

        super(VisualOdometry2DEngine, self).__init__(_vision, debug=debug, display_results=display_results, *args, **kwargs)

    def compute(self, absolute_scale=1.0):
        frame = self.vision.capture()
        if not frame:
            return None
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
                                           method=cv2.RANSAC, prob=0.999, threshold=self._reproj_thresh)
            #last = np.float32([kp for m, kp in zip(mask, last) if m])
            #current = np.float32([kp for m, kp in zip(mask, current) if m])
            _, R, t, mask = cv2.recoverPose(E, current, last,
                                            focal=self.camera.focal_point[0], pp=self.camera.center)
            if self._pose:
                self._pose = self._pose._replace(translation=self._pose.translation + absolute_scale * self._pose.rotation.dot(t),
                                                rotation=R.dot(self._pose.rotation))
            else:
                self._pose = Pose(R, t)
            self._last_pose = Pose(R, t)

        self._last_features = current_image.features
        return self._pose

    def _compute_track(self, current_image, absolute_scale):
        self.vision.enable = False
        if not self._last_image:
            self._last_kps = np.array([x.pt for x in current_image.features.points], dtype=np.float32)
        else:
            self._last_kps, cur_kps = self._track_features(self._last_image.image, current_image.image, self._last_kps)

            if self.debug:
                for a, b in zip(self._last_kps, cur_kps):
                    cv2.line(current_image.image, (a[0], a[1]), (b[0], b[1]), (0, 0, 255))

            E, mask = cv2.findEssentialMat(cur_kps, self._last_kps,
                                           focal=self.camera.focal_point[0], pp=self.camera.center,
                                           method=cv2.RANSAC, prob=0.999, threshold=self._reproj_thresh)
            last_kps = np.float32([kp for m, kp in zip(mask, self._last_kps) if m])
            cur_kps = np.float32([kp for m, kp in zip(mask, cur_kps) if m])
            _, R, t, mask = cv2.recoverPose(E, cur_kps, last_kps,
                                            focal=self.camera.focal_point[0], pp=self.camera.center)
            if self._pose:
                self._pose = self._pose._replace(translation=self._pose.translation + absolute_scale * self._pose.rotation.dot(t),
                                                rotation=R.dot(self._pose.rotation))
            else:
                self._pose = Pose(R, t)
            self._last_pose = Pose(R, t)

            if len(self._last_kps) < self._min_features:
                current_image = self.vision.process(current_image)
                cur_kps = np.array([x.pt for x in current_image.features.points], dtype=np.float32)

            self._last_kps = cur_kps

        return self._pose

    @property
    def feature_type(self):
        return self._feature_type

    @property
    def pose(self):
        return self._pose

    @pose.setter
    def pose(self, value):
        if not isinstance(value, Pose) and value is not None:
            raise TypeError("Pose must be of type VisualOdometryEngine.Pose")
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
        return "Monocular Visual Odometry inspired by https://github.com/uoip/monoVO-python (2D-2D)"

    @property
    def capabilities(self):
        return EngineCapabilities(
                (ProcessorBase, FeatureExtraction),
                (Frame, Pose),
                {}
            )

    def _track_features(self, image_ref, image_cur, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **self._lk_params)  # shape: [k,2] [k,1] [k,1]

        umat = isinstance(st, cv2.UMat)
        if umat:
            st = st.get()
            kp2 = kp2.get()

        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]

        return kp1, kp2

    def _match_features(self, featuresA, featuresB, ratio=0.7, distance_thresh=30, min_matches=5):
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB

        matches = super(VisualOdometry2DEngine, self)._match_features(descriptorsA, descriptorsB, self._feature_type, ratio, distance_thresh, min_matches)

        if matches is None or not matches:
            return None

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

        return ptsA, ptsB