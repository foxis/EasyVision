# -*- coding: utf-8 -*-
from .base import EngineBase
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction, CalibratedCamera
import cv2
import numpy as np


class VisualOdometryEngine(EngineBase):
    Pose = namedtuple('Pose', ['rotation', 'translation'])

    def __init__(self, vision, feature_type='GFTT', pose=None, min_features=5000, debug=False, display_results=False, *args, **kwargs):
        if not isinstance(vision, CalibratedCamera):
            raise TypeError("Vision must be CalibratedCamera")

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
            defaults['nfeatures'] = 10000
            #defaults['scoreType'] = cv2.ORB_FAST_SCORE
            defaults['nlevels'] = 16
            defaults.update(kwargs)

        self._feature_type = feature_type
        _vision = FeatureExtraction(vision, feature_type=feature_type, extract=self._extract, **defaults)

        self._min_features = min_features

        self._lk_params = dict(
            winSize=(21, 21),
            #maxLevel = 3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        if _vision.get_source("CalibratedCamera") is None:
            raise ValueError("vision processor stack must contain CalibratedCamera")

        self.camera = _vision.get_source("CalibratedCamera").camera
        self._last_image = None
        self._last_kps = None
        self._last_features = None
        self.pose = pose

        self._matcher_bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary

        self._matcher_flann = cv2.FlannBasedMatcher(index_params, search_params)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self._matcher_flann_hamming = cv2.FlannBasedMatcher(index_params, search_params)

        super(VisualOdometryEngine, self).__init__(_vision, debug=debug, display_results=display_results, *args, **kwargs)

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
                                           method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, mask = cv2.recoverPose(E, cur_kps, self._last_kps,
                                            focal=self.camera.focal_point[0], pp=self.camera.center)
            if self._pose:
                self._pose = self._pose._replace(translation=self._pose.translation + absolute_scale * self._pose.rotation.dot(t),
                                                rotation=R.dot(self._pose.rotation))
            else:
                self._pose = self.Pose(R, t)

            if len(self._last_kps) < self._min_features:
                current_image = self.vision.process(current_image)
                cur_kps = np.array([x.pt for x in current_image.features.points], dtype=np.float32)

            self._last_kps = cur_kps

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

    def _track_features(self, image_ref, image_cur, px_ref):
        kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **self._lk_params)  #shape: [k,2] [k,1] [k,1]

        st = st.reshape(st.shape[0])
        kp1 = px_ref[st == 1]
        kp2 = kp2[st == 1]

        return kp1, kp2

    def _match_features(self, featuresA, featuresB, ratio=0.7, reprojThresh=5.0, distance_thresh=30, min_matches=5):
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = featuresB

        if self._feature_type in ['ORB', 'AKAZE']:
            matches = self._matcher_flann_hamming.knnMatch(descriptorsA, descriptorsB, 2)
        else:
            matches = self._matcher_flann.knnMatch(descriptorsA, descriptorsB, 2)

        if matches is None:
            return None

        #matches.sort(key=lambda x: x[0].distance)
        matches = [m for m, n in matches if m.distance < n.distance * ratio and m.distance < distance_thresh]

        if not matches:
            return None

        print matches

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        #M = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

        #H, inliers = M

        #if H is None:
        #    return None

        #if sum(inliers) < min_matches:
        #    return None

        #inliers = inliers.ravel().tolist()


        #ptsA = np.float32([kpsA[m.queryIdx].pt for i, m in zip(inliers, matches) if i])
        #ptsB = np.float32([kpsB[m.trainIdx].pt for i, m in zip(inliers, matches) if i])

        return ptsA, ptsB