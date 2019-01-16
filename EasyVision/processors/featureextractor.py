# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *
from collections import namedtuple


class FeatureExtraction(ProcessorBase):

    def __init__(self, vision, feature_type, extract=True, *args, **kwargs):
        if feature_type in ['FAST', 'GFTT'] and extract:
                raise ValueError("Cannot extract features with %s detector" % feature_type)
        self._kwargs = dict(**kwargs)
        self._kwargs.pop('enabled', None)
        self._kwargs.pop('debug', None)
        self._kwargs.pop('display_results', None)
        self._feature_type = feature_type
        self._extract = extract
        self._detector = self._descriptor = None
        super(FeatureExtraction, self).__init__(vision, *args, **kwargs)

    def setup(self):
        super(FeatureExtraction, self).setup()
        if self._feature_type == 'ORB':
            defaults = dict(nfeatures=10000)
            defaults.update(self._kwargs)
            self._descriptor = cv2.ORB_create(**defaults)
        elif self._feature_type == 'BRISK':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.BRISK_create(**defaults)
        elif self._feature_type == 'SURF':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.xfeatures2d.SURF_create(**defaults)
        elif self._feature_type == 'SIFT':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.xfeatures2d.SIFT_create(**defaults)
        elif self._feature_type == 'KAZE':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.KAZE_create(**defaults)
        elif self._feature_type == 'AKAZE':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.AKAZE_create(**defaults)
        elif self._feature_type == 'FREAK':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.xfeatures2d.FREAK_create(**defaults)
            self._detector = cv2.xfeatures2d.SURF_create()
        elif self._feature_type == 'FAST':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.FastFeatureDetector_create(**defaults)
        elif self._feature_type == 'GFTT':
            defaults = dict()
            defaults.update(self._kwargs)
            self._descriptor = cv2.GFTTDetector_create(**defaults)
        else:
            raise ValueError("Invalid feature type")

    def release(self):
        super(FeatureExtraction, self).release()

    @property
    def description(self):
        return "Simple Feature Detection/Extraction processor"

    @property
    def feature_type(self):
        return self._feature_type

    def process(self, image):
        if not self._detector:
            keypoints, descriptors = self._descriptor.detect(image.image, image.mask), None
        else:
            keypoints, descriptors = self._detector.detect(image.image, image.mask), None

        if self._extract:
            keypoints, descriptors = self._descriptor.compute(image.image, keypoints)

        if self.display_results:
            self._draw_keypoints(image.image, keypoints)

        #if isinstance(descriptors, cv2.UMat):
        #    # this will enhance matching when using OCL detector/extractor
        #    descriptors = descriptors.get()

        return image._replace(features=Features(keypoints, descriptors), feature_type=self._feature_type)

    def _draw_keypoints(self, image, keypoints):
        img = cv2.drawKeypoints(image, keypoints, np.array([]), color=(0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(self.name, img)


class FeatureMatchingMixin(object):
    SLOTS = ('_matcher_h', '_matcher_l')
    __slots__ = ()

    def __init__(self, *args, **kwargs):

        super(FeatureMatchingMixin, self).__init__(*args, **kwargs)

    def setup(self):
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self._matcher_h = cv2.FlannBasedMatcher(index_params, search_params)

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)   # or pass empty dictionary
        self._matcher_l = cv2.FlannBasedMatcher(index_params, search_params)
        super(FeatureMatchingMixin, self).setup()

    def _match_features(self, descriptorsA, descriptorsB, feature_type, ratio=0.7, distance_thresh=30, min_matches=10):
        if feature_type in ['ORB', 'AKAZE', 'FREAK', 'BRISK']:
            matches = self._matcher_h.knnMatch(descriptorsA, descriptorsB, 2)
        else:
            matches = self._matcher_l.knnMatch(descriptorsA, descriptorsB, 2)

        if matches is None:
            return None

        matches = [M[0] for M in matches if len(M) == 2 and M[0].distance < M[1].distance * ratio and M[0].distance < distance_thresh]
        if len(matches) < min_matches:
            return None

        return matches
