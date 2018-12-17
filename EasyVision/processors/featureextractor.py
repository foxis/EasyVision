# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *


class FeatureExtraction(ProcessorBase):
    Features = namedtuple('Features', ['points', 'descriptors'])

    def __init__(self, vision, feature_type, extract=True, debug=False, display_results=False, *args, **kwargs):
        if feature_type == 'ORB':
            defaults = dict(nfeatures=10000)
            defaults.update(kwargs)
            self._descriptor = cv2.ORB_create(**defaults)
        elif feature_type == 'KAZE':
            defaults = dict()
            defaults.update(kwargs)
            self._descriptor = cv2.KAZE_create(**kwargs)
        elif feature_type == 'AKAZE':
            defaults = dict()
            defaults.update(kwargs)
            self._descriptor = cv2.AKAZE_create(**kwargs)
        elif feature_type == 'FAST':
            defaults = dict()
            defaults.update(kwargs)
            if extract:
                raise ValueError("Cannot extract features with FAST detector")
            self._descriptor = cv2.FastFeatureDetector_create(**kwargs)
        elif feature_type == 'GFTT':
            defaults = dict()
            defaults.update(kwargs)
            if extract:
                raise ValueError("Cannot extract features with GFTT detector")
            self._descriptor = cv2.GFTTDetector_create(**kwargs)
        else:
            raise ValueError("Invalid feature type")
        self._feature_type = feature_type
        self._extract = extract
        super(FeatureExtraction, self).__init__(vision, debug=debug, display_results=display_results, *args, **kwargs)

    @property
    def description(self):
        return "Simple Feature Detection/Extraction processor"

    def process(self, image):
        mask = None
        if isinstance(image, ImageWithMask):
            mask = self._make_mask(image.mask)

        keypoints, descriptors = self._descriptor.detect(image.image, mask), None
        if self._extract:
            keypoints, descriptors = self._descriptor.compute(image.image, keypoints)

        if self.display_results:
            self._draw_keypoints(image.image, keypoints)

        if mask:
            return ImageWithMaskAndFeatures(self, image.image, image.mask, self.Features(keypoints, descriptors), self._feature_type)
        else:
            return ImageWithFeatures(self, image.image, self.Features(keypoints, descriptors), self._feature_type)

    def _make_mask(self, mask):
        raise NotImplementedError()

    def _draw_keypoints(self, image, keypoints):
        img = cv2.drawKeypoints(image, keypoints, np.array([]), color=(0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(self.name, img)
