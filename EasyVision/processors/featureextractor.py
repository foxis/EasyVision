# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *


class FeatureExtraction(ProcessorBase):
    Features = namedtuple('Features', ['points', 'descriptors'])

    def __init__(self, vision, feature_type, debug=False, display_results=False, *args, **kwargs):
        if feature_type == 'ORB':
            defaults = dict(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
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
        else:
            raise ValueError("Invalid feature type")
        super(FeatureExtraction, self).__init__(vision, debug=debug, display_results=display_results, *args, **kwargs)

    @property
    def description(self):
        return "Simple Feature Extractor processor"

    def process(self, image):
        mask = None
        if isinstance(image, ImageWithMask):
            mask = self._make_mask(image.mask)

        keypoints = self._descriptor.detectAndCompute(image.image, mask)

        if self.display_results:
            self._draw_keypoints(image.image, keypoints[0])

        if mask:
            return ImageWithMaskAndFeatures(self, image.image, image.mask, self.Features(*keypoints))
        else:
            return ImageWithFeatures(self, image.image, self.Features(*keypoints))

    def _make_mask(self, mask):
        raise NotImplementedError()

    def _draw_keypoints(self, image, keypoints):
        img = cv2.drawKeypoints(image, keypoints, np.array([]), color=(0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(self.name, img)

    def display_results_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)