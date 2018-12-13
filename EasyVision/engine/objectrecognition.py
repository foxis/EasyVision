# -*- coding: utf-8 -*-
from .base import EngineBase
from EasyVision.models import ObjectModel, ModelView
import cv2
import numpy as np


class ObjectRecognitionEngine(EngineBase):

    def __init__(self, vision, feature_type, max_matches=10, *args, **kwargs):
        super(ObjectRecognitionEngine, self).__init__(vision, *args, **kwargs)

        self._models = []
        self._max_matches = max_matches

        if feature_type == 'ORB':
            self._descriptor = cv2.ORB_create(nfeatures=10000, scoreType=cv2.ORB_FAST_SCORE)
        elif feature_type == 'KAZE':
            self._descriptor = cv2.KAZE_create()
        elif feature_type == 'AKAZE':
            self._descriptor = cv2.AKAZE_create()

    def compute(self):
        frame = self.vision.capture()
        if frame:
            views = self._extract_features(frame)
            self._match_models(frame, views)

    def enroll(self, image, mask=None):
        pass

    @property
    def description(self):
        return "Object Recognition Engine using SIFT/AKAZE/ORB"

    @property
    def models(self):
        return self._models

    @property
    def capabilities(self):
        return {
            "models": (ObjectModel, ),
            "feature_types": ('ORB', 'KAZE', 'AKAZE')
        }

    def _extract_features(self, frame):
        keypoints = [ModelView(image, self._descriptor.detectAndCompute(image.image, np.array([])), self._feature_type) for image in frame.images]
        return tuple(keypoints)

    def _match_models(self, frame, views):
        scores = [model.compute(frame, views) for model in self._models]
        return sorted(scores, key=lambda x: x.score, reverse=True)[0:self._max_matches]
