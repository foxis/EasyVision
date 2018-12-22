# -*- coding: utf-8 -*-
from .base import EngineBase
from EasyVision.models import ObjectModel, ModelView
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction
import cv2
import numpy as np


class ObjectRecognitionEngine(EngineBase):

    def __init__(self, vision, feature_type=None, max_matches=10, *args, **kwargs):
        feature_extractor_provided = False
        if not isinstance(vision, ProcessorBase) and not isinstance(vision, VisionBase):
            raise TypeError("Vision must be either VisionBase or ProcessorBase")
        if isinstance(vision, ProcessorBase):
            if vision.get_source('FeatureExtraction'):
                feature_type = vision.feature_type
                feature_extractor_provided = True
            elif not feature_type:
                raise TypeError("Feature type must be provided")

        self._models = {}
        self._feature_type = feature_type
        self._max_matches = max_matches

        _vision = FeatureExtraction(vision, feature_type=feature_type) if not feature_extractor_provided else vision

        super(ObjectRecognitionEngine, self).__init__(_vision, *args, **kwargs)

    def compute(self):
        frame = self.vision.capture()
        return frame, self._match_models(frame)

    def enroll(self, name, image, add=False, **kwargs):
        processed = self.vision.process(image)
        model = ObjectModel.from_processed_image(name, processed, **kwargs)
        if model is None:
            return None
        if add:
            if name in self._models:
                self._models[name].update(model)
            else:
                self._models[name] = model
        return model

    def release(self):
        super(ObjectRecognitionEngine, self).release()

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

    def _match_models(self, frame):
        results = (model.compute(frame) for model in self._models.values())
        results = [i for i in results if i]
        print results
        return sorted(results, key=lambda x: x.score, reverse=False)[0:self._max_matches]
