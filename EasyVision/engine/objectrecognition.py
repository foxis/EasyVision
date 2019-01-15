# -*- coding: utf-8 -*-
from .base import *
from EasyVision.models import ObjectModel, ModelView
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction
from EasyVision.processors import FeatureMatchingMixin
import cv2
import numpy as np
from collections import namedtuple


class MatchResults(namedtuple('MatchResults', 'results')):
    """Container for object matching results"""

    pass


class ObjectRecognitionEngine(FeatureMatchingMixin, EngineBase):

    def __init__(self, vision, feature_type=None, max_matches=10, *args, **kwargs):
        feature_extractor_provided = False
        if not isinstance(vision, ProcessorBase) and not isinstance(vision, VisionBase):
            raise TypeError("Vision must be either VisionBase or ProcessorBase")
        if isinstance(vision, ProcessorBase):
            if vision.get_source('FeatureExtraction') is not None:
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
        if not frame:
            return None
        return frame, self._match_models(frame)

    def enroll(self, name, image, model=None, add=False, **kwargs):
        if model is None:
            model = self._models.get(name, None)

        if model is not None and not isinstance(image, Frame):
            raise TypeError("Updating model requires a frame")
        elif model is None and isinstance(image, Frame):
            image = image.images[0]
            if not image.features or not image.feature_type:
                image = self.vision.process(image)

        if model is not None:
            return model.update_from_processed_frame(image, self, **kwargs)

        model = ObjectModel.create_from_processed_image(name, image, **kwargs)
        if model is None:
            return None
        if add:
            if name in self._models:
                self._models[name].update(model)
            else:
                self._models[name] = model
        return model

    @property
    def description(self):
        return "Object Recognition Engine using various feature extractors"

    @property
    def models(self):
        return self._models

    @property
    def capabilities(self):
        return EngineCapabilities(
                (ProcessorBase, FeatureExtraction, ObjectModel),
                (Frame, MatchResults),
                {'feature_type': ('FREAK', 'SURF', 'SIFT', 'ORB', 'KAZE', 'AKAZE')}
            )

    def _match_models(self, frame):
        results = (model.compute(frame, self) for model in self._models.values())
        return MatchResults(sum(tuple(i for i in results if i), ()))
