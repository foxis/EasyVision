# -*- coding: utf-8 -*-
"""Implements object recognition and object enrollment using specified features.
"""

from EasyVision.engine.base import *
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
    """Class implementing Object recognition algorithm using feature matching.
    This implementation matches processed features with all the models and all model views.

    Actual matching is delegated to the model.
    """

    def __init__(self, vision, feature_type=None, max_matches=10, *args, **kwargs):
        """Instance initialization

        :param vision: capturing source object.
        :param feature_type: specify feature type. May be left None if capturing source contains ``FeatureExtraction``
        :param max_matches: maximum number of features if using ORB features
        """
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
        """Enrolls a model into the matching engine.
        If no model is provided then this method will create a new model with the supplied name.
        Otherwise a supplied model will be updated with the image.

        Will extract features if image does not already contain them.

        NOTE: For multiprocessing and especially Pyro4 stacks passing unprocessed image may incurr
        performance penalties, as the image will be transferred back and forth to a different process or even machine.

        :param name: model name
        :param image: Image containing an overlapping view of the object
        :param model: model to enroll image into
        :param add: indicates whether to add the newly enrolled model to the engine
        :param kwargs: kwargs to pass to model's ``update_from_processed_frame`` or ``create_from_processed_image``
        :return: enrolled model
        """
        if model is None:
            model = self._models.get(name, None)

        if model is not None and not isinstance(image, Frame):
            raise TypeError("Updating model requires a frame")
        elif model is None and isinstance(image, Frame):
            image = image.images[0]
        elif not isinstance(image, Image):
            raise TypeError("image must be Image.")

        if image.features is None or not image.feature_type:
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
        """Returns a list of enrolled models"""
        return self._models

    @property
    def capabilities(self):
        return EngineCapability(
                (ProcessorBase, FeatureExtraction, ObjectModel),
                (Frame, MatchResults),
                {'feature_type': ('FREAK', 'SURF', 'SIFT', 'ORB', 'KAZE', 'AKAZE')}
            )

    def _match_models(self, frame):
        """Helper method to find all matching models with all the matching views.
        Will return MatchResults, where results will be a tuple of all the matching views."""
        results = (model.compute(frame, self) for model in self._models.values())
        return MatchResults(sum((i for i in results if i), ()))
