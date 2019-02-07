# -*- coding: utf-8 -*-
"""Implements Bag Of Words algorithm and matching.

If pyDBoW3 is available, will use that instead of openCV kmeans bow algorithm
"""

from EasyVision.engine.base import EngineBase, EngineCapability
from EasyVision.models import ObjectModel, ModelView
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction
import cv2
import numpy as np

dbow_available = False
try:
    import pyDBoW3 as bow
    dbow_available = True
except:
    try:
        import pyDBoW3_32 as bow
        dbow_available = True
    except:
        pass


class BOWVocabularyBuilderEngine(EngineBase):
    """Class implementing BOW dictionary

    """

    def __init__(self, vision, feature_type, clusters, dbow3_trainer=dbow_available,
                 k=10, L=5, weighting=bow.WeightingType.TF_IDF, scoring=bow.ScoringType.L1_NORM,
                 *args, **kwargs):
        """Instance initialization.

        :param vision: capturing source object.
        :param feature_type: type of features to use
        :param clusters: number of clusters
        :param dbow3_trainer: True tells to use DBoW3 library
        :param k: DBoW3 parameter
        :param L: DBoW3 parameter
        :param weighting: DBoW3 parameter
        :param scoring: DBoW3 parameter
        """

        if not isinstance(vision, ProcessorBase) and not isinstance(vision, VisionBase):
            raise TypeError("Vision must be either VisionBase or ProcessorBase")
        if not isinstance(vision, ProcessorBase) and not feature_type:
            raise TypeError("Feature type must be provided")

        if not dbow3_trainer and feature_type not in ['SIFT', 'SURF']:
            raise NotImplementedError("OpenCV KMeans trainer only supports floating point features. Use DBoW3 instead.")

        if dbow3_trainer and not dbow_available:
            raise NotImplementedError("pyDBoW3 library not imported")

        self._feature_type = feature_type
        self._vocabulary_valid = False
        _vision = FeatureExtraction(vision, feature_type=feature_type) if not isinstance(vision, FeatureExtraction) else vision

        if dbow3_trainer:
            self._trainer = bow.Vocabulary(k, L, weighting, scoring)
            self._features = []
        else:
            self._trainer = cv2.BOWKMeansTrainer(clusters)

        super(BOWVocabularyBuilderEngine, self).__init__(_vision, *args, **kwargs)

    def compute(self):
        frame = self.vision.capture()
        self._vocabulary_valid = False
        for image in frame.images:
            if hasattr(self, '_features'):
                self._features += [image.features.descriptors]
            else:
                self._trainer.add(image.features.descriptors)
        return frame

    def save(self, path):
        """Saves BOW dictionary to a file"""
        if hasattr(self, '_features'):
            self._trainer.save(path, True)
        else:
            raise NotImplementedError()

    def load(self, path):
        """Loads BOW dictionary from a file"""
        if hasattr(self, '_features'):
            self._trainer.load(path)
        else:
            raise NotImplementedError()

    def create_vocabulary(self):
        """Creates a vocabulary using collected features"""
        if hasattr(self, '_features'):
            self._trainer.create(self._features)
        else:
            self._vocabulary = self._trainer.cluster()

    @property
    def vocabulary(self):
        """Returns vocabulary implementation"""
        return self._trainer if hasattr(self, '_features') else self._vocabulary

    def release(self):
        super(BOWVocabularyBuilderEngine, self).release()
        del self._trainer

    @property
    def description(self):
        return "Bag Of Visual Words engine using DBoW3 library or OpenCV KMeans trainer"

    @property
    def capabilities(self):
        return EngineCapability(
                (ProcessorBase, FeatureExtraction),
                (Frame,),
                {'dictionaries': ('kmeans', 'dbow3')}
            )


class BOWMatchingMixin(object):
    """Mixin class implementing BOW matching.

    """

    SLOTS = ('_bow_extractor', '_database')
    __slots__ = SLOTS

    def __init__(self, *args, **kwargs):
        self._bow_extractor = None
        super(BOWMatchingMixin, self).__init__(*args, **kwargs)

    def initBOW(self, extractor, matcher, vocabulary, feature_type):
        """Must be called for the mixin initialization. Usually being called from ``setup`` method.

        :param extractor: is used for openCV BOW algorithm
        :param matcher: is used for openCV BOW algorithm
        :param vocabulary: BOW vocabulary instance or a string. if a string is provided, will use dbow3 and will treat it as path to the database.
        :param feature_type: type of features used in the matching/bow calculation process
        :return: None
        """

        self._dbow3 = isinstance(vocabulary, bow.Vocabulary) or isinstance(vocabulary, str)
        if self._dbow3 and feature_type not in ['SIFT', 'SURF']:
            raise NotImplementedError("OpenCV KMeans trainer only supports floating point features. Use DBoW3 instead.")

        if not self._dbow3:
            self._bow_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
            self._bow_extractor.setVocabulary(vocabulary)
            self._database = []
        else:
            self._database = bow.Database()
            if isinstance(vocabulary, str):
                self._database.loadVocabulary(vocabulary)
            else:
                self._database.setVocabulary(vocabulary)

    def release(self):
        """Releases BOW matching engine. Must be called as ``super().release()``"""
        super(BOWMatchingMixin, self).release()
        del self._database

    def _add_keyframe(self, descriptors):
        """Adds a keyframe to the matching database"""
        if isinstance(self._database, list):
            self._database += [(self._compute_bow(descriptors), descriptors)]
        else:
            self._database.add(descriptors)

    def _query_frame(self, descriptors, max_results=10):
        """Finds a list of matching keyframes using provided descriptors"""
        if isinstance(self._database, list):
            current_bow = self._compute_bow(descriptors)
            matches = (self._match_bow(current_bow, bow) for bow, descriptor in self._database)
            all_results = [(i, score) for i, score in enumerate(matches)].sorted(key=lambda x: x[1])
        else:
            all_results = [(result.Id, result.Score) for result in self._database.query(descriptors)].sorted(key=lambda x: x[1])
        return all_results[:max_results]

    def _compute_bow(self, descriptors):
        """Helper method to compute BOW using feature descriptors"""
        return self._bow_extractor.compute(descriptors)

    def _match_bow(self, bowA, bowB, method=cv2.HISTCMP_INTERSECT):
        """Compares two histograms"""
        return cv2.compareHist(bowA, bowB, method)
