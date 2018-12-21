# -*- coding: utf-8 -*-
from .base import EngineBase
from EasyVision.models import ObjectModel, ModelView
from EasyVision.processors.base import *
from EasyVision.processors import FeatureExtraction
import cv2
import numpy as np


class BOWVocabularyBuilderEngine(EngineBase):

    def __init__(self, vision, feature_type, clusters, *args, **kwargs):
        if not isinstance(vision, ProcessorBase) and not isinstance(vision, VisionBase):
            raise TypeError("Vision must be either VisionBase or ProcessorBase")
        if not isinstance(vision, ProcessorBase) and not feature_type:
            raise TypeError("Feature type must be provided")

        self._feature_type = feature_type
        self._vocabulary_valid = False
        _vision = FeatureExtraction(vision, feature_type=feature_type) if not isinstance(vision, FeatureExtraction) else vision

        self._trainer = cv2.BOWKMeansTrainer(clusters)

        super(BOWVocabularyBuilderEngine, self).__init__(_vision, *args, **kwargs)

    def compute(self):
        frame = self.vision.capture()
        self._vocabulary_valid = False
        for image in frame.images:
            self._trainer.add(image.features.descriptors)
        return frame

    @property
    def vocabulary(self):
        if self._vocabulary_valid:
            return self._vocabulary
        else:
            self._vocabulary = self._trainer.cluster()
            return self._vocabulary

    def release(self):
        super(BOWVocabularyBuilderEngine, self).release()

    @property
    def description(self):
        return "Bag Of Visual Words engine"

    @property
    def capabilities(self):
        return {}

    def k_means(data, k=2, max_iter=100):
        """Assigns data points into clusters using the k-means algorithm.

           NOTE: Code taken from https://codereview.stackexchange.com/questions/205097/k-means-using-numpy
                 And modified to support Hamming Distance

        Parameters
        ----------
        data : ndarray
            A 2D array containing data points to be clustered.
        k : int, optional
            Number of clusters (default = 2).
        max_iter : int, optional
            Number of maximum iterations

        Returns
        -------
        labels : ndarray
            A 1D array of labels for their respective input data points.
        """

        # data_max/data_min : array containing column-wise maximum/minimum values
        data_max = np.max(data, axis=0)
        data_min = np.min(data, axis=0)

        n_samples = data.shape[0]
        n_features = data.shape[1]

        # labels : array containing labels for data points, randomly initialized
        labels = np.random.randint(low=0, high=k, size=n_samples)
        # centroids : 2D containing centroids for the k-means algorithm
        # randomly initialized s.t. data_min <= centroid < data_max
        centroids = np.random.uniform(low=0., high=1., size=(k, n_features))
        centroids = centroids * (data_max - data_min) + data_min

        # k-means algorithm
        for i in range(max_iter):
            # distances : between datapoints and centroids
            distances = np.array(
                [np.linalg.norm(data - c, axis=1) for c in centroids])
            # new_labels : computed by finding centroid with minimal distance
            new_labels = np.argmin(distances, axis=0)

            if (labels == new_labels).all():
                # labels unchanged
                labels = new_labels
                print('Labels unchanged ! Terminating k-means.')
                break
            else:
                # labels changed
                # difference : percentage of changed labels
                difference = np.mean(labels != new_labels)
                print('%4f%% labels changed' % (difference * 100))
                labels = new_labels
                for c in range(k):
                    # computing centroids by taking the mean over associated data points
                    centroids[c] = np.mean(data[labels == c], axis=0)

        return labels


class BOWMatchingMixin(object):
    SLOTS = ('_bow_extractor',)
    __slots__ = ()

    def __init__(self, *args, **kwargs):
        self._bow_extractor = None
        super(BOWMatchingMixin, self).__init__(*args, **kwargs)

    def initBOW(self, extractor, matcher, vocabulary):
        self._bow_extractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
        self._bow_extractor.setVocabulary(vocabulary)

    def _compute_bow(self, descriptors):
        return self._bow_extractor.compute(descriptors)

    def _match_bow(self, bowA, bowB, method=cv2.HISTCMP_INTERSECT):
        return cv2.compareHist(bowA, bowB, method)
