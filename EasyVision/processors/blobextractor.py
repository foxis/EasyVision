# -*- coding: utf-8 -*-
import cv2
import numpy as np
from .base import *
from .histogrambackprojection import HistogramBackprojection
from collections import namedtuple


KeyPoint = namedtuple('KeyPoint', ['pt', 'size', 'angle', 'response', 'octave', 'class_id'])


class Blobs(namedtuple('Features', ['points', 'descriptors'])):
    __slots__ = ()

    @property
    def keypoints(self):
        return [cv2.KeyPoint(x=pt.pt[0], y=pt.pt[1], _size=pt.size, _angle=pt.angle,
                             _response=pt.response, _octave=pt.octave, _class_id=pt.class_id) for pt in self.points]

    @classmethod
    def _make(cls, keypoints, descriptors):
        points = [KeyPoint(pt.pt, pt.size, pt.angle, pt.response, pt.octave, pt.class_id) for pt in keypoints]
        return super(Blobs, cls)._make((points, descriptors))


class BlobExtraction(ProcessorBase):

    def __init__(self, vision, histogram, blur_size=(30, 30), channels=(0, 1), ranges=(0, 180, 0, 256),
                 area=(100, 400 * 400), min_circularity=None, min_convexity=None, min_inertia=None, *args, **kwargs):
        vision = HistogramBackprojection(vision, histogram, channels=channels, ranges=ranges) if not isinstance(vision, HistogramBackprojection) else vision

        if not isinstance(blur_size, tuple) or len(blur_size) != 2:
            raise TypeError("Blur Kernel Size must be a tuple of two integers")
        if not isinstance(area, tuple) or len(area) != 2:
            raise TypeError("Area constraints must be a tuple of two integers")

        self._blur_size = blur_size
        self._params = cv2.SimpleBlobDetector_Params()
        #self._params.minThreshold = 10
        #self._params.maxThreshold = 200

        if area is not None:
            self._params.filterByArea = True
            self._params.minArea = area[0]
            self._params.maxArea = area[1]
        else:
            self._params.filterByArea = False
        if min_circularity is not None:
            self._params.filterByCircularity = True
            self._params.minCircularity = min_circularity
        else:
            self._params.filterByCircularity = False
        if min_convexity is not None:
            self._params.filterByConvexity = True
            self._params.minConvexity = min_convexity
        else:
            self._params.filterByConvexity = False
        if min_inertia is not None:
            self._params.filterByInertia = True
            self._params.minInertiaRatio = min_inertia
        else:
            self._params.filterByInertia = False

        super(BlobExtraction, self).__init__(vision, *args, **kwargs)

    def setup(self):
        self._detector = cv2.SimpleBlobDetector_create(self._params)
        super(BlobExtraction, self).setup()

    def release(self):
        super(BlobExtraction, self).release()
        del self._detector

    @property
    def description(self):
        return "Simple Feature Detection/Extraction processor"

    @property
    def feature_type(self):
        return 'blobs'

    def process(self, image):
        keypoints = []

        for i, mask in enumerate(image.mask):
            mask = 255 - mask
            mask = cv2.blur(mask, self._blur_size)
            _, mask = cv2.threshold(mask, 50, 255, 0)
            kps = self._detector.detect(mask)
            keypoints += kps
            if self.display_results:
                cv2.imshow("Mask%i" % i, mask)

        if self.display_results:
            self._draw_keypoints(image.image, keypoints)

        return image._replace(features=Blobs._make(keypoints, None), feature_type='blobs')

    def _draw_keypoints(self, image, keypoints):
        img = cv2.drawKeypoints(image, keypoints, np.array([]), color=(0, 0, 255),
                                 flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.imshow(self.name, img)


class BlobMatchingMixin(object):
    SLOTS = ()
    __slots__ = ()

    def __init__(self, *args, **kwargs):

        super(BlobMatchingMixin, self).__init__(*args, **kwargs)

    def setup(self):
        super(BlobMatchingMixin, self).setup()

    def _match_features(self, descriptorsA, descriptorsB, feature_type, ratio=0.7, distance_thresh=30, min_matches=10):
        return None
