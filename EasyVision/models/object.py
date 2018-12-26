# -*- coding: utf-8 -*-
from .base import *
from collections import namedtuple
from EasyVision.vision import Image, Frame
from EasyVision.processors import FeatureMatchingMixin
import cv2
import numpy as np


class ObjectModel(FeatureMatchingMixin, ModelBase):
    __slots__ = FeatureMatchingMixin.SLOTS
    MatchResult = namedtuple('MatchResult', ['matches', 'homography', 'outline', 'view'])
    ComputeResult = namedtuple('ComputeResult', ['model', 'score', 'homography', 'outline', 'matches'])

    def __init__(self, name, views, *args, **kwargs):
        super(ObjectModel, self).__init__(name, views, *args, **kwargs)
        self.setup()

    def compute(self, frame, **kwargs):
        if not isinstance(frame, Frame):
            raise TypeError("frame must be Frame type")

        for image in frame.images:
            if not hasattr(image, "features"):
                raise ValueError("Image must implement features")
            if not hasattr(image, "feature_type"):
                raise ValueError("Image must implement feature_type")

        views = [self._match_view(frame, view, **kwargs) for view in self]

        if all(view is None for view in views):
            return None

        best = max(views, key=lambda x: sum(len(y.matches) for y in x))
        homography = tuple(x.homography for x in best)
        outline = tuple(x.outline for x in best)
        score = min(sum(m.distance for m in x.matches) for x in best) ** .5

        return self.ComputeResult(model=self, score=score, homography=homography, outline=outline, matches=views)

    def setup(self):
        super(ObjectModel, self).setup()

    def release(self):
        super(ObjectModel, self).release()

    @property
    def description(self):
        return "Simple feature(SIFT/ORB/KAZE/AKAZE) based object model"

    @staticmethod
    def from_processed_image(name, image, **kwargs):
        if not isinstance(image, Image):
            raise TypeError("Image must be Image type")
        if not hasattr(image, "features"):
            raise ValueError("Image must implement features")
        if not hasattr(image, "feature_type"):
            raise ValueError("Image must implement feature_type")

        if hasattr(image, "mask"):
            #calculate outline
            pass
        else:
            h, w, _ = image.image.shape
            pts = (
                (0, 0), (w - 1, 0),
                (w - 1, h - 1), (0, h - 1)
            )
            outline = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))

        view = ModelView(image.image, outline, image.features, image.feature_type)
        return ObjectModel(name, [view], **kwargs)

    def _match_view(self, frame, view, **kwargs):
        view_matches = [self._match_features(image.features, view, **kwargs) for image in frame.images if image.feature_type == view.feature_type]

        if self.display_results:
            self._draw(frame, view_matches)

        if all(match is None for match in view_matches):
            return None
        else:
            return view_matches

    def _calculate_outline(self, mask):
        pass

    def _match_features(self, featuresA, view, ratio=0.7, reprojThresh=5.0, distance_thresh=50, min_matches=5):
        matches = super(ObjectModel, self)._match_features(featuresA.descriptors, view.features.descriptors, view.feature_type, ratio, distance_thresh, min_matches)

        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = view.features
        outline = view.outline

        if matches is None or not matches:
            return None

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

        M = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reprojThresh)

        H, inliers = M

        if H is None:
            return None

        if sum(inliers) < min_matches:
            return None

        outline = cv2.perspectiveTransform(view.outline, H)

        matches = [m for m, pt in zip(matches, ptsA) if cv2.pointPolygonTest(outline, (pt[0], pt[1]), False) >= 0]

        if len(matches) < min_matches:
            return None

        return self.MatchResult(matches, H, outline, view)

    def _draw(self, frame, view_matches):
        for source, matches in zip(frame.images, view_matches):
            if matches is None:
                continue

            name = "{} = {}[{}]".format(source.source.name, self._name, self._views.index(matches.view))

            params = dict(
                               singlePointColor=None,
                               matchColor=(0, 255, 0),
                               flags=2)
            res = cv2.drawMatches(source.image, source.features.points,
                                  matches.view.image, matches.view.features.points,
                                  matches.matches, None, **params)
            res = cv2.polylines(res, [np.int32(matches.outline)], True, [255, 0, 0], 3, 8)
            cv2.imshow(name, res)