# -*- coding: utf-8 -*-
from .base import *
from collections import namedtuple
from EasyVision.vision import Image, Frame
import cv2
import numpy as np


MatchResult = namedtuple('MatchResult', 'model view image matches homography outline')


class ObjectModel(ModelBase):
    __slots__ = ()

    def __init__(self, name, views, *args, **kwargs):
        super(ObjectModel, self).__init__(name, views, *args, **kwargs)
        self.setup()

    def compute(self, frame, matcher, **kwargs):
        if not isinstance(frame, Frame):
            raise TypeError("frame must be Frame type")

        for image in frame.images:
            if not image.features:
                raise ValueError("Image must implement features")
            if not image.feature_type:
                raise ValueError("Image must implement feature_type")

        views = (self._match_view(frame, view, matcher, **kwargs) for view in self)

        return sum((v for v in views if v), ())

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
        if not image.features:
            raise ValueError("Image must have features")
        if not image.feature_type:
            raise ValueError("Image must have feature_type")

        if image.mask:
            #calculate outline
            raise NotImplementedError()
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

    def update(image, **kwargs):
        pass

    def _match_view(self, frame, view, matcher, **kwargs):
        view_matches = (self._match_features(image, view, matcher, **kwargs) for image in (i for i in frame.images if i.feature_type == view.feature_type))
        view_matches = sum((v for v in view_matches if v), ())

        if self.display_results:
            self._draw(view_matches)

        return view_matches if view_matches else None

    def _calculate_outline(self, mask):
        pass

    def _match_features(self, image, view, matcher, min_matches=5, reproj_thresh=5, **kwargs):
        kpsA, descriptorsA = image.features
        kpsB, descriptorsB = view.features
        outline = view.outline

        matches = matcher._match_features(descriptorsA, view.features.descriptors, view.feature_type, min_matches=min_matches, **kwargs)

        if matches is None or len(matches) < min_matches:
            return None

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

        results = ()

        while len(ptsA) >= min_matches:
            H, inliers = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, reproj_thresh)

            if H is None:
                return results

            if sum(inliers) < min_matches:
                return results

            outline = cv2.perspectiveTransform(view.outline, H)

            inside = [inlier and cv2.pointPolygonTest(outline, (pt[0], pt[1]), False) >= 0 for inlier, pt in zip(inliers, ptsA)]

            if sum(inside) < min_matches:
                return results
            _matches = [m for i, m in zip(inside, matches) if i]
            matches = [m for i, m in zip(inside, matches) if not i]
            ptsA = np.float32([pt for i, pt in zip(inside, ptsA) if not i])
            ptsB = np.float32([pt for i, pt in zip(inside, ptsB) if not i])
            results += (MatchResult(self, view, image, _matches, H, outline),)

        return results

    def _draw(self, view_matches):
        for index, match in enumerate(view_matches):
            name = "{} = {}[{}]{}".format(match.image.source.name, self._name, self._views.index(match.view), index)
            params = dict(
                               singlePointColor=None,
                               matchColor=(0, 255, 0),
                               flags=2)
            res = cv2.drawMatches(match.image.image, match.image.features.keypoints,
                                  match.view.image, match.view.features.keypoints,
                                  match.matches, None, **params)
            res = cv2.polylines(res, [np.int32(match.outline)], True, [255, 0, 0], 3, 8)
            cv2.imshow(name, res)