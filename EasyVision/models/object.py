# -*- coding: utf-8 -*-
"""Implements feature based object model

"""

from .base import *
from collections import namedtuple
from EasyVision.vision import Image, Frame
import cv2
import numpy as np


MatchResult = namedtuple('MatchResult', 'model view image matches homography outline')


class ObjectModel(ModelBase):
    """Feature based object model. Allows models to contain several views that represent the same model.

    """

    __slots__ = ('_min_matches', '_reproj_thresh')

    def __init__(self, name, views, min_matches=10, reproj_thresh=5.0, *args, **kwargs):
        self._min_matches = min_matches
        self._reproj_thresh = reproj_thresh
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

    def todict(self):
        """Convert object model to dict"""
        d = {
            'name': self.name,
            'views': [v.todict() for v in self._views]
        }
        return d

    @staticmethod
    def fromdict(d):
        """Create object model from dict"""
        return ObjectModel(d['name'], [ModelView.fromdict(v) for v in d['views']])

    @property
    def description(self):
        return "Simple feature(SIFT/ORB/KAZE/AKAZE) based object model"

    @staticmethod
    def _calculate_outline(mask):
        """Helper method to calculate outline using a mask.

        NOTE: Will select the largest contour as the assumption is to use this method in enrolling an object and in
        that case the object will occupy most of the camera view.
        """
        _, mask = cv2.threshold(mask, 180, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Isolate largest contour
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
        if not contour_sizes:
            return None

        return max(contour_sizes, key=lambda x: x[0])[1]

    @staticmethod
    def _get_outline(image):
        """Returns an outline for the object either from image shape or will calculate one from supplied mask"""
        if image.mask is not None:
            mask = image.mask.copy()
            outline = ObjectModel._calculate_outline(mask)
            if outline is None:
                return None, None, None
            mask = cv2.merge(tuple(mask for i in range(image.image.shape[2])))
            thumb = cv2.bitwise_and(image.image, mask)
            outline = outline.reshape((-1, 1, 2))
            x, y, w, h = cv2.boundingRect(outline)

            if w < 100 or h < 100:
                return None, None, None

            thumb = thumb[y:y + h, x:x + w]
            outline = np.float32([(i[0][0] - x, i[0][1] - y) for i in outline])
            kp = [p._replace(pt=(p.pt[0] - x, p.pt[1] - y)) for p in image.features.points]
            features = image.features._replace(points=kp)
        else:
            h, w, _ = image.image.shape
            pts = (
                (0, 0), (w - 1, 0),
                (w - 1, h - 1), (0, h - 1)
            )
            outline = np.float32(pts)
            thumb = image.image
            features = image.features

        return outline, thumb, features

    @staticmethod
    def create_from_processed_image(name, image, display_results=False, **kwargs):
        """Creates object model from processed image.
        Basically will take the features, outline, crop the object and return created ObjectModel.

        :param name: name of the object
        :param image: processed image
        :param display_results: indicates whether to display intermediate results
        :param kwargs: kwargs to pass to the newly created ObjectModel
        :return: ObjectModel instance with a single view
        """
        if not isinstance(image, Image):
            raise TypeError("Image must be Image type")
        if image.features is None or len(image.features.points) < 5:
            return None
        if not image.feature_type:
            return None

        if display_results and image.mask is not None:
            cv2.imshow("%s mask" % name, image.mask)

        outline, thumb, features = ObjectModel._get_outline(image)
        if outline is None:
            return None

        if display_results:
            img = cv2.drawKeypoints(thumb, features.keypoints, np.array([]), color=(0, 0, 255),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img = cv2.polylines(img, [np.int32(outline)], True, [255, 0, 0], 3, 8)
            cv2.imshow(name, img)

        view = ModelView(thumb, outline, features, image.feature_type)
        return ObjectModel(name, [view], display_results=display_results, **kwargs)

    def update_from_processed_frame(self, frame, matcher, **kwargs):
        """Will update current instance of ObjectModel with new frame.
        Will match existing views, determine if adding a new view will benefit and then add.

        :param frame: processed frame
        :param matcher: matches which subclases FeatureMatchingMixin
        :param kwargs: kwargs to pass to matcher
        :return: None if not enough features found or no outline could be calculated. Otherwise return self.
        """
        if not isinstance(frame, Frame):
            raise TypeError("Image must be Frame type")
        image = frame.images[0]
        if not image.features or len(image.features.points) < 5:
            return None
        if not image.feature_type:
            return None

        if len(image.features.points) < self._min_matches:
            return None

        image = frame.images[0]
        outline, thumb, features = ObjectModel._get_outline(image)

        if outline is None:
            return None

        if kwargs.get('display_results', False):
            img = cv2.drawKeypoints(thumb, features.keypoints, np.array([]), color=(0, 0, 255),
                                     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            img = cv2.polylines(img, [np.int32(outline)], True, [255, 0, 0], 3, 8)
            cv2.imshow(self.name, img)
            cv2.imshow("%s mask" % self.name, image.mask)

        views = (self._match_view(frame, view, matcher, **kwargs) for view in self)
        views = sum((v for v in views if v), ())

        if not views:
            self._views += [ModelView(thumb, outline, features, image.feature_type)]
            return self

        N = max(views, key=lambda x: len(x.matches))
        if N <= len(image.features.points) / 3:
            self.views += [ModelView(thumb, outline, features, image.feature_type)]
            return self

    def _match_view(self, frame, view, matcher, **kwargs):
        """Helper method to match a view against a processed frame"""
        view_matches = (self._match_features(image, view, matcher, **kwargs) for image in (i for i in frame.images if i.feature_type == view.feature_type))
        view_matches = sum((v for v in view_matches if v), ())

        if self.display_results:
            self._draw(view_matches)

        return view_matches if view_matches else None

    def _match_features(self, image, view, matcher, display_results=False, **kwargs):
        """Helper method to match features from a view to processed image.
        Will verify matching quality by computing homography, transforming outline and checking if features are
        contained by the outline.
        """
        kpsA, descriptorsA, _ = image.features
        kpsB, descriptorsB, _ = view.features
        outline = view.outline
        _outline = outline.reshape((-1, 1, 2))

        matches = matcher._match_features(descriptorsA, descriptorsB, view.feature_type, min_matches=self._min_matches, **kwargs)

        if matches is None or len(matches) < self._min_matches:
            return None

        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])

        results = ()

        while len(ptsA) >= self._min_matches:
            H, inliers = cv2.findHomography(ptsB, ptsA, cv2.RANSAC, self._reproj_thresh)

            if H is None:
                return results

            if sum(inliers) < self._min_matches or sum(inliers) < len(descriptorsB) * .01:
                return results

            __outline = cv2.perspectiveTransform(_outline, H)
            inside = [inlier and cv2.pointPolygonTest(__outline, (pt[0], pt[1]), False) >= 0 for inlier, pt in zip(inliers, ptsA)]

            if sum(inside) < self._min_matches:
                return results

            _matches = [m for i, m in zip(inside, matches) if i]
            matches = [m for i, m in zip(inside, matches) if not i]
            ptsA = np.float32([pt for i, pt in zip(inside, ptsA) if not i])
            ptsB = np.float32([pt for i, pt in zip(inside, ptsB) if not i])
            results += (MatchResult(self, view, image, _matches, H, __outline),)

        return results

    def _draw(self, view_matches):
        """Helper method to draw matches"""
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