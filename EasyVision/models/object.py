# -*- coding: utf-8 -*-
from .base import *
from collections import namedtuple
from EasyVision.vision import Image, Frame
import cv2
import numpy as np


class ObjectModel(ModelBase):
    __slots__ = ('_matcher')
    MatchResult = namedtuple('MatchResult', ['matches', 'homography', 'status', 'outline', 'view'])
    ComputeResult = namedtuple('ComputeResult', ['score', 'homography', 'outline', 'matches'])

    def __init__(self, name, views, *args, **kwargs):
        self._matcher = cv2.DescriptorMatcher_create("BruteForce")
        super(ObjectModel, self).__init__(name, views, *args, **kwargs)

    def compute(self, frame, **kwargs):
        if not isinstance(frame, Frame):
            raise TypeError("frame must be Frame type")

        for image in frame.images:
            if not hasattr(image, "features"):
                raise ValueError("Image must implement features")

        views = [self._match_view(frame, view) for view in self]

        if all(view is None for view in views):
            return None

        best = max(views, key=lambda x: sum(len(y.matches) for y in x))
        homography = tuple(x.homography for x in best)
        outline = tuple(x.outline for x in best)
        score = sum(len(x.matches) for x in best) / sum(len(x.view.features.points) for x in best)

        return self.ComputeResult(score=score, homography=homography, outline=outline, matches=views)

    def release(self):
        pass

    @property
    def description(self):
        return "Simple feature(SIFT/ORB/KAZE/AKAZE) based object model"

    @staticmethod
    def from_processed_image(name, image, feature_type):
        if not isinstance(image, Image):
            raise TypeError("Image must be Image type")
        if not hasattr(image, "features"):
            raise ValueError("Image must implement features")

        if hasattr(image, "mask"):
            #calculate outline
            pass
        else:
            h, w, _ = image.image.shape
            outline = (
                (0, 0), (w, 0),
                (w, h), (0, h)
            )

        view = ModelView(image.image, outline, image.features, feature_type)
        return ObjectModel(name, [view])

    def _match_view(self, frame, view, **kwargs):
        return [self._match_keypoints(image.features, view, **kwargs) for image in frame.images]

    def _calculate_outline(self, view, match):
        pass

    def _match_keypoints(self, featuresA, view, ratio=0.75, reprojThresh=4.0):
        kpsA, descriptorsA = featuresA
        kpsB, descriptorsB = view.features
        outline = view.outline

        rawMatches = self._matcher.knnMatch(descriptorsA, descriptorsB, 2)
        matches = [(m[0].trainIdx, m[0].queryIdx) for m in rawMatches if len(m) == 2 and m[0].distance < m[1].distance * ratio]

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i].pt for (_, i) in matches])
            ptsB = np.float32([kpsB[i].pt for (i, _) in matches])

            # compute the homography between the two sets of points
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return self.MatchResult(matches, H, status, outline, view)