# -*- coding: utf-8 -*-
from .base import *
from collections import namedtuple
import cv2
import numpy as np


class ObjectModel(ModelBase):
    __slots__ = ('_matcher')
    Keypoints = namedtuple('Keypoints', ['view', 'matches'])
    ComputeResult = namedtuple('ComputeResult', ['score', 'homography', 'matches'])

    def __init__(self, name, views, *args, **kwargs):
        super(ObjectModel, self).__init__(name, views, *args, **kwargs)
        self._matcher = cv2.DescriptorMatcher_create("BruteForce")

    def compute(self, frame, views):
        pass

    def release(self):
        pass

    @property
    def description(self):
        return "Simple feature(SIFT/ORB/KAZE/AKAZE) based object model"

    def _match_keypoints(self, kpsA, kpsB, featuresA, featuresB, ratio=0.75, reprojThresh=4.0):
        # compute the raw matches and initialize the list of actual
        # matches
        rawMatches = self._matcher.knnMatch(featuresA, featuresB, 2)
        matches = [(m[0].trainIdx, m[0].queryIdx) for m in rawMatches if len(m) == 2 and m[0].distance < m[1].distance * ratio]

        # computing a homography requires at least 4 matches
        if len(matches) > 4:
            # construct the two sets of points
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # compute the homography between the two sets of points
            H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # return the matches along with the homograpy matrix
            # and status of each matched point
            return (matches, H, status)