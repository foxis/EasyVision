# -*- coding: utf-8 -*-
"""Implements BOG algorithm using OpenCV.
MOG, MOG2 and GMG algorithms are exposed.

"""

import cv2
from .base import *


class BackgroundSeparation(ProcessorBase):
    """Class that produces a mask calculated using different Background separation algorithms present in OpenCV.

    Currently supports MOG, MOG2 and GMG algorithms.
    When ``max_background_num`` is set, will stop learning background model after specified number of frames.
    By setting ``background_num`` you can reset background learning.
    """

    def __init__(self, vision, algorithm='MOG', *args, **kwargs):
        if algorithm not in ('MOG', 'MOG2', 'GMG'):
            raise ValueError("Algorithm must be one of MOG/MOG2/GMG")
        self._algorithm = algorithm
        self._background_num = 0
        self._max_background_num = 200

        super(BackgroundSeparation, self).__init__(vision, *args, **kwargs)

    def setup(self):
        self._subtractor = {
            'MOG': cv2.bgsegm.createBackgroundSubtractorMOG,
            'MOG2': cv2.createBackgroundSubtractorMOG2,
            'GMG': cv2.bgsegm.createBackgroundSubtractorGMG
        }[self._algorithm](history=self._max_background_num)
        self._disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        super(BackgroundSeparation, self).setup()

    def release(self):
        super(BackgroundSeparation, self).release()

    @property
    def description(self):
        return "Background Separation processor"

    @property
    def background_num(self):
        """Setter/Getter for resetting background separation learning"""
        return self._background_num

    @background_num.setter
    def background_num(self, value):
        self._background_num = value

    def process(self, image):
        lr = 0 if self._background_num > self._max_background_num else .9
        mask = self._subtractor.apply(image.image, learningRate=lr)
        mask = cv2.filter2D(mask, -1, self._disc)
        _, mask = cv2.threshold(mask, 100, 255, 0)
        if image.mask is not None:
            mask = cv2.bitwise_and(mask, image.mask)
        self._background_num += 1
        if self.display_results:
             cv2.imshow("%s" % self.name, mask)
        return image._replace(mask=mask)
