# -*- coding: utf-8 -*-
import cv2
from .base import *


class BackgroundSeparation(ProcessorBase):
    def __init__(self, vision, algorithm='MOG', *args, **kwargs):
        if algorithm not in ('MOG', 'MOG2', 'GMG'):
            raise ValueError("Algorithm must be one of MOG/MOG2/GMG")
        self._algorithm = algorithm

        super(BackgroundSeparation, self).__init__(vision, *args, **kwargs)

    def setup(self):
        self._subtractor = {
            'MOG': cv2.bgsegm.createBackgroundSubtractorMOG,
            'MOG2': cv2.createBackgroundSubtractorMOG2,
            'GMG': cv2.bgsegm.createBackgroundSubtractorGMG
        }[self._algorithm]()
        super(BackgroundSeparation, self).setup()

    def release(self):
        super(BackgroundSeparation, self).release()
        del self._subtractor

    @property
    def description(self):
        return "Background Separation processor"

    def process(self, image):
        mask = self._subtractor.apply(image.image)
        return image._replace(mask=mask)