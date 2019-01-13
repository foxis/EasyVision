# -*- coding: utf-8 -*-
import cv2
from .base import *


class BackgroundSeparation(ProcessorBase):
    def __init__(self, vision, algorithm='MOG', *args, **kwargs):
        if algorithm not in ('MOG', 'MOG2', 'GMG', 'grabCut', 'diff'):
            raise ValueError("Algorithm must be one of MOG/MOG2/GMG")
        self._algorithm = algorithm
        self._background_num = 0
        self._max_background_num = 200

        super(BackgroundSeparation, self).__init__(vision, *args, **kwargs)

    def setup(self):
        if self._algorithm == 'grabCut':
            bgdModel = np.zeros((1,65),np.float64)
            fgdModel = np.zeros((1,65),np.float64)
        elif self._algorithm == 'diff':
            pass
        else:
            self._subtractor = {
                'MOG': cv2.bgsegm.createBackgroundSubtractorMOG,
                'MOG2': cv2.createBackgroundSubtractorMOG2,
                'GMG': cv2.bgsegm.createBackgroundSubtractorGMG
            }[self._algorithm](history=self._max_background_num)
        super(BackgroundSeparation, self).setup()

    def release(self):
        super(BackgroundSeparation, self).release()

    @property
    def description(self):
        return "Background Separation processor"

    @property
    def background_num(self):
        return self._background_num

    @background_num.setter
    def background_num(self, value):
        self._background_num = value

    def process(self, image):
        if self._algorithm == 'grabCut':
            return self._process_grabCut(image)
        elif self._algorithm == 'diff':
            return self._process_diff(image)
        else:
            lr = 0 if self._background_num > self._max_background_num else -1
            mask = self._subtractor.apply(image.image, learningRate=lr)
            self._background_num += 1
            return image._replace(mask=mask)

    def _process_grabCut(image):
        cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    def _process_diff(image):
        if self._background is None:
            self._background = image.image
            self._mask = np.zeros(img.shape[:2],np.uint8)

        if self._background_num < self._max_background:
            self._background = (self._background + image.image) / 2
            return image._replace(mask=self._mask)

        mask = image.image - self._background
        mask = cv2.inRange(mask, (40, 40, 40), (255, 255, 255))