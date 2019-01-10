# -*- coding: utf-8 -*-
from .base import *
import cv2


class HistogramBackprojection(ProcessorBase):

    def __init__(self, vision, histogram, channels=(0, 1), ranges=(0, 180, 0, 256),
                 range_min=(0., 40., 32.), range_max=(180., 255., 255.), *args, **kwargs):
        if isinstance(histogram, tuple) and not histogram:
            raise ValueError("If Histogram is a list of histograms, then it must not be empty.")
        self._hist = histogram if isinstance(histogram, tuple) else (histogram, )
        self._channels = channels
        self._ranges = ranges
        self._range_min = range_min
        self._range_max = range_max
        super(HistogramBackprojection, self).__init__(vision, *args, **kwargs)

    def setup(self):
        super(HistogramBackprojection, self).setup()

    def release(self):
        super(HistogramBackprojection, self).release()

    @property
    def description(self):
        return "Color Histogram backprojection processor"

    @staticmethod
    def calculate_histogram(image, mask=None, channels=(0, 1), bins=(32, 32), ranges=(0, 180, 0, 256),
                            range_min=(0., 40., 32.), range_max=(180., 255., 255.)):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _mask = cv2.inRange(hsv, range_min, range_max)
        if mask is not None:
            _mask = cv2.bitwise_and(_mask, mask)
        hist = cv2.calcHist((hsv,), channels, _mask, bins, ranges)
        return cv2.normalize(hist, None, 0, 256, cv2.NORM_MINMAX)

    def process(self, image):
        hsv = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        _mask = cv2.inRange(hsv, self._range_min, self._range_max)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        masks = ()
        result = image.image
        for hist in self._hist:
            mask = cv2.calcBackProject((hsv,), self._channels, hist, self._ranges, 1)
            mask = cv2.filter2D(mask, -1, disc)
            _, mask = cv2.threshold(mask, 50, 255, 0)
            mask = cv2.bitwise_and(_mask, mask)
            masks += (mask,)
            mask = cv2.merge((mask, mask, mask))
            result = cv2.bitwise_and(result, mask)

        if self.display_results:
            cv2.imshow("result", result)

        return image._replace(image=result, mask=masks, original=image.image)
