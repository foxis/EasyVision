# -*- coding: utf-8 -*-
from .base import *
import cv2


class HistogramBackprojection(ProcessorBase):
    CHANNELS = (0, 1)
    RANGES = (0, 180, 0, 256)
    RANGE_MIN = (0., 20., 22.)
    RANGE_MAX = (180., 255., 255.)
    BINS = (32, 32)

    def __init__(self, vision, histogram, combine_masks=False, invert=False, channels=CHANNELS, ranges=RANGES,
                 range_min=RANGE_MIN, range_max=RANGE_MAX, *args, **kwargs):
        if isinstance(histogram, tuple) and not histogram:
            raise ValueError("If Histogram is a list of histograms, then it must not be empty.")
        self._hist = histogram if isinstance(histogram, tuple) else (histogram, )
        self._channels = channels
        self._ranges = ranges
        self._range_min = range_min
        self._range_max = range_max
        self._invert = invert
        self._combine_masks = combine_masks
        super(HistogramBackprojection, self).__init__(vision, *args, **kwargs)

    def setup(self):
        super(HistogramBackprojection, self).setup()

    def release(self):
        super(HistogramBackprojection, self).release()

    @property
    def description(self):
        return "Color Histogram backprojection processor"

    @staticmethod
    def calculate_histogram(image, mask=None, channels=CHANNELS, bins=BINS, ranges=RANGES,
                            range_min=RANGE_MIN, range_max=RANGE_MAX):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _mask = cv2.inRange(hsv, range_min, range_max)
        if mask is not None:
            _mask = cv2.bitwise_and(_mask, mask)
        hist = cv2.calcHist((hsv,), channels, _mask, bins, ranges)
        return cv2.normalize(hist, None, 0, 256, cv2.NORM_MINMAX)

    def process(self, image):
        hsv = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV)
        _mask = cv2.inRange(hsv, self._range_min, self._range_max)

        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        masks = ()
        for hist in self._hist:
            mask = cv2.calcBackProject((hsv,), self._channels, hist, self._ranges, 1)
            mask = cv2.filter2D(mask, -1, disc)
            mask = cv2.blur(mask, (50, 50))
            _, mask = cv2.threshold(mask, 50, 255, 0)
            #mask = cv2.bitwise_and(_mask, mask)
            if self._invert:
                mask = 255 - mask
            if image.mask is not None:
                mask = cv2.bitwise_and(mask, image.mask)
            masks += (mask,)

        if self._combine_masks:
            masks = sum(masks)

        if self.display_results:
            _masks = masks if self._combine_masks else sum(masks)
            cv2.imshow("%s" % self.name, _masks)

        return image._replace(mask=masks)
