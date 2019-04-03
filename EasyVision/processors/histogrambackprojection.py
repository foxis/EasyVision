# -*- coding: utf-8 -*-
"""Implements Histogram Backprojection using OpenCV

"""

from .base import *
import cv2


class HistogramBackprojection(ProcessorBase):
    """Class that implements Histogram Backprojection algorithm

    Provided a tuple of histograms will calculate masks for each histogram.
    If ``combine_masks`` is False, then ``Image.mask`` will be a tuple containing all the masks for each histogram.
    Otherwise ``Image.mask`` will contain a combined mask.
    """

    CHANNELS = (0, 1)
    RANGES = (0, 180, 0, 256)
    RANGE_MIN = (0., 20., 22.)
    RANGE_MAX = (180., 255., 255.)
    BINS = (32, 32)

    def __init__(self, vision, histogram, combine_masks=False, invert=False, channels=CHANNELS, ranges=RANGES,
                 range_min=RANGE_MIN, range_max=RANGE_MAX, *args, **kwargs):
        """HistogramBackprojection instance initialization

        :param vision: source capturing object
        :param histogram: a tuple of histograms
        :param combine_masks: flag indicating that one should combine all the masks for all the histograms
        :param invert: flag indicating that resulting mask will be inverted (useful for e.g. hand rejection)
        :param channels: a tuple of channels used in histogram
        :param ranges: channel value ranges
        :param range_min: min value for filtering of color
        :param range_max: max value for filtering of color
        """
        if not isinstance(histogram, np.ndarray) and not isinstance(histogram, tuple) and not isinstance(histogram, list):
            raise TypeError("Histogram must be either a list, a tuple or numpy darray")

        if not isinstance(histogram, np.ndarray):
            if isinstance(histogram[0], float):
                histogram = np.float32(histogram)
            elif isinstance(histogram[0], list) or isinstance(histogram[0], tuple):
                histogram = tuple(np.float32(i) for i in histogram)
            else:
                raise TypeError("Wrong histogram format. Must be a numpy array, a list or a tuple or a list/tuple of ndarrays/lists/tuples")

        if isinstance(histogram, tuple) and not histogram:
            raise ValueError("If Histogram is a list of histograms, then it must not be empty.")
        self._hist = histogram if isinstance(histogram, tuple) else (histogram, )
        self._channels = channels
        self._ranges = ranges
        self._range_min = range_min
        self._range_max = range_max
        self._invert = invert
        self._cache_img = None
        self._cache_mask = None
        self._cache_mask1 = None
        self._combine_masks = combine_masks
        self._disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
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
        """Helper method to calculate a histogram from an image using a mask if provided.

        :param image: numpy array representing an BGR image
        :param mask: numpy array representing a grayscale image
        :param channels: a tuple of channels used to calculate the histogram
        :param bins: a tuple for number of bins for each channel
        :param ranges: channel value ranges
        :param range_min: min value for filtering of color
        :param range_max: max value for filtering of color
        :return: numpy array representing calculated histogram
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        _mask = cv2.inRange(hsv, range_min, range_max)
        if mask is not None:
            _mask = cv2.bitwise_and(_mask, mask, dst=_mask)
        hist = cv2.calcHist((hsv,), channels, _mask, bins, ranges)
        return cv2.normalize(hist, hist, 0, 256, cv2.NORM_MINMAX)

    def process(self, image):
        hsv = cv2.cvtColor(image.image, cv2.COLOR_BGR2HSV, dst=self._cache_img)
        #_mask = cv2.inRange(hsv, self._range_min, self._range_max)
        self._cache_img = hsv

        masks = ()
        for hist in self._hist:
            mask = cv2.calcBackProject((hsv,), self._channels, hist, self._ranges, 1, dst=self._cache_mask)
            self._cache_mask = mask
            mask = cv2.filter2D(mask, -1, self._disc, dst=mask)
            mask = cv2.blur(mask, (50, 50), dst=mask)
            _, mask = cv2.threshold(mask, 50, 255, 0, dst=mask)
            #mask = cv2.bitwise_and(_mask, mask)
            if self._invert:
                mask = 255 - mask
            if image.mask is not None:
                mask = cv2.bitwise_and(mask, image.mask, dst=mask)
            masks += (mask,)

        if self._combine_masks:
            masks = sum(masks)

        if self.display_results:
            _masks = masks if self._combine_masks else sum(masks)
            cv2.imshow("%s" % self.name, _masks)

        return image._replace(mask=masks)
