# -*- coding: utf-8 -*-
""" Implements image reader capturing adapter.

"""

from .base import *
import cv2
from datetime import datetime


class ImagesReader(VisionBase):
    """Class for capturing frames from a list of images."""

    def __init__(self, image_paths, img_args=(), *args, **kwargs):
        """Initializes ImagesReader object

        :param image_paths: a list of image paths to be captured
        :param img_args: arguments for cv2.imread method
        """
        self._name = 'images'
        self._paths = image_paths[:]
        self._images = image_paths
        self._frame_count = len(image_paths)
        self._frame_index = 0
        self._img_args = img_args
        super(ImagesReader, self).__init__(*args, **kwargs)

    def setup(self):
        super(ImagesReader, self).setup()
        self._frame_index = 0

    def release(self):
        super(ImagesReader, self).release()

    def capture(self):
        super(ImagesReader, self).capture()
        if not self._frame_index < self._frame_count:
            return None
        image = ImagesReader.load_image(self._paths[self._frame_index], _self=self, img_args=self._img_args)
        self._frame_index += 1
        timestamp = datetime.now()
        if self.display_results:
            cv2.imshow(self.name, image.image)
        return Frame(timestamp, self._frame_index - 1, (image, ))

    @staticmethod
    def load_image(image_path, mask_path=None, _self=None, img_args=()):
        """Reads an imagefile and returns an Image object.

        :param image_path: path to a supported image file
        :param mask_path: path to a supported image file that will be used as a mask
        :param _self: Owner of the image. default = None
        :param img_args: arguments for cv2.imread method
        :return: Image object
        """
        image = cv2.imread(image_path, *img_args)
        if image is None:
            raise IOError("Could not read the image: " + image_path)
        mask = None
        if mask_path:
            mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY)
            if mask is None:
                raise IOError("Could not read the mask image: " + mask_path)
        return Image(_self, image, mask=mask)

    @property
    def frame_size(self):
        """Not supported"""
        return (0, 0)

    @property
    def fps(self):
        """Not supported"""
        return 0

    @property
    def frame_count(self):
        return self._frame_count

    @property
    def is_open(self):
        return self._frame_index < self._frame_count

    @property
    def name(self):
        return self._name

    @property
    def description(self):
        return "Monocular image file reader"

    @property
    def path(self):
        return self._paths[self._frame_index]

    @property
    def devices(self):
        """Not supported"""
        return ()

    def display_results_changed(self, last, current):
        """Will create a cv2 namedWindow if display_results is set to True"""
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)

    @property
    def autoexposure(self):
        """Not supported"""
        return False

    @property
    def autofocus(self):
        """Not supported"""
        return False

    @property
    def autowhitebalance(self):
        """Not supported"""
        return False

    @property
    def autogain(self):
        """Not supported"""
        return False

    @property
    def exposure(self):
        """Not supported"""
        return 0

    @property
    def focus(self):
        """Not supported"""
        return 0

    @property
    def whitebalance(self):
        """Not supported"""
        return 0

    @property
    def gain(self):
        """Not supported"""
        return 0
