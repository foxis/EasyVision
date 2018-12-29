# -*- coding: utf-8 -*-
from .base import *
import cv2
from datetime import datetime


class ImagesReader(VisionBase):

    def __init__(self, image_paths, img_args=(), *args, **kwargs):
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
        image = cv2.imread(image_path, *img_args)
        if image is None:
            raise IOError("Could not read the image: " + image_path)
        mask = None
        if mask_path:
            mask = cv2.imread(mask_path)
            if mask is None:
                raise IOError("Could not read the mask image: " + mask_path)
        return ImageWithMask(_self, image, mask) if mask else Image(_self, image)

    @property
    def frame_size(self):
        return (0, 0)

    @property
    def fps(self):
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
        return ()

    def display_results_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)