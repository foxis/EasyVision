# -*- coding: utf-8 -*-
"""A simple processor that does simple image transforms, e.g. color transform or transferring an image to a GPU

"""

import cv2
from .base import *


class ImageTransform(ProcessorBase):
    """Class implementing simple image transforms

    """

    def __init__(self, vision, ocl=False, color=None, operator=None, *args, **kwargs):
        """ImageTransform instance initialization

        :param vision: source capturing object
        :param ocl: will transform an image to UMat effectively transferring it to GPU and back
        :param color: specify ``cv2.COLOR_*`` which will be used for ``cvtColor``
        :param operator: specify a callable that will be called in ``process`` method
        """
        self._color = color
        self._ocl = ocl
        self._operator = operator
        self._cache_img = None

        super(ImageTransform, self).__init__(vision, *args, **kwargs)

    @property
    def description(self):
        return "Image Transform processor"

    def process(self, image):
        if not self._ocl and not self._color and not self._operator:
            return image

        img = image.image
        if self._ocl:
            img = cv2.UMat(img)
        elif isinstance(img, cv2.UMat):
            img = img.get()
        if self._color:
            img = cv2.cvtColor(img, self._color)
        if self._operator:
            img = self._operator(img)

        if self.display_results:
            cv2.imshow(self.name, img)

        self._cache_img = img
        return image._replace(image=img)
