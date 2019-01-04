# -*- coding: utf-8 -*-
import cv2
from .base import *


class ImageTransform(ProcessorBase):
    def __init__(self, vision, ocl=False, color=None, operator=None, *args, **kwargs):
        self._color = color
        self._ocl = ocl
        self._operator = operator

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
        if self._color:
            img = cv2.cvtColor(img, self._color)
        if self._operator:
            img = self._operator(img)

        if self.display_results:
            cv2.imshow(self.name, img)

        return image._replace(image=img)