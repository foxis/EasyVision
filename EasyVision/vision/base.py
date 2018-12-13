# -*- coding: utf-8 -*-
from EasyVision.base import *
from collections import namedtuple
from datetime import datetime


class Image(namedtuple('Image', ['source', 'image'])):
    __slots__ = ()

    def __init__(self, source, image):
        if not isinstance(source, VisionBase):
            raise TypeError("Source must be VisionBase")
        super(Image, self).__init__(source, image)


class Frame(namedtuple('Frame', ['timestamp', 'index', 'images'])):
    __slots__ = ()

    def __init__(self, timestamp, index, images):
        if not isinstance(timestamp, datetime):
            raise TypeError("Timestamp must be datetime object")
        if not isinstance(index, int):
            raise TypeError("Index must be integer")
        if not isinstance(images, tuple) or not all(isinstance(i, Image) for i in images):
            raise TypeError("Images must be a tuple of Image objects")
        super(Frame, self).__init__(timestamp, index, images)


class VisionBase(EasyVisionBase):
    def __init__(self, *args, **kwargs):
        super(VisionBase, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.frame_count

    def next(self):
        if self.is_open:
            return self.capture()
        else:
            raise StopIteration()

    @abstractmethod
    def capture(self):
        """
        :return: either None or (timestamp, (image1, image2, image3, etc))
        """
        pass

    @abstractproperty
    def is_open(self):
        pass

    @abstractproperty
    def frame_size(self):
        pass

    @abstractproperty
    def fps(self):
        pass

    @abstractproperty
    def frame_count(self):
        pass

    @abstractproperty
    def path(self):
        pass

    @abstractproperty
    def devices(self):
        """
        :return: [{name:, description:, path:, etc:}]
        """
        pass