# -*- coding: utf-8 -*-
from EasyVision.base import *
from collections import namedtuple
from datetime import datetime
from operator import itemgetter


_Image = namedtuple('_Image', ['source', 'image', 'original', 'mask', 'features', 'feature_type'])


class Image(NamedTupleExtendHelper, _Image):
    __slots__ = ()

    def __new__(cls, source, image, original=None, mask=None, features=None, feature_type=None):
        if source is not None and not isinstance(source, VisionBase):
            raise TypeError("Source must be VisionBase")
        return tuple.__new__(cls, (source, image, original, mask, features, feature_type))

    def tobytes(self):
        raise NotImplementedError()

    @staticmethod
    def frombytes(data):
        raise NotImplementedError()

    def tobuffer(self):
        raise NotImplementedError()

    @staticmethod
    def frombuffer(data):
        raise NotImplementedError()


class Frame(NamedTupleExtendHelper, namedtuple('Frame', ['timestamp', 'index', 'images'])):
    __slots__ = ()

    def __new__(cls, timestamp, index, images):
        if not isinstance(timestamp, datetime):
            raise TypeError("Timestamp must be datetime object")
        if not isinstance(index, int):
            raise TypeError("Index must be integer")
        if not isinstance(images, tuple) or not all(isinstance(i, Image) for i in images):
            raise TypeError("Images must be a tuple of Image objects")
        return super(Frame, cls).__new__(cls, timestamp, index, tuple(images))

    def get_image(self, source):
        if isinstance(source, VisionBase):
            for img in self.images:
                if img.source == source:
                    return img
        elif isinstance(source, str):
            for img in self.images:
                if isinstance(img.source, VisionBase) and img.source.name == source:
                    return img
        else:
            raise TypeError("Only either VisionBase or string are supported")
        return None


class VisionBase(EasyVisionBase):
    def __init__(self, *args, **kwargs):
        super(VisionBase, self).__init__(*args, **kwargs)

    def __len__(self):
        return self.frame_count

    def next(self):
        super(VisionBase, self).next()
        frame = self.capture()
        if frame is None:
            raise StopIteration()
        return frame

    @abstractmethod
    def capture(self):
        super(VisionBase, self).next()

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