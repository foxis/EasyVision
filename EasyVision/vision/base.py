# -*- coding: utf-8 -*-
"""Base and helper classess for capturing adapters.

"""

from EasyVision.base import *
from collections import namedtuple
from datetime import datetime
import cv2

try:
    import cPickle as pickle
except ImportError:
    import pickle


class Image(NamedTupleExtendHelper, namedtuple('_Image', ['source', 'image', 'original', 'mask', 'features', 'feature_type'])):
    """Image is class derived from namedtuple and represents a captured and/or processed image.

    Contains such fields:
        source
            Is a reference to the algorithm that worked on the image last.
        image
            Current Image. Usually a ndarray representing the image.
        original
            Optional Original image before algorithm modified it.
        mask
            Optional mask for the image.
        features
            Features object. Depends on the type of features.
        feature_type
            String describing Feature Type. Depends on the type of features and algorithm that provided them.

    Implements methods:
        tobytes
            converts image object to bytes
        frombytes
            converts bytes to image object
        tobuffer
            writes bytes to buffer-like object
        frombuffer
            reads bytes from buffer-like object

    """

    __slots__ = ()

    def __new__(cls, source, image, original=None, mask=None, features=None, feature_type=None):
        if source is not None and not isinstance(source, VisionBase):
            raise TypeError("Source must be VisionBase")
        return super(Image, cls).__new__(cls, source, image, original, mask, features, feature_type)

    def tobytes(self):
        """Uses pickle to serialize Image object into bytes"""
        return pickle.dumps(self, protocol=-1)

    @staticmethod
    def frombytes(data):
        """Uses pickle to deserialize Image object from bytes"""
        return pickle.loads(data)

    def tobuffer(self, buf):
        """Uses pickle to serialize Image object into a buffer

        :param buf: Buffer-like object, that supports read/write methods
        :return: None
        """
        pickle.dump(self, buf, protocol=-1)

    @staticmethod
    def frombuffer(buf):
        """Uses pickle to deserialize Image object from a buffer

        :param buf: Buffer-like object, that supports read/write methods
        :return: Image object
        """
        return pickle.load(buf)

    def __reduce__(self):
        """Used for pickle to properly convert between UMat and numpy array"""
        d = (
            None,
            self.image.get() if isinstance(self.image, cv2.UMat) else self.image,
            None,
            self.mask.get() if isinstance(self.mask, cv2.UMat) else self.mask,
            self.features,
            self.feature_type,
        )
        return self.__class__, d


class Frame(NamedTupleExtendHelper, namedtuple('_Frame', ['timestamp', 'index', 'images', 'processor_mask'])):
    """Frame is a class derived from namedtuple and represents a synchronously captured/processed set of images.

    Contains fields:
        timestamp
            datetime object representing the timestamp of when the frame was taken.
        index
            Number of the frames taken prior.
        images
            A tuple containing Image objects.
        processor_mask
            A string of "1" and "0" that is used to determine if the image from the tuple has to be processed.

    Implements methods:
        get_image
            Returns an image from specified source object
        tidy_processor_mask
            Returns a string of "1" and "0" from iterable containing booleans

    """

    __slots__ = ()

    def __new__(cls, timestamp, index, images, processor_mask=None):
        if not isinstance(timestamp, datetime):
            raise TypeError("Timestamp must be datetime object")
        if not isinstance(index, int):
            raise TypeError("Index must be integer")
        if not isinstance(images, tuple) or not all(isinstance(i, Image) for i in images):
            raise TypeError("Images must be a tuple of Image objects")
        return super(Frame, cls).__new__(cls, timestamp, index, tuple(images), Frame.tidy_processor_mask(processor_mask))

    def get_image(self, source):
        """Returns an image from specified source object. Useful to get left/right views for stereo camera setup.

        :param source: an object that implements VisionBase interface.
        :return: An image from the images tuple that was produced by specified source or None if no such image found.
        :raises: TypeError
        """
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

    @staticmethod
    def tidy_processor_mask(processor_mask):
        """Converts an iterable of booleans into a string object of "1" and "0".

        :param processor_mask: either a tuple of booleans or a string of "1" and "0"
        :return: a string of "1" and "0"
        :raises: TypeError
        """
        if processor_mask is not None and not isinstance(processor_mask, tuple) and not isinstance(processor_mask, str):
            raise TypeError("Processor mask must be either a tuple of booleans or a string of 1 and 0")
        return "".join(i and "1" or "0" for i in processor_mask) if isinstance(processor_mask, tuple) else processor_mask

    def tobytes(self):
        """Uses pickle to serialize Frame object into bytes"""
        return pickle.dumps(self, protocol=-1)

    @staticmethod
    def frombytes(data):
        """Uses pickle to deserialize Frame object from bytes"""
        return pickle.loads(data)

    def tobuffer(self, buf):
        """Uses pickle to serialize Frame object into a buffer

        :param buf: Buffer-like object, that supports read/write methods
        :return: None
        """
        pickle.dump(self, buf, protocol=-1)

    @staticmethod
    def frombuffer(buf):
        """Uses pickle to deserialize Frame object from a buffer

        :param buf: Buffer-like object, that supports read/write methods
        :return: Frame object
        """
        return pickle.load(buf)


class VisionBase(EasyVisionBase):
    """VisionBase is an abstract base class for frame capturers and processors.

    Implements:
        __len__
            Returns a number of frames available. returns a self.frame_count property
        next
            Captures a frame by calling self.capture(). Will stop iteration if frame is None.

    Abstract methods/properties:
        capture
            Captures a frame. Will return None if no more frames available.
        is_open
            Property indicating if the capturing is still available.
        frame_size
            Property returning a tuple of width and height of the captured frame
        frame_count
            Returns a number of frames available.
        path
            Returns the current path of the image/video file.
        devices
            Returns a dictionary enumerating various capturing devices and their properties.

    Camera control properties
        fps
            A property returning number of frames per second the device is outputing.

        autoexposure
            A property to get/set automatic Exposure for video camera.
        autofocus
            A property to get/set automatic Focus for video camera.
        autowhitebalance
            A property to get/set automatic White Balance for video camera.
        autogain
            A property to get/set automatic Gain for video camera.

        exposure
            A property to get/set Exposure for video camera.
        focus
            A property to get/set Focus for video camera.
        whitebalance
            A property to get/set White Balance for video camera.
        gain
            A property to get/set Gain for video camera.

    """

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
        """Abstract method. Captures a frame from capturing device. ``setup`` must be called before calling this method."""
        super(VisionBase, self).next()

    @abstractproperty
    def is_open(self):
        """Abstract property. Indicates whether any frames are available."""
        pass

    @abstractproperty
    def frame_size(self):
        """Abstract property. Size of the frame that was requested from the device."""
        pass

    @abstractproperty
    def fps(self):
        """Abstract property. Frames per second that the opened device supports."""
        pass

    @abstractproperty
    def frame_count(self):
        """Abstract property. Number of frames available. Negative values mean indefinite amount of frames."""
        pass

    @abstractproperty
    def path(self):
        """Abstract property. Current image path or a capturing device."""
        pass

    @abstractproperty
    def devices(self):
        """Abstract property. A property to get available capturing devices the adaptor supports.

        :return: [{name:, description:, path:, etc:}]
        """
        pass

    @abstractproperty
    def autoexposure(self):
        """Abstract property. A property to get/set automatic Exposure for video camera."""
        pass

    @abstractproperty
    def autofocus(self):
        """Abstract property. A property to get/set automatic Focus for video camera."""
        pass

    @abstractproperty
    def autowhitebalance(self):
        """Abstract property. A property to get/set automatic White Balance for video camera."""
        pass

    @abstractproperty
    def autogain(self):
        """Abstract property. A property to get/set automatic Gain for video camera."""
        pass

    @abstractproperty
    def exposure(self):
        """Abstract property. A property to get/set Exposure for video camera."""
        pass

    @abstractproperty
    def focus(self):
        """Abstract property. A property to get/set Focus for video camera."""
        pass

    @abstractproperty
    def whitebalance(self):
        """Abstract property. A property to get/set White Balance for video camera."""
        pass

    @abstractproperty
    def gain(self):
        """Abstract property. A property to get/set Gain for video camera."""
        pass
