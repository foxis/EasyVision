# -*- coding: utf-8 -*-
"""Contains Base and helper classes for image processing algorithms.
Processors can be stacked by passing other processors or capturing adaptors as first argument.
Can also be used to build processor stacks with the help of Processor Stack Builder.

"""

from EasyVision.vision.base import *
import cv2
import numpy as np

try:
    from future_builtins import zip
except:
    pass

try:
    import cPickle as pickle
except:
    import pickle


class KeyPoint(namedtuple('KeyPoint', 'pt size angle response octave class_id')):
    """KeyPoint struct that mirrors cv2.KeyPoint. Mainly used for serialization as pickle does not understand cv2.KeyPoint.
    """
    __slots__ = ()

    def todict(self):
        """Converts KeyPoint into a dictionary"""
        return self._asdict()

    @staticmethod
    def fromdict(d):
        """Creates KeyPoint object from a dictionary"""
        return KeyPoint(**d)


class Features(namedtuple('Features', 'points descriptors points3d')):
    """Image Features structure.
    Contains feature points either as 2d points or KeyPoint, descriptors and associated 3d points.
    Basically points can be anything.
    """
    __slots__ = ()

    def __new__(cls, points, descriptors, points3d=None):
        if len(points) and hasattr(points[0], 'pt'):
            points = tuple(KeyPoint(pt.pt, pt.size, pt.angle, pt.response, pt.octave, pt.class_id) for pt in points)
        elif not isinstance(points, np.ndarray):
            points = np.float32(points)
        if not isinstance(points3d, np.ndarray) and points3d is not None:
            points3d = np.float32(points3d)
        return super(Features, cls).__new__(cls, points, descriptors, points3d)

    @property
    def keypoints(self):
        """Returns a list of cv2.KeyPoint items for displaying purposes"""
        return [cv2.KeyPoint(x=pt.pt[0], y=pt.pt[1], _size=pt.size, _angle=pt.angle,
                             _response=pt.response, _octave=pt.octave, _class_id=pt.class_id) for pt in self.points]

    def todict(self):
        """Converts Features into a dictionary"""
        d = {
            'points': [pt.todict() for pt in self.points] if len(points) and isinstance(points[0], KeyPoint) else self.points.tolist(),
            'points3d': self.points.tolist(),
            'descriptors': self.descriptors.tolist(),
            'dtype': self.descriptors.dtype.name
        }
        return d

    @staticmethod
    def fromdict(d):
        """Creates Features object from a dictionary"""
        pts = d['points']
        if len(pts) and isinstance(pts[0], dict):
            points = [KeyPoint.fromdict(pt) for pt in pts]
        else:
            points = pts
        descriptors = np.array(d['descriptors'], dtype=np.dtype(d['dtype']))
        return Features(points, descriptors, d['points3d'])

    def tobytes(self):
        """Uses pickle to serialize Features into bytes"""
        return pickle.dumps(self, protocol=-1)

    @staticmethod
    def frombytes(data):
        """Uses pickle to deserialize Features from bytes"""
        return pickle.loads(data)

    def tobuffer(self, buf):
        """Uses pickle to serialize Features to buffer-like object"""
        pickle.dump(self, buf, protocol=-1)

    @staticmethod
    def frombuffer(buf):
        """Uses pickle to deserialize Features from buffer-like object"""
        return pickle.load(buf)

    def __reduce__(self):
        """Used for pickle serialization in order to deal with UMat descriptors"""
        return self.__class__, (self.points, self.descriptors.get() if isinstance(self.descriptors, cv2.UMat) else self.descriptors)


class ProcessorBase(VisionBase):
    """Abstract Base class for image processor algorithms

    Capture will call process on each image in the frame if associated mask is set to '1'.
    Mask is a string of '1' and '0', where index of the mask is the same as the index of frame image.
    Processor mask will override frame processor mask.

    Abstract methods:
        process

    """

    def __init__(self, vision, processor_mask=None, append=False, enabled=True, *args, **kwargs):
        """Instance initialization. Must be called using super().__init__(*args, *kwargs)

        :param vision: capturing source object.
        :param processor_mask: a mask specifying which images in a frame should be processed
        :param append: indicates whether to replace images or append to the frame
        :param enabled: indicates whether to run processing
        """
        if not isinstance(vision, VisionBase) and vision is not None:
            raise TypeError("Vision object must be of type VisionBase")
        self._vision = vision
        self._processor_mask = Frame.tidy_processor_mask(processor_mask)
        self._enabled = True
        self._append = append
        self.enabled = enabled
        super(ProcessorBase, self).__init__(*args, **kwargs)

    @abstractmethod
    def process(self, image):
        """Processes frame image. Must return a valid Image instance
        Note, that this method will only be called if self.enabled is True and associated processor mask is '1'

        :param image: instance of Image struct - input image
        :return: an instance of Image struct.
        """
        pass

    def capture(self):
        super(ProcessorBase, self).capture()
        frame = self._vision.capture()
        if not self.enabled:
            return frame
        elif frame:
            processor_mask = self._processor_mask if self._processor_mask is not None else frame.processor_mask
            if processor_mask is None:
                processor_mask = "1" * len(frame.images)
            if not self._append:
                images = tuple(m == "0" and img or self.process(img)._replace(source=self) for m, img in zip(processor_mask, frame.images))
            else:
                images = tuple(self.process(img)._replace(source=self) for m, img in zip(processor_mask, frame.images) if m != "0")
                images = frame.images + images
            return frame._replace(images=images)

    def setup(self):
        if self._vision is not None:
            self._vision.setup()
        super(ProcessorBase, self).setup()

    def release(self):
        if self._vision is not None:
            self._vision.release()
        super(ProcessorBase, self).release()

    @property
    def source(self):
        """Returns a source for this processor"""
        return self._vision

    def get_source(self, name):
        """Recursively searches for a source class by class name"""
        if self.__class__.__name__ == name:
            return self
        elif isinstance(self._vision, ProcessorBase) or self._vision.__class__.__name__ == 'CameraPairProxy':
            return self._vision.get_source(name)
        elif self._vision.__class__.__name__ == name:
            return self._vision

    def __getattr__(self, name):
        """Allows to access attributes of deeper sources"""
        return getattr(self._vision, name)

    @property
    def enabled(self):
        """Sets/Gets a flag indicated whether process method should be called"""
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        lastenabled, self._enabled = self._enabled, value
        if lastenabled != value and hasattr(self, 'enabled_changed'):
            self.enabled_changed(lastenabled, value)

    @property
    def is_open(self):
        return self._vision.is_open

    @property
    def frame_size(self):
        return self._vision.frame_size

    @property
    def fps(self):
        return self._vision.fps

    @property
    def name(self):
        return "{} <- {}".format(super(ProcessorBase, self).name, self._vision.name)

    @property
    def frame_count(self):
        return self._vision.frame_count

    @property
    def path(self):
        return self._vision.path

    @property
    def devices(self):
        return self._vision.devices

    def display_results_changed(self, last, current):
        if current:
            cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        else:
            cv2.destroyWindow(self.name)

    @property
    def autoexposure(self):
        return self._vision.autoexposure

    @property
    def autofocus(self):
        return self._vision.autofocus

    @property
    def autowhitebalance(self):
        return self._vision.autowhitebalance

    @property
    def autogain(self):
        return self._vision.autogain

    @property
    def exposure(self):
        return self._vision.exposure

    @property
    def focus(self):
        return self._vision.focus

    @property
    def whitebalance(self):
        return self._vision.whitebalance

    @property
    def gain(self):
        return self._vision.gain

    @autoexposure.setter
    def autoexposure(self, value):
        self._vision.autoexposure = value

    @autofocus.setter
    def autofocus(self, value):
        self._vision.autofocus = value

    @autowhitebalance.setter
    def autowhitebalance(self, value):
        self._vision.autowhitebalance = value

    @autogain.setter
    def autogain(self, value):
        self._vision.autogain = value

    @exposure.setter
    def exposure(self, value):
        self._vision.exposure = value

    @focus.setter
    def focus(self, value):
        self._vision.focus = value

    @whitebalance.setter
    def whitebalance(self, value):
        self._vision.whitebalance = value

    @gain.setter
    def gain(self, value):
        self._vision.gain = value