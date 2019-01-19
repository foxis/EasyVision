# -*- coding: utf-8 -*-
from EasyVision.vision.base import *
import cv2
import numpy as np
import cPickle
from future_builtins import zip


class KeyPoint(namedtuple('KeyPoint', ['pt', 'size', 'angle', 'response', 'octave', 'class_id'])):
    def todict(self):
        return self._asdict()

    @staticmethod
    def fromdict(d):
        return KeyPoint(**d)


class Features(namedtuple('Features', ['points', 'descriptors'])):
    __slots__ = ()

    def __new__(cls, points, descriptors):
        points = [KeyPoint(pt.pt, pt.size, pt.angle, pt.response, pt.octave, pt.class_id) for pt in points] if len(points) and isinstance(points[0], cv2.KeyPoint) else points
        return super(Features, cls).__new__(cls, points, descriptors)

    @property
    def keypoints(self):
        return [cv2.KeyPoint(x=pt.pt[0], y=pt.pt[1], _size=pt.size, _angle=pt.angle,
                             _response=pt.response, _octave=pt.octave, _class_id=pt.class_id) for pt in self.points]

    def todict(self):
        d = {
            'points': [pt.todict() for pt in self.points],
            'descriptors': self.descriptors.tolist(),
            'dtype': self.descriptors.dtype.name
        }
        return d

    @staticmethod
    def fromdict(d):
        points = [KeyPoint.fromdict(pt) for pt in d['points']]
        descriptors = np.array(d['descriptors'], dtype=np.dtype(d['dtype']))
        return Features(points, descriptors)

    def tobytes(self):
        return cPickle.dumps(self, protocol=-1)

    @staticmethod
    def frombytes(data):
        return cPickle.loads(data)

    def tobuffer(self, buf):
        cPickle.dump(self, buf, protocol=-1)

    @staticmethod
    def frombuffer(buf):
        return cPickle.load(self, buf)

    def __reduce__(self):
        return (self.__class__, (self.points, self.descriptors.get() if isinstance(self.descriptors, cv2.UMat) else self.descriptors))


class ProcessorBase(VisionBase):

    def __init__(self, vision, processor_mask=None, enabled=True, *args, **kwargs):
        if not isinstance(vision, VisionBase) and vision is not None:
            raise TypeError("Vision object must be of type VisionBase")
        self._vision = vision
        self._processor_mask = Frame.tidy_processor_mask(processor_mask)
        self._enabled = True
        self.enabled = enabled
        super(ProcessorBase, self).__init__(*args, **kwargs)

    @abstractmethod
    def process(self, image):
        pass

    def capture(self):
        super(ProcessorBase, self).capture()
        frame = self._vision.capture()
        if not self.enabled:
            return frame
        elif frame:
            processor_mask = frame.processor_mask if frame.processor_mask else self._processor_mask
            if not processor_mask:
                processor_mask = "1" * len(frame.images)
            images = tuple(m == "0" and img or self.process(img)._replace(source=self) for m, img in zip(processor_mask, frame.images))
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
        return self._vision

    def get_source(self, name):
        if self.__class__.__name__ == name:
            return self
        elif isinstance(self._vision, ProcessorBase) or self._vision.__class__.__name__ == 'CameraPairProxy':
            return self._vision.get_source(name)
        elif self._vision.name == name:
            return self._vision

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            return super(ProcessorBase, self).__getattr__(name)
        return getattr(self._vision, name)

    @property
    def enabled(self):
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