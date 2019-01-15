# -*- coding: utf-8 -*-
from EasyVision.base import *
from EasyVision.exceptions import *
from EasyVision.vision.base import VisionBase
import numpy as np
from collections import namedtuple


class EngineCapability(namedtuple('EngineCapability', 'inputs', 'outputs', 'misc')):
    """A class describing Engine Capability

        inputs: a list of input Processor classes supported
        outputs: a list of classes returned. e.g. (Frame, Pose) for Odometry or (Frame, MatchResults) for Object Recognition
        misc: Miscellaneous information about the engine specific to the algorithms implemented
    """
    pass


class EngineBase(EasyVisionBase):
    """EngineBase is a base class for all engines that implement various algorithms such as Visual Odometry or Object Recognition

    """
    def __init__(self, vision, *args, **kwargs):
        if not isinstance(vision, VisionBase):
            raise TypeError("Object must be VisionBase instance")
        self._vision = vision
        super(EngineBase, self).__init__(*args, **kwargs)

    def next(self):
        if self._vision.is_open:
            result = self.compute()
            if result:
                return result
            else:
                raise StopIteration()
        else:
            raise StopIteration()

    def __len__(self):
        return len(self._vision)

    @abstractmethod
    def compute(self):
        """ Will compute the algorithm and return a frame and computation result
        """
        pass

    def setup(self):
        self._vision.setup()
        super(EngineBase, self).setup()

    def release(self):
        self._vision.release()
        super(EngineBase, self).release()

    @abstractproperty
    def capabilities(self):
        """Will return capabilities of the algorithm.

        :return: EngineCapability class
        """
        pass

    @property
    def vision(self):
        return self._vision


class Pose(namedtuple('Pose', ['rotation', 'translation'])):
    """A class describing agent pose

        rotation: a rotation matrix
        translation: a translation vector
    """

    def todict(self):
        d = {
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist()
        }
        return d

    @staticmethod
    def fromdict(d):
        return Pose(np.float32(d['rotation']), np.float32(d['translation']))


class OdometryBase(EngineBase):
    """An abstract Odometry Base class for Odometry Engines
    """
    def __init__(self, *args, **kwargs):
        super(OdometryBase, self).__init__(*args, **kwargs)

    @abstractproperty
    def pose(self):
        """Cumulative pose
        """
        pass

    @abstractproperty
    def relative_pose(self):
        """Relative pose from t-1 to t
        """
        pass

    @abstractproperty
    def camera_orientation(self):
        """Camera transformation, that determines the relative position of the camera relative to the agent origin"""
        pass

    @abstractproperty
    def feature_type(self):
        """Feature Type that the odometry is working with"""
        pass