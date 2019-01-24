# -*- coding: utf-8 -*-
from EasyVision.base import *
from EasyVision.exceptions import *
from EasyVision.vision.base import VisionBase
from EasyVision.processors.base import Features
import numpy as np
from collections import namedtuple


class EngineCapability(namedtuple('EngineCapability', 'inputs outputs misc')):
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


class Pose(namedtuple('Pose', 'timestamp rotation translation features')):
    """A class describing agent pose

        rotation: a rotation matrix
        translation: a translation vector
    """
    __slots__ = ()

    def __new__(cls, timestamp, rotation, translation, features=None):
        if not isinstance(features, Features) and features is not None:
            raise TypeError("Features must be of type Features")
        if not isinstance(rotation, np.ndarray):
            rotation = np.float32(rotation)
        if not isinstance(translation, np.ndarray):
            translation = np.float32(translation)
        return super(Pose, cls).__new__(cls, timestamp, rotation, translation, features)

    def todict(self):
        d = {
            'timestamp': self.timestamp,
            'rotation': self.rotation.tolist(),
            'translation': self.translation.tolist(),
            'features': self.features.todict()
        }
        return d

    @staticmethod
    def fromdict(d):
        return Pose(d['timestamp'], d['rotation'], d['translation'], Features.fromdict(d['features']))


class OdometryBase(EngineBase):
    """An abstract Odometry Base class for Odometry Engines

    Abstract methods:
        ...
    Abstract properties:
        pose
        relative_pose
        camera_orientation
        feature_type
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


class MapBase(EasyVisionBase):
    """ Map Base is an abstract base class for Mapping with Visual Odometry

    Abstract methods:
        update
            Updates the map and returns (corrected) pose
        plan
            Given a target returns a path of poses to reach a target

    Abstract properties
        map_raw
            returns RAW map
        pose
            returns current pose
        path
            returns current path
    """
    def __init__(self, *args, **kwargs):
        super(MapBase, self).__init__(*args, **kwargs)

    @abstractproperty
    def map_raw(self):
        """Returns RAW map. Either a graph or np.array"""
        pass

    @abstractproperty
    def pose(self):
        """Returns current pose"""
        pass

    @abstractproperty
    def path(self):
        """Returns the whole path as a list of poses"""
        pass

    @abstractmethod
    def update(self, pose, *args, **kwargs):
        """Updates the map.

        :param pose: current pose of the robot
        :return: Current pose (may be different from input pose due to correction)
        """
        pass

    @abstractmethod
    def plan(self, target, radius, **kwargs):
        """Plans a path towards a target

        :param target: a target based on map implementation
        :param radius: radius of agent for obstacle detection
        :return: a list of poses
        """
        pass