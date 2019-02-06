# -*- coding: utf-8 -*-
"""Contains base and helper classes for object models

"""

from EasyVision.base import *
from EasyVision.processors import Features
from collections import namedtuple
import numpy as np


class ModelView(NamedTupleExtendHelper, namedtuple('ModelView', 'image outline features feature_type')):
    """Structure holding feature-based model view

    Contains these fields:
        image - cropped image of the object
        outline - np.array of outline points
        features - Features object containing keypoints and descriptors
        feature_type - feature type of the Features object
    """

    __slots__ = ()

    def __new__(cls, image, outline, features, feature_type):
        return super(ModelView, cls).__new__(cls, image, outline, features, feature_type)

    def todict(self):
        """Convert ModelView to dict"""
        d = {
            'image': self.image.tolist(),
            'outline': self.outline.tolist(),
            'features': self.features.todict(),
            'feature_type': self.feature_type
        }
        return d

    @staticmethod
    def fromdict(d):
        """Create ModelView from dict"""
        image = None if d['image'] is None else np.uint8(d['image'])
        features = Features.fromdict(d['features'])
        outline = np.float32(d['outline'])
        return ModelView(image, outline, features, d['feature_type'])


class ModelBase(EasyVisionBase):
    """Base class for object Model

    """

    __slots__ = ('_name', '_views', '_view_index')

    def __init__(self, name, views, *args, **kwargs):
        if not all(isinstance(view, ModelView) for view in views):
            raise TypeError("Views must be iterable with items of type ModelView")
        self._view_index = 0
        self._name = name
        self._views = [i for i in views]
        super(ModelBase, self).__init__(*args, **kwargs)

    def __len__(self):
        """Will return number of views"""
        return len(self._views)

    def __iter__(self):
        """Will iterate over views"""
        self._view_index = 0
        return self

    def setup(self):
        """This method is not required for models"""
        pass

    def release(self):
        """This method is not required for models"""
        pass

    def next(self):
        if self._view_index < len(self._views):
            result = self._views[self._view_index]
            self._view_index += 1
            return result
        else:
            raise StopIteration()

    @property
    def name(self):
        return self._name

    @abstractmethod
    def compute(self, frame, **kwargs):
        pass

    def update(self, model):
        """Method to update models with a view or join two model views"""
        if isinstance(model, ModelView):
            self._views += [model]
        elif isinstance(model, ModelBase):
            self._views += model._views
        else:
            raise TypeError("Model must be either ModelBase or ModelView")