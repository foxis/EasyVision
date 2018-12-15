# -*- coding: utf-8 -*-
from EasyVision.base import *
from EasyVision.exceptions import *
from collections import namedtuple


class ModelView(NamedTupleExtendHelper, namedtuple('ModelView', ['image', 'outline', 'features', 'feature_type'])):
    __slots__ = ()

    def __new__(cls, image, outline, features, feature_type):
        return super(ModelView, cls).__new__(cls, image, outline, features, feature_type)


class ModelBase(EasyVisionBase):
    __slots__ = ('_name', '_views', '_view_index')

    def __init__(self, name, views, *args, **kwargs):
        if not all(isinstance(view, ModelView) for view in views):
            raise TypeError("Views must be iterable with items of type ModelView")
        super(ModelBase, self).__init__(*args, **kwargs)
        self._view_index = 0
        self._name = name
        self._views = [i for i in views]

    def __len__(self):
        return len(self._views)

    def __iter__(self):
        self._view_index = 0
        return self

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
        if isinstance(model, ModelView):
            self._views += [model]
        elif isinstance(model, ModelBase):
            self._views += model._views
        else:
            raise TypeError("Model must be either ModelBase or ModelView")