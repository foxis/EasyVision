# -*- coding: utf-8 -*-
from EasyVision.base import *
from EasyVision.exceptions import *
from EasyVision.vision.base import VisionBase


class EngineBase(EasyVisionBase):
    def __init__(self, vision, *args, **kwargs):
        if not isinstance(vision, VisionBase):
            raise TypeError("Object must be VisionBase instance")
        self._vision = vision
        super(EngineBase, self).__init__(*args, **kwargs)

    def next(self):
        if self._vision.is_open:
            return self.compute()
        else:
            raise StopIteration()

    def __len__(self):
        return len(self._vision)

    @abstractmethod
    def compute(self):
        pass

    @abstractproperty
    def capabilities(self):
        pass

    @property
    def vision(self):
        return self._vision