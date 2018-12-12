# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod, abstractproperty


class EngineBase(ABC):
    def __init__(self, *args, **kwargs):
        super(EngineBase, self).__init__(*args, **kwargs)

    @abstractmethod
    def init(self, frame=None, timestamp=None):
        pass

    @abstractmethod
    def compute(self, frame, timestamp=None):
        pass

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def description(self):
        pass

    @abstractproperty
    def capabilities(self):
        pass
