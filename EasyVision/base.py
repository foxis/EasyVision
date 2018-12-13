# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class EasyVisionBase(object):
    __metaclass__ = ABCMeta
    __slots__ = ['_debug', '_display_results']

    def __init__(self, *args, **kwargs):
        self._debug = False
        self.debug = kwargs.get('debug', False)
        self._display_results = False
        self.display_results = kwargs.get('display_results', False)
        super(EasyVisionBase, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.release()

    def __iter__(self):
        return self

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def description(self):
        pass

    @abstractmethod
    def release(self):
        pass

    @property
    def debug(self):
        return self._debug

    @property
    def display_results(self):
        return self._display_results

    @debug.setter
    def debug(self, value):
        lastdebug, self._debug = self._debug, value
        if lastdebug != value and hasattr(self, 'debug_changed'):
            self.debug_changed(lastdebug, value)

    @display_results.setter
    def display_results(self, value):
        lastdisplay, self._display_results = self._display_results, value
        if lastdisplay != value and hasattr(self, 'display_results_changed'):
            self.display_results_changed(lastdisplay, value)