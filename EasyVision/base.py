# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod, abstractproperty


class NamedTupleExtendHelper(object):
    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, ", ".join("%s=%r" % field for field in zip(self._fields, self)))

    @classmethod
    def _make(cls, iterable, new=None, len=len):
        'Make a new object from a sequence or iterable'
        result = cls.__new__(cls, *iterable)
        if len(result) != len(cls._fields):
            raise TypeError('Expected %d arguments, got %d' % (len(cls._fields), len(result)))
        return result

    def _replace(self, **kwds):
        'Return a new object replacing specified fields with new values'
        result = self._make(map(kwds.pop, self._fields, self))
        if kwds:
            raise ValueError('Got unexpected field names: %r' % kwds.keys())
        return result


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

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @property
    def name(self):
        return self.__class__.__name__

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