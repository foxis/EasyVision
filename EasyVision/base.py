# -*- coding: utf-8 -*-
"""Base module containing Base and helper classes for all of EasyVision algorithms.

"""

from abc import ABCMeta, abstractmethod, abstractproperty

try:
    from functools import lru_cache
except ImportError:
    import functools

    class lru_cache(object):
        """Simple lru_cache analog for python 2.7 as python 3.x has it builtin"""
        def __init__(self, *args, **kwargs):
            self._cache = {}

        def __call__(self, func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                arg = "".join(str(i) for i in args)
                kwarg = "".join("{}={}".format(k, v) for k, v in kwargs.items())
                key = (arg, kwarg)
                if key in self._cache:
                    return self._cache[key]

                result = func(*args, **kwargs)
                self._cache[key] = result
                return result
            return wrapper


class NamedTupleExtendHelper(object):
    """NamedTupleExtendedHelper is a helper Mixin style class that enables to extend namedtuple derived classes with fields

    Implements:
        __repr__
            Builds a string representation of the object
        _make
            Make a new object from a sequence or iterable
        _replace
            Return a new object replacing specified fields with new values

    example

    .. code-block:: python

        MyBase = namedtuple('MyBase', 'a b')

        class MyNamedTuple(MyBase):
            _fields = MyBase._fields + ('c')
            __slots__ = ()

            def __new__(cls, *args, **kwargs):
                return super(MyNamedTuple, self).__new__(*args, **kwargs)

            c = property(itemgetter(2), doc="Alias for field number 3")
    """

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


class EasyVisionBase(metaclass=ABCMeta):
    """EasyVisionBase is an abstract class for all EasyVision algorithms
    Contains simple setup/release, debug/display_results, setup/release and context functionality.

    Uses __slots__ to conserve space, so derived classes can choose to continue using __slots__.

    Implements these:
        __enter__
            Implements a context manager
        __exit__
            Implements a context manager
        __next__
            Implements an iterator for Python 3.7
        __iter__
            Implements an iterator
        __len__
            Allows to receive a preliminary number of frames available
        name
            Returns Derived Class name

    Abstract methods:
        next
            For Python 2.7 compatibility. Implements iterator.
        description
            Must return a brief description of the algorithm.
        setup
            Must allocate resources.
        release
            Must deallocate resources.

    Optional methods:
        debug_changed
            Being called when debug property value is changed
        display_results_changed
            Being called when display_results property value is changed

    """

    __metaclass__ = ABCMeta
    __slots__ = ('_debug', '_display_results', '__setup_called')

    def __init__(self, debug=False, display_results=False, *args, **kwargs):
        """Initializes the instance. super must be called after all the derived class
        initialization is complete.

        :param debug: Indicates whether this algorithm should output debug info
        :param display_results: Indicates whether this algorithm should output results
        """
        self._debug = False
        self._display_results = False
        self.__setup_called = False
        self.debug = debug
        self.display_results = display_results
        super(EasyVisionBase, self).__init__()

    def __enter__(self):
        """Makes derived class into a context manager. Will call setup()."""
        self.setup()
        return self

    def __exit__(self, type, value, traceback):
        """Makes derived class into a context manager. Will call release()."""
        self.release()

    def __next__(self):
        """For Python 3.x support"""
        return self.next()

    def __iter__(self):
        """Makes derived class into iterable."""
        return self

    @abstractmethod
    def __len__(self):
        """Abstract method. Used for len(MyAlgorithm). Should return the number of images/frames/computations available.
        Will return negative numbers or zero for freerun capturing devices such as video camera.
        """
        pass

    @abstractmethod
    def next(self):
        """Abstract method. Legacy method for Python 2.7"""
        assert(self.__setup_called)

    @abstractmethod
    def setup(self):
        """Setup method that should be called before any call to __next__/capture/compute.

        This method should be used for resource allocation, algorithm initialization, opening of devices, etc.
        All derived classes must implement this method and call it using super after all the initialization is complete.
        This is being handled if using the algorithm with "with" statement.
        """
        assert(not self.__setup_called)
        self.__setup_called = True

    @abstractmethod
    def release(self):
        """Release method that should be called in order to release allocated resources.

        This method should be used for resource deallocation and cleaning up.
        All derived classes must implement this method and call it using super after all the deallocation is complete.
        This is being handled if using the algorithm with "with" statement.
        """
        assert(self.__setup_called)
        self.__setup_called = False

    @property
    def name(self):
        """Returns the name of the class."""
        return self.__class__.__name__

    @abstractproperty
    def description(self):
        """Abstract property. Should return a brief description of the algorithm."""
        pass

    @property
    def debug(self):
        """Property specifying whether to perform some outpuf for debugging purposes"""
        return self._debug

    @property
    def display_results(self):
        """Property specifying whether to display intermediate results"""
        return self._display_results

    @debug.setter
    def debug(self, value):
        """This property will call debug_changed if it finds one in the derived class."""
        lastdebug, self._debug = self._debug, value
        if lastdebug != value and hasattr(self, 'debug_changed'):
            self.debug_changed(lastdebug, value)

    @display_results.setter
    def display_results(self, value):
        """This property will call display_results_changed if it finds one in the derived class."""
        lastdisplay, self._display_results = self._display_results, value
        if lastdisplay != value and hasattr(self, 'display_results_changed'):
            self.display_results_changed(lastdisplay, value)
