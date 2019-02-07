# -*- coding: utf-8 -*-
"""Helper class for building processor stacks.

Example::

    builder = Builder(
        VideoCapture, Args(camera, width=640, height=480, fps=30),
        CalibratedCamera, Args(camera_model, display_results=True)
    )

    with builder.build() as vision:
        for frame in vision:
            if cv2.waitKey(1) == 27:
                break

Example using json:

.. code-block:: json

    {
        "args": [
            {
                "args": [0],
                "class": "VideoCapture",
                "objects": {},
                "kwargs": {"width":  1280, "height":  720}
            },
            {
                "args": [],
                "class": "CalibratedCamera",
                "objects": {
                    "object__PinholeCamera0": {
                        "rectify": null,
                        "distortion": [
                            [
                                0.07507829590903714,
                                0.2133670120228787,
                                0.004960489645345226,
                                -0.0019449662761104394,
                                -1.0317011493764785
                            ]
                        ],
                        "projection": null,
                        "matrix": [
                            [
                                732.8937676878295,
                                0.0,
                                311.31379638926603
                            ],
                            [
                                0.0,
                                728.1072411106162,
                                261.6539111360498
                            ],
                            [
                                0.0,
                                0.0,
                                1.0
                            ]
                        ],
                        "size": [
                            640,
                            480
                        ]
                    }
                },
                "kwargs": {"camera": "object__PinholeCamera0"}
            }
        ],
        "objects": {},
        "kwargs": {}
    }

.. code-block:: python

    classes = (
        ImagesReader,
        VideoCapture,
        PyroCapture,
        CalibratedCamera,
        CalibratedStereoCamera,
        PinholeCamera,
        StereoCamera,
        ImageTransform,
        FeatureExtraction,
        BackgroundSeparation,
        BlobExtraction
    )
    with open(json_file) as f:
        builder = Builder.fromdict(json.load(f), classes)

    with builder.build() as vision:
        for frame in vision:
            if cv2.waitKey(1) == 27:
                break


"""

from EasyVision.vision.base import VisionBase
from EasyVision.processors.base import ProcessorBase
from EasyVision.engine.base import EngineBase
import inspect


class Args(object):
    """Args class contains arguments for the vision/processor instance creation"""

    MISC_TYPES = (dict, tuple, list, set, frozenset)
    TYPES = (int, float, str, bytearray, bool, complex) + MISC_TYPES

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def todict(self):
        """Converts Args instance into a dictionary"""
        objects = {}

        objects, object_idx, args = Args.convert_args(objects, 0, self.args)
        objects, object_idx, kwargs = Args.convert_kwargs(objects, object_idx, self.kwargs)

        return {
            'args': args,
            'kwargs': kwargs,
            'objects': objects
        }

    @staticmethod
    def convert_args(objects, object_idx, _args):
        """Converts argument objects into object representation for todict"""
        args = ()
        for pos, arg in enumerate(_args):
            if isinstance(arg, object) and hasattr(arg, 'todict') and hasattr(arg, 'fromdict') and arg is not None:
                name = 'object__{}{}'.format(arg.__class__.__name__, object_idx)
                objects[name] = arg.todict()
                args += (name,)
                object_idx += 1
            elif not any(isinstance(arg, tp) for tp in Args.TYPES) and arg is not None:
                raise TypeError("Args only support class objects that implement todict/fromdict at position %i" % pos)
            else:
                args += (arg,)
        return objects, object_idx, args

    @staticmethod
    def convert_kwargs(objects, object_idx, _kwargs):
        """Converts kwargs objects into object representation for todict"""
        kwargs = {}
        for arg, value in _kwargs.items():
            if isinstance(value, object) and hasattr(value, 'todict') and hasattr(value, 'fromdict') and value is not None:
                name = 'object__{}{}'.format(value.__class__.__name__, object_idx)
                objects[name] = value.todict()
                kwargs[arg] = name
                object_idx += 1
            elif not any(isinstance(value, tp) for tp in Args.TYPES) and value is not None:
                raise TypeError("Args only support class objects that implement todict/fromdict at kwarg=%s" % arg)
            else:
                kwargs[arg] = value
        return objects, object_idx, kwargs

    @staticmethod
    def fromdict(d, classes=None):
        """Converts a dictionary into Args.
        If dictionary contains custom objects, then classes must be specified.

        :param d: Dictionary that was created using Args.todict()
        :param classes: A list of classes that custom objects can be created from
        :return: New Args object
        """
        classes = Args.convert_classes(classes)
        return Args(*tuple(Args._retrieve_object(classes, d['objects'], arg) for arg in d['args']),
                    **{arg: Args._retrieve_object(classes, d['objects'], value) for arg, value in d['kwargs'].items()})

    @staticmethod
    def convert_classes(classes):
        """Helper method to convert classes into lookup dictionary"""
        if isinstance(classes, dict):
            return classes
        return {cls.__name__: cls for cls in classes if hasattr(cls, 'todict') and hasattr(cls, 'fromdict')} if classes else {}

    @staticmethod
    def _retrieve_object(classes, objects, value):
        """Helper method to retrieve objects from their dict representation for fromdict"""
        if isinstance(value, str) and value.startswith('object__'):
            obj = objects[value]
            name = value[8:-1]
            return classes[name].fromdict(obj)
        else:
            return value


class Builder(object):
    """Builder class is a processor stack builder from arguments or dictionary.

    Note: first argument must always be a subclass of VisionBase. Even argument must be a subclass of ProcessorBase or Builder object. Odd argument must be a class of Args.

    Usage::

        builder = Builder(
            Builder(
                ImagesReader, Args("path/to/left_images", keyword_argument=True),
                ImageTransform, Args(color=cv2.COLOR_BGR2GRAY),
                CalibratedCamera, Args(camera=None),
                FeatureExtraction, Args(feature_type="ORB")
            ),
            Builder(
                ImagesReader, Args("path/to/right_images", keyword_argument=True),
                ImageTransform, Args(color=cv2.COLOR_BGR2GRAY),
                CalibratedCamera, Args(camera=None),
                FeatureExtraction, Args(feature_type="ORB")
            ),
            CalibratedStereoCamera, Args(camera=camera)
            display_results=True
        )

        with builder.build() as vision:
            for frame in vision:
                pass

    fromdict example::

        json = {
            'args': (
                {
                    'class': 'Builder',
                    'args': [
                        {
                            'class': 'ImageReader',
                            'args': ['path/to/images'],
                            'kwargs': {'keyword_argument': True}
                        }
                        {
                            'class': 'ImageTransform',
                            'kwargs': {'color': cv2.COLOR_BGR2GRAY}
                        }
                        {
                            'class': 'CalibratedCamera',
                            'kwargs': {'camera': null}
                        }
                        {
                            'class': 'FeatureExtraction',
                            'kwargs': {'feature_type': "ORB"}
                        }
                    ],
                    'kwargs': {},
                    'objects': {}
                },
                {
                    'class': 'Builder',
                    'args': [
                        {
                            'class': 'ImageReader',
                            'args': ['path/to/images'],
                            'kwargs': {'keyword_argument': True}
                        }
                        {
                            'class': 'ImageTransform',
                            'kwargs': {'color': cv2.COLOR_BGR2GRAY}
                        }
                        {
                            'class': 'CalibratedCamera',
                            'kwargs': {'camera': null}
                        }
                        {
                            'class': 'FeatureExtraction',
                            'kwargs': {'feature_type': "ORB"}
                        }
                    ],
                    'kwargs': {},
                    'objects': {}
                },
                {
                    'class': 'CalibratedStereoCamera',
                    'kwargs': {'camera': 'object__StereoCamera0'},
                    'objects': {
                        'object_StereoCamera0': {...}
                    }
                }
            ),
            'kwargs': {'display_results': True},
            'objects': {}
        ]
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def build(self):
        """Builds the processor stack using provided processors and their arguments"""
        index = 0
        args = ()
        cls = None
        for pos, arg in enumerate(self.args):
            if isinstance(arg, Builder):
                args += (arg.build(),)
            elif inspect.isclass(arg) and (issubclass(arg, VisionBase) or issubclass(arg, EngineBase)):
                if index and not inspect.isclass(arg) and not (issubclass(arg, ProcessorBase) or issubclass(arg, EngineBase)):
                    raise TypeError("Class at position %i must be either a subclass of ProcessorBase or EngineBase" % pos)
                cls = arg
                index += 1
            elif isinstance(arg, Args) and index % 2:
                if not cls:
                    raise ValueError("Either of Vision/Processor/Engine Base subclass must be provided before Args at position %i" % pos)
                default = {}
                default.update(arg.kwargs)
                default.update(self.kwargs)
                obj = cls(*(args + arg.args), **default)
                args = (obj, )
                index += 1
            else:
                raise ValueError("Invalid arguments at position: %i" % pos)
        assert(len(args) == 1)
        assert(isinstance(args[0], VisionBase) or isinstance(args[0], EngineBase))
        return args[0]

    def todict(self):
        """Transforms processor stack into a dictionary"""
        d = {'args': ()}
        cls = None
        for pos, arg in enumerate(self.args):
            if isinstance(arg, Builder):
                args = arg.todict()
                args.update({'class': arg.__class__.__name__})
                d['args'] += (args, )
            elif inspect.isclass(arg) and (issubclass(arg, VisionBase) or issubclass(arg, EngineBase)):
                cls = arg
            elif isinstance(arg, Args):
                if not cls:
                    raise ValueError("Either VisionBase or ProcessorBase subclass must be provided before Args at position %i" % pos)
                args = arg.todict()
                args.update({'class': cls.__name__})
                d['args'] += (args, )
                cls = None
            else:
                raise TypeError("Unsupported type at position %i" % pos)

            objects, _, kwargs = Args.convert_kwargs({}, 0, self.kwargs)
            d['kwargs'] = kwargs
            d['objects'] = objects
        return d

    @staticmethod
    def fromdict(d, classes=None):
        """Transforms a dictionary into Builder object"""
        args = ()
        _classes = Builder.convert_classes(classes)
        arg_classes = Args.convert_classes(classes)
        for pos, arg in enumerate(d['args']):
            if arg['class'] == Builder.__name__:
                args += (Builder.fromdict(arg, _classes),)
            else:
                args += (_classes[arg['class']], Args.fromdict(arg, arg_classes))

        kwargs = {}
        for arg, value in d['kwargs'].items():
            kwargs[arg] = Args._retrieve_object(arg_classes, d['objects'], value)

        return Builder(*args, **kwargs)

    @staticmethod
    def convert_classes(classes):
        """Helper method to convert classes into lookup dict"""
        if isinstance(classes, dict):
            return classes
        return {cls.__name__: cls for cls in classes} if classes else {}
